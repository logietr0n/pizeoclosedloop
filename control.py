"""
control.py — Displacement closed-loop control algorithms (pure numeric, no Tk).

Ported from /Users/nccer/Documents/meter:
  * SoftwarePID  <- +sim/runSim.m lines 421-434 (PID branch)
  * PreviewADRC  <- +sim/runSim.m lines 171-176, 236-245, 364-387
  * imc_tune     <- +sim/tunedPID.m + +sim/tunedADRC.m
  * fit_fopdt_step <- tune/system_id.py  _fit_fopdt()
  * SimulatedPlant <- +sim/runSim.m lines 272-282 (plant + dead-band + Bouc-Wen)

Everything here is deliberately free of Tk / MQTT / Moku imports so it can be
unit-tested headless (see test_control.py).

PLANT MODEL
-----------
First-order plus dead time (FOPDT), the same model meter identifies from a
single step response:

    dy/dt = (K * v_eff - y) / tau ,   v_eff = max(0, v - v_dead)

with the measurement delayed by theta_sensor and the actuator by theta_plant.
At the controller rate dt the exact (ZOH) discrete form is

    y[n] = a*y[n-1] + b*u[n-1-d],   a = exp(-dt/tau), b = K*(1-a), d = round(theta/dt)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

# ─────────────────────────────────────────────────────────────────────────────
# Plant model
# ─────────────────────────────────────────────────────────────────────────────

# Defaults follow meter/+sim/defaultParams.m lines 3-6, 19, 26.
DEFAULT_K_NM_PER_V   = 410.0
DEFAULT_TAU_S        = 0.080
DEFAULT_THETA_PLANT  = 0.010
DEFAULT_THETA_SENSOR = 0.0
DEFAULT_V_MAX        = 5.0
DEFAULT_V_DEAD       = 0.0


@dataclass
class PlantModel:
    """FOPDT piezo model plus actuator limits."""

    K_nm_per_V:     float = DEFAULT_K_NM_PER_V
    tau_s:          float = DEFAULT_TAU_S
    theta_plant_s:  float = DEFAULT_THETA_PLANT
    theta_sensor_s: float = DEFAULT_THETA_SENSOR
    v_max:          float = DEFAULT_V_MAX
    v_dead:         float = DEFAULT_V_DEAD

    def __post_init__(self) -> None:
        if self.K_nm_per_V <= 0.0:
            raise ValueError("K must be > 0 nm/V")
        if self.tau_s <= 0.0:
            raise ValueError("tau must be > 0 s")
        if self.v_max <= 0.0:
            raise ValueError("v_max must be > 0 V")

    def discrete(self, dt: float) -> tuple[float, float, int]:
        """(a, b, d) of y[n] = a*y[n-1] + b*u[n-1-d] at controller period dt."""
        a = math.exp(-dt / self.tau_s)
        b = self.K_nm_per_V * (1.0 - a)
        d = max(0, round(self.theta_plant_s / dt))
        return a, b, d

    @property
    def stroke_nm(self) -> float:
        """Full-scale displacement at v_max (useful for sanity-checking a setpoint)."""
        return self.K_nm_per_V * max(0.0, self.v_max - self.v_dead)


# ─────────────────────────────────────────────────────────────────────────────
# Reference trajectory
# ─────────────────────────────────────────────────────────────────────────────

WAVEFORMS = ("DC", "Sine", "Square", "Ramp", "Pulse")


class TrajectoryGenerator:
    """
    Closed-form reference r(t) in nm.

    Closed-form (rather than meter's precomputed spArr, +sim/makeSetpoint.m) is
    what makes preview possible here: the preview feedforward needs r at
    t + d*dt and t + (d+1)*dt, and a formula can be evaluated at any future
    instant with no lookahead buffer at all.

    It also means Python owns the time origin. The Moku waveform generator's
    absolute phase is not observable from Python, so the reference must be
    synthesised here rather than read back from the FPGA.
    """

    def __init__(
        self,
        waveform: str = "Sine",
        amplitude_nm: float = 200.0,
        frequency_hz: float = 0.5,
        offset_nm: float = 0.0,
        phase_deg: float = 0.0,
        duty_pct: float = 50.0,
        symmetry_pct: float = 50.0,
    ) -> None:
        if waveform not in WAVEFORMS:
            raise ValueError(f"unsupported waveform {waveform!r}; choose from {WAVEFORMS}")
        self.waveform     = waveform
        self.amplitude_nm = float(amplitude_nm)
        self.frequency_hz = float(frequency_hz)
        self.offset_nm    = float(offset_nm)
        self.phase_deg    = float(phase_deg)
        self.duty         = min(max(float(duty_pct) / 100.0, 0.0), 1.0)
        self.symmetry     = min(max(float(symmetry_pct) / 100.0, 0.0), 1.0)

    def r(self, t: float) -> float:
        """Reference displacement (nm) at absolute loop time t (s)."""
        A, o = self.amplitude_nm, self.offset_nm

        if self.waveform == "DC" or self.frequency_hz <= 0.0:
            return o + (A if self.waveform == "DC" else 0.0)

        # Normalised phase within one period, in [0, 1)
        ph = (t * self.frequency_hz + self.phase_deg / 360.0) % 1.0

        if self.waveform == "Sine":
            return o + A * math.sin(2.0 * math.pi * ph)

        if self.waveform == "Square":
            return o + (A if ph < self.duty else -A)

        if self.waveform == "Ramp":
            # Symmetric triangle: `symmetry` sets the fraction spent rising.
            s = self.symmetry
            if s <= 0.0:
                return o + A * (1.0 - 2.0 * ph)
            if s >= 1.0:
                return o + A * (2.0 * ph - 1.0)
            if ph < s:
                return o + A * (2.0 * (ph / s) - 1.0)
            return o + A * (1.0 - 2.0 * ((ph - s) / (1.0 - s)))

        # Pulse: rests at the offset, rises to offset + A for `duty` of the period.
        return o + (A if ph < self.duty else 0.0)

    def bandwidth_warning(self, model: PlantModel) -> bool:
        """True if the reference is faster than the plant's own -3 dB bandwidth."""
        bw_hz = 1.0 / (2.0 * math.pi * model.tau_s)
        return self.frequency_hz > bw_hz


@dataclass
class FeedforwardCommand:
    """One Moku waveform configuration computed from a desired displacement.

    The computer sends this once. The Moku plays it. Sample-by-sample
    correction stays in the UMD2; this object is not a feedback law.
    """

    amplitude_vpp: float
    offset_v: float
    phase_deg: float
    boost: float


def feedforward_command(
    waveform: str,
    amplitude_nm: float,
    offset_nm: float,
    frequency_hz: float,
    model: PlantModel,
    phase_deg: float = 0.0,
    duty_pct: float = 50.0,
) -> FeedforwardCommand:
    """Convert a desired displacement shape into one Moku voltage waveform.

    Sine, square, and ramp are symmetric about the offset, so the Moku
    peak-to-peak voltage is ``2 * amplitude_nm / K``. A sine is also
    pre-emphasized by the first-order plant inverse: amplitude grows by
    ``sqrt(1+(ωτ)²)`` and the phase leads by ``atan(ωτ)``. Square, ramp,
    pulse, and noise stay a static scale, because those shapes cannot carry
    that inverse and still be the waveform the Moku generator knows.

    Pulse rests at ``offset_nm`` and rises by ``amplitude_nm``. The Moku
    pulse is symmetric about its offset, so the voltage offset is shifted
    up by half the pulse height.
    """
    if model.K_nm_per_V <= 0.0:
        raise ValueError("K must be > 0 nm/V")
    if amplitude_nm < 0.0:
        raise ValueError("amplitude_nm must be >= 0")

    scale = 1.0 / model.K_nm_per_V
    boost = 1.0
    phase = float(phase_deg) % 360.0
    offset_v = float(offset_nm) * scale

    if waveform == "Sine" and frequency_hz > 0.0 and model.tau_s > 0.0:
        w_tau = 2.0 * math.pi * float(frequency_hz) * model.tau_s
        boost = math.sqrt(1.0 + w_tau * w_tau)
        phase = (phase + math.degrees(math.atan(w_tau))) % 360.0

    if waveform == "Pulse":
        height_v = float(amplitude_nm) * scale
        return FeedforwardCommand(
            amplitude_vpp=height_v,
            offset_v=offset_v + 0.5 * height_v,
            phase_deg=phase,
            boost=boost,
        )

    if waveform == "Noise":
        return FeedforwardCommand(
            amplitude_vpp=float(amplitude_nm) * scale,
            offset_v=0.0,
            phase_deg=0.0,
            boost=1.0,
        )

    return FeedforwardCommand(
        amplitude_vpp=2.0 * float(amplitude_nm) * scale * boost,
        offset_v=offset_v,
        phase_deg=phase,
        boost=boost,
    )


def expected_mean_nm(
    waveform: str,
    offset_nm: float,
    amplitude_nm: float,
    duty_pct: float = 50.0,
) -> float:
    """Displacement mean a symmetric wave, or a pulse, should sit at."""
    duty = min(max(float(duty_pct) / 100.0, 0.0), 1.0)
    if waveform == "Pulse":
        return float(offset_nm) + float(amplitude_nm) * duty
    if waveform == "Square":
        return float(offset_nm) + float(amplitude_nm) * (2.0 * duty - 1.0)
    return float(offset_nm)


def feedforward_trim(
    waveform: str,
    offset_nm: float,
    amplitude_nm: float,
    samples_nm: list[float],
    offset_tol_nm: float,
    amplitude_tol_frac: float,
    duty_pct: float = 50.0,
) -> tuple[float, float, bool]:
    """Nudge the feedforward command when the measured trace is shifted or scaled.

    Returns ``(offset_nm, amplitude_nm, changed)``. One call is one update of
    the command that will be sent again; it is not a per-sample controller.
    Noise has no stable amplitude, so only its offset is considered, and a
    pulse's mean is compared with ``offset + amplitude * duty``.
    """
    if len(samples_nm) < 8:
        return float(offset_nm), float(amplitude_nm), False

    measured_mean = float(sum(samples_nm) / len(samples_nm))
    target_mean = expected_mean_nm(waveform, offset_nm, amplitude_nm, duty_pct)
    new_offset = float(offset_nm)
    new_amp = float(amplitude_nm)
    changed = False

    if abs(measured_mean - target_mean) > float(offset_tol_nm):
        new_offset = float(offset_nm) + (target_mean - measured_mean)
        changed = True

    if waveform not in ("Noise", "Pulse") and amplitude_nm > 0.0:
        measured_amp = 0.5 * (max(samples_nm) - min(samples_nm))
        if measured_amp > 1.0:
            ratio = measured_amp / float(amplitude_nm)
            if abs(ratio - 1.0) > float(amplitude_tol_frac):
                new_amp = float(amplitude_nm) / ratio
                changed = True

    if new_amp < 0.0:
        new_amp = 0.0
    return new_offset, new_amp, changed


# ─────────────────────────────────────────────────────────────────────────────
# Controllers
# ─────────────────────────────────────────────────────────────────────────────

class SoftwarePID:
    """
    Discrete PI(D) with anti-windup and a filtered derivative on the
    measurement. Ported from meter/+sim/runSim.m lines 421-434 (which is the
    richer of the two meter PID implementations -- meter/control/pid.py lacks
    the derivative filter).

    Derivative acts on -y rather than on e so that a setpoint step does not
    produce a derivative kick.
    """

    def __init__(
        self,
        model: PlantModel,
        kp: float,
        ki: float,
        kd: float = 0.0,
        d_filter_n: float = 20.0,
    ) -> None:
        self.model      = model
        self.kp         = float(kp)
        self.ki         = float(ki)
        self.kd         = float(kd)
        self.d_filter_n = float(d_filter_n)
        self.reset()

    name = "Software PID"

    def reset(self) -> None:
        self._integral = 0.0
        self._d_filt   = 0.0
        self._prev_y   = None   # None => no derivative on the very first tick
        self.last_v    = 0.0

    def update(self, setpoint_nm: float, y_meas_nm: float, dt: float) -> float:
        err = setpoint_nm - y_meas_nm

        self._integral += err * dt
        if self.ki > 0.0:
            # Anti-windup: the integral alone may not exceed the whole voltage
            # budget (runSim.m:423-426). Unlike Preview there is no feedforward
            # term competing for that budget, so v_max is the right scope.
            lim = self.model.v_max / self.ki
            self._integral = min(max(self._integral, -lim), lim)

        n = self.d_filter_n
        if n > 0.0 and self.kd != 0.0 and self._prev_y is not None:
            self._d_filt = (
                self._d_filt / (1.0 + n * dt)
                + self.kd * n / (1.0 + n * dt) * (self._prev_y - y_meas_nm)
            )
        else:
            self._d_filt = 0.0
        self._prev_y = y_meas_nm

        v = self.kp * err + self.ki * self._integral + self._d_filt
        self.last_v = min(max(v, 0.0), self.model.v_max)
        return self.last_v

    def diagnostics(self) -> dict[str, float]:
        return {"integral": self._integral, "d_filt": self._d_filt}


class PreviewADRC:
    """
    Preview feedforward fused with a disturbance-and-rate observer.
    Ported from meter/+sim/runSim.m lines 171-176, 236-245, 364-387.

    Despite the name this is NOT Katayama/Tomizuka preview-LQ control: there is
    no Riccati solve and no preview-gain window. "Preview" here means exact
    one-step inversion of the discrete FOPDT model evaluated against a known
    future setpoint, so the preview horizon is exactly theta_plant + dt.

        uFF     = (r[k+d+1] - a*r[k+d]) / b
        errTrim = y_model_delayed - y_meas
        [res1; res2] <- Ad_res @ [res1; res2] + Bd_res * errTrim
        v = clip(uFF + res1/K, 0, v_max)

    THREE THINGS THAT MUST NOT BE "SIMPLIFIED" (runSim.m:215-235):

    1. The observer input matrix is a single column [2w0; w0^2] -- there is no
       b0*u term. The residual is provably not driven by u: a trim voltage dV
       moves the reference model and the real plant by the same K*dV, so it
       cancels out of the difference. Only the exogenous disturbance drives
       this channel.

    2. The correction is res1/K, using the STATIC gain -- not res2/b0 and not
       ADRC's z2/b0. A conventional ESO must lump the plant's own -y/tau
       relaxation into z2 because it has no other way to know it; the
       feedforward here has already removed that relaxation exactly, so
       subtracting it a second time destabilises the loop. Hysteresis is an
       additive disturbance, so cancelling res1 nm of it takes exactly
       res1/K volts.

    3. errTrim compares against the delay-aligned reference model, not against
       the raw setpoint. Using (sp - y) makes the trim double-correct the
       transient the feedforward is already closing: meter measured edge
       overshoot going from ~2% to ~19%.

    Only adrc_w0 matters in this law; there is no wc term and no sp_dot term.
    """

    name = "Preview + ADRC"

    def __init__(self, model: PlantModel, w0: float, dt: float) -> None:
        if w0 <= 0.0:
            raise ValueError("w0 must be > 0 rad/s")
        self.model = model
        self.w0    = float(w0)
        self.dt    = float(dt)

        # Feedforward constants at the controller rate (runSim.m:171-176)
        self.a, self.b, self.d = model.discrete(dt)

        # Disturbance-and-rate observer, ZOH-exact (runSim.m:236-245).
        # Continuous:  res1' = res2 + 2*w0*(e - res1),  res2' = w0^2*(e - res1)
        Ac = np.array([[-2.0 * self.w0, 1.0],
                       [-self.w0 ** 2,  0.0]])
        Bc = np.array([[2.0 * self.w0],
                       [self.w0 ** 2]])
        Maug = np.block([[Ac, Bc], [np.zeros((1, 3))]])
        Phi = expm(Maug * dt)
        self.Ad_res = Phi[:2, :2]
        self.Bd_res = Phi[:2, 2]

        # Reference model runs at the controller rate using the exact discrete
        # FOPDT form. meter integrated it with 1us forward Euler (runSim.m:313)
        # purely because its plant loop already ran at 1us; the closed form is
        # the same ZOH solution without 50k Euler steps per tick.
        self._sensor_delay_steps = max(0, round(model.theta_sensor_s / dt))

        self.reset()

    def reset(self) -> None:
        self._res = np.zeros(2)                                    # [disturbance nm, its rate]
        self._y_model = 0.0                                        # dead-time-free reference model
        self._delay_buf = [0.0] * self._sensor_delay_steps         # sensor-delay alignment
        self.last_v = 0.0
        self._last_uff = 0.0
        self._last_err_trim = 0.0

    def preview_times(self, t: float) -> tuple[float, float]:
        """The two future instants the feedforward needs: t+d*dt and t+(d+1)*dt."""
        return t + self.d * self.dt, t + (self.d + 1) * self.dt

    def update(
        self,
        r_at_d: float,
        r_at_d1: float,
        y_meas_nm: float,
        dt: float | None = None,
    ) -> float:
        """
        One controller tick.

        r_at_d  : reference at t + d*dt
        r_at_d1 : reference at t + (d+1)*dt
        y_meas  : measured displacement (nm)
        dt      : measured tick period; only used for the reference model. The
                  observer and feedforward matrices stay at the nominal dt they
                  were discretised for (recomputing expm every tick would be
                  both slow and jittery).
        """
        K = self.model.K_nm_per_V
        step_dt = self.dt if dt is None else dt

        # 1. Exact-inversion feedforward from the known future setpoint.
        uFF = (r_at_d1 - self.a * r_at_d) / self.b

        # 2. Advance the reference model with the voltage actually applied last
        #    tick. Deliberately no dead-band compensation -- matching the
        #    feedforward -- so dead-band mismatch shows up in the residual as a
        #    genuine model error instead of being hidden (runSim.m:308-315).
        a_m = math.exp(-step_dt / self.model.tau_s)
        b_m = K * (1.0 - a_m)
        self._y_model = a_m * self._y_model + b_m * self.last_v

        # 3. Delay the model the same way the sensor delays the measurement.
        if self._sensor_delay_steps > 0:
            y_model_delayed = self._delay_buf.pop(0)
            self._delay_buf.append(self._y_model)
        else:
            y_model_delayed = self._y_model

        # 4. Residual -> observer. Sign is model minus measurement.
        err_trim = y_model_delayed - y_meas_nm
        self._res = self.Ad_res @ self._res + self.Bd_res * err_trim

        # 5. Proportional, self-limiting correction via the static gain.
        v = uFF + self._res[0] / K

        self._last_uff = uFF
        self._last_err_trim = err_trim
        self.last_v = float(min(max(v, 0.0), self.model.v_max))
        return self.last_v

    def diagnostics(self) -> dict[str, float]:
        return {
            "uff": self._last_uff,
            "err_trim": self._last_err_trim,
            "res1": float(self._res[0]),
            "res2": float(self._res[1]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────

def imc_tune(
    model: PlantModel,
    dt: float,
    fs_sample_hz: float = 610.35,
) -> dict[str, float]:
    """
    IMC auto-tune. Ported from meter/+sim/tunedPID.m and +sim/tunedADRC.m,
    which extend +sim/imcTune.m with a 0.5/fs term for the sensor's own
    zero-order-hold lag.

    Returns kp, ki, kd, d_filter_n for the PID and w0 for Preview+ADRC.

    Note the two theta_eff differ: the PID must live with the plant dead time,
    while the preview feedforward inverts it away, so theta_plant drops out of
    the observer bandwidth.
    """
    tau     = model.tau_s
    theta_p = model.theta_plant_s
    theta_s = model.theta_sensor_s
    zoh_fs  = 0.5 / fs_sample_hz if fs_sample_hz > 0.0 else 0.0

    # ── PID (tunedPID.m) ────────────────────────────────────────────────
    theta_eff = theta_p + theta_s + dt / 2.0 + zoh_fs
    lam   = 2.0 * theta_eff
    denom = tau + theta_eff / 2.0
    kp = denom / (model.K_nm_per_V * (lam + theta_eff / 2.0))
    ki = kp / denom

    # kd / N follow imcTune.m (meter/tune/imc.py:96-98): plant theta only.
    theta_pd = max(theta_p, 1e-9)
    kd = kp * tau * theta_pd / (2.0 * tau + theta_pd)
    d_filter_n = float(round((2.0 * tau + theta_pd) / theta_pd))

    # ── Preview+ADRC observer (tunedADRC.m) ─────────────────────────────
    # theta_plant is absent: the feedforward predicts it away.
    theta_eff_adrc = theta_s + dt / 2.0 + zoh_fs
    wc = 1.0 / max(tau + theta_eff_adrc, 1e-9)
    w0 = 5.0 * wc

    return {"kp": kp, "ki": ki, "kd": kd, "d_filter_n": d_filter_n, "w0": w0}


def fit_fopdt_step(
    t: np.ndarray,
    y_nm: np.ndarray,
    delta_v: float,
    y0_nm: float | None = None,
) -> tuple[float, float, float]:
    """
    Fit K, tau, theta from one step response. Ported from the fitting half of
    meter/tune/system_id.py:_fit_fopdt (its hardware-driving half depends on
    meter's UMD2Reader and is not portable here).

        y(t) = y0 + K*dV*(1 - exp(-(t-theta)/tau))  for t > theta,  else y0

    t and y_nm must start at the instant the step was commanded.
    Returns (K_nm_per_V, tau_s, theta_s).
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    y_nm = np.asarray(y_nm, dtype=float)
    if t.size < 8 or t.size != y_nm.size:
        raise ValueError("need at least 8 matching (t, y) samples to fit a step")
    if abs(delta_v) < 1e-6:
        raise ValueError("delta_v is ~0 V; cannot identify a gain from it")

    y0 = float(y_nm[0]) if y0_nm is None else float(y0_nm)

    def model(t_arr, K, tau, theta):
        tau = max(tau, 1e-6)
        return np.where(
            t_arr > theta,
            y0 + K * delta_v * (1.0 - np.exp(-(t_arr - theta) / tau)),
            y0,
        )

    tail = max(len(y_nm) // 5, 10)
    y_inf = float(np.mean(y_nm[max(len(y_nm) - tail, 0):]))
    K0 = (y_inf - y0) / delta_v if abs(delta_v) > 0.01 else DEFAULT_K_NM_PER_V
    p0 = [K0 if K0 > 1.0 else DEFAULT_K_NM_PER_V,
          float(t[-1]) / 4.0,
          min(0.02, float(t[-1]) / 10.0)]

    popt, _ = curve_fit(
        model, t, y_nm,
        p0=p0,
        bounds=([1.0, 1e-4, 0.0], [5000.0, 60.0, 10.0]),
        maxfev=8000,
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


# ─────────────────────────────────────────────────────────────────────────────
# Simulation (for headless testing — no hardware required)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedPlant:
    """
    FOPDT + dead-band + optional Bouc-Wen hysteresis, stepped at the controller
    rate. Ported from meter/+sim/runSim.m lines 272-282.

    meter runs its plant at a fixed 1 us substep; here the substep is the
    controller period because that is the only rate this project can actually
    command the Moku at.
    """

    model: PlantModel
    noise_nm: float = 0.0
    bw_enable: bool = False
    bw_A: float = 1.0
    bw_beta: float = 0.5
    bw_gamma: float = 0.5
    bw_D: float = 30.0
    seed: int | None = 0

    _y: float = field(default=0.0, init=False)
    _z_bw: float = field(default=0.0, init=False)
    _prev_v_eff: float = field(default=0.0, init=False)
    _plant_buf: list = field(default_factory=list, init=False)
    _sensor_buf: list = field(default_factory=list, init=False)
    _rng: np.random.Generator = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, dt: float) -> None:
        self._y = 0.0
        self._z_bw = 0.0
        self._prev_v_eff = 0.0
        self._plant_buf = [0.0] * max(0, round(self.model.theta_plant_s / dt))
        self._sensor_buf = [0.0] * max(0, round(self.model.theta_sensor_s / dt))
        self._rng = np.random.default_rng(self.seed)

    def step(self, v_cmd: float, dt: float, disturbance_nm: float = 0.0) -> tuple[float, float]:
        """Apply v_cmd for dt. Returns (y_true_nm, y_measured_nm)."""
        m = self.model

        # Actuator dead time
        if self._plant_buf:
            v_plant = self._plant_buf.pop(0)
            self._plant_buf.append(v_cmd)
        else:
            v_plant = v_cmd

        v_eff = max(0.0, v_plant - m.v_dead)

        # Bouc-Wen hysteresis (n=1), driven by the voltage *change*
        du = v_eff - self._prev_v_eff
        if self.bw_enable:
            self._z_bw += (
                self.bw_A * du
                - self.bw_beta * abs(du) * self._z_bw
                - self.bw_gamma * du * abs(self._z_bw)
            )
            hyst_nm = -self.bw_D * self._z_bw
        else:
            hyst_nm = 0.0
        self._prev_v_eff = v_eff

        # Exact discrete FOPDT state update
        a = math.exp(-dt / m.tau_s)
        self._y = a * self._y + m.K_nm_per_V * (1.0 - a) * v_eff

        y_true = self._y + hyst_nm + disturbance_nm

        # Sensor dead time, then noise
        if self._sensor_buf:
            y_sensed = self._sensor_buf.pop(0)
            self._sensor_buf.append(y_true)
        else:
            y_sensed = y_true
        if self.noise_nm > 0.0:
            y_sensed += float(self._rng.normal(0.0, self.noise_nm))

        return y_true, y_sensed


def simulate(
    controller,
    plant: SimulatedPlant,
    traj: TrajectoryGenerator,
    dt: float,
    duration_s: float,
) -> dict[str, np.ndarray]:
    """
    Run a controller against a SimulatedPlant. Mirrors how closed_loop.py's
    worker drives the real hardware, so a controller that works here is wired
    the same way as one that works on the bench.
    """
    n = int(round(duration_s / dt))
    plant.reset(dt)
    controller.reset()

    t_arr = np.zeros(n)
    r_arr = np.zeros(n)
    y_arr = np.zeros(n)
    v_arr = np.zeros(n)

    y_meas = 0.0
    y_true = 0.0
    for k in range(n):
        t = k * dt
        r = traj.r(t)

        if isinstance(controller, PreviewADRC):
            t_d, t_d1 = controller.preview_times(t)
            v = controller.update(traj.r(t_d), traj.r(t_d1), y_meas, dt)
        else:
            v = controller.update(r, y_meas, dt)

        # Log the state as it is AT t, before this tick's voltage takes effect.
        # plant.step returns the state at t+dt, so logging its return value here
        # would pair r(t) with y(t+dt) and credit the controller with one free
        # sample of lookahead.
        t_arr[k], r_arr[k], y_arr[k], v_arr[k] = t, r, y_true, v

        y_true, y_meas = plant.step(v, dt)

    err = r_arr - y_arr
    return {
        "t": t_arr, "r": r_arr, "y": y_arr, "v": v_arr, "e": err,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(np.abs(err))),
        "iae": float(np.sum(np.abs(err)) * dt),
    }
