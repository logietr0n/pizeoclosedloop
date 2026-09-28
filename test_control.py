"""
Headless tests for control.py — no hardware, no Tk, no MQTT.

Run:  python3 -m pytest test_control.py -v
  or: python3 test_control.py

The Preview-vs-PID acceptance criterion mirrors meter/test_simulate.m:192
(`test_runSim_Preview_exactModel_tracksBetterThanPID`): with an exact model and
no noise, the preview feedforward must track substantially better than PID.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

from control import (
    PlantModel,
    PreviewADRC,
    SimulatedPlant,
    SoftwarePID,
    TrajectoryGenerator,
    feedforward_command,
    feedforward_trim,
    fit_fopdt_step,
    imc_tune,
    simulate,
)

DT = 0.01
DURATION = 6.0

# The piezo is unipolar: v is clamped to [0, v_max], so y can only be >= 0. A
# reference must therefore sit on a positive offset to be reachable at all --
# a sine centred on 0 nm would demand negative displacement for half of every
# cycle and no controller could track it.
REF_OFFSET_NM = 800.0
REF_AMP_NM = 400.0


def tracking_traj(freq_hz: float = 0.5) -> TrajectoryGenerator:
    return TrajectoryGenerator("Sine", amplitude_nm=REF_AMP_NM,
                               frequency_hz=freq_hz, offset_nm=REF_OFFSET_NM)


def settled_rmse(out, frac: float = 0.5) -> float:
    """
    RMSE over the last `1-frac` of the run.

    The piezo starts at 0 nm while the reference starts at REF_OFFSET_NM, so
    every run opens with an unavoidable approach transient governed by the
    plant's own time constant. Steady-state tracking is what distinguishes the
    control laws, so the transient is excluded -- the same split meter uses
    when it reports convergence and tracking as separate protocols.
    """
    e = out["e"]
    return float(np.sqrt(np.mean(e[int(len(e) * frac):] ** 2)))


def make_model(**kw) -> PlantModel:
    base = dict(K_nm_per_V=410.0, tau_s=0.08, theta_plant_s=0.0,
                theta_sensor_s=0.0, v_max=5.0, v_dead=0.0)
    base.update(kw)
    return PlantModel(**base)


def tuned_pair(model: PlantModel, dt: float = DT):
    g = imc_tune(model, dt)
    pid = SoftwarePID(model, kp=g["kp"], ki=g["ki"], kd=g["kd"], d_filter_n=g["d_filter_n"])
    adrc = PreviewADRC(model, w0=g["w0"], dt=dt)
    return pid, adrc


# ─────────────────────────────────────────────────────────────────────────────
# PlantModel
# ─────────────────────────────────────────────────────────────────────────────

def test_plant_model_rejects_nonsense():
    for bad in (dict(K_nm_per_V=0.0), dict(tau_s=0.0), dict(v_max=-1.0)):
        with pytest.raises(ValueError):
            make_model(**bad)


def test_discrete_matches_zoh_solution():
    m = make_model(tau_s=0.08, theta_plant_s=0.02)
    a, b, d = m.discrete(0.01)
    assert a == pytest.approx(math.exp(-0.01 / 0.08))
    assert b == pytest.approx(410.0 * (1 - a))
    assert d == 2                       # round(0.02/0.01)


def test_stroke_accounts_for_dead_band():
    assert make_model(v_dead=1.0).stroke_nm == pytest.approx(410.0 * 4.0)


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryGenerator
# ─────────────────────────────────────────────────────────────────────────────

def test_sine_is_closed_form_and_periodic():
    tr = TrajectoryGenerator("Sine", amplitude_nm=100.0, frequency_hz=2.0)
    assert tr.r(0.0) == pytest.approx(0.0, abs=1e-9)
    assert tr.r(0.125) == pytest.approx(100.0)       # quarter period
    assert tr.r(3.7) == pytest.approx(tr.r(3.7 + 0.5))

def test_arbitrary_future_time_is_evaluable():
    """The whole point of a closed form: no lookahead buffer, no horizon limit."""
    tr = TrajectoryGenerator("Sine", amplitude_nm=100.0, frequency_hz=1.0)
    assert tr.r(1e6) == pytest.approx(tr.r(0.0), abs=1e-6)


def test_square_respects_duty():
    tr = TrajectoryGenerator("Square", amplitude_nm=50.0, frequency_hz=1.0, duty_pct=25.0)
    assert tr.r(0.1) == pytest.approx(50.0)
    assert tr.r(0.5) == pytest.approx(-50.0)


def test_ramp_is_continuous_triangle():
    tr = TrajectoryGenerator("Ramp", amplitude_nm=100.0, frequency_hz=1.0, symmetry_pct=50.0)
    assert tr.r(0.0) == pytest.approx(-100.0)
    assert tr.r(0.5) == pytest.approx(100.0)
    assert tr.r(0.25) == pytest.approx(0.0)


def test_pulse_rests_at_offset():
    tr = TrajectoryGenerator("Pulse", amplitude_nm=80.0, frequency_hz=1.0,
                             offset_nm=10.0, duty_pct=20.0)
    assert tr.r(0.1) == pytest.approx(90.0)
    assert tr.r(0.5) == pytest.approx(10.0)


def test_zero_frequency_holds_static():
    tr = TrajectoryGenerator("Sine", amplitude_nm=100.0, frequency_hz=0.0, offset_nm=42.0)
    assert tr.r(0.0) == pytest.approx(42.0)
    assert tr.r(99.0) == pytest.approx(42.0)


def test_bandwidth_warning_fires_above_plant_bandwidth():
    m = make_model(tau_s=0.08)          # ~2 Hz
    assert TrajectoryGenerator("Sine", frequency_hz=50.0).bandwidth_warning(m)
    assert not TrajectoryGenerator("Sine", frequency_hz=0.1).bandwidth_warning(m)


def test_rejects_unknown_waveform():
    with pytest.raises(ValueError):
        TrajectoryGenerator("Sawtooth")


# ─────────────────────────────────────────────────────────────────────────────
# Controller invariants (meter/test_simulate.m:171,181)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("which", ["pid", "adrc"])
def test_output_stays_in_voltage_bounds_and_finite(which):
    model = make_model()
    pid, adrc = tuned_pair(model)
    ctrl = pid if which == "pid" else adrc
    traj = tracking_traj(freq_hz=1.0)

    out = simulate(ctrl, SimulatedPlant(model), traj, DT, DURATION)

    assert np.all(np.isfinite(out["v"])), "voltage went NaN/inf"
    assert np.all(np.isfinite(out["y"])), "displacement went NaN/inf"
    assert out["v"].min() >= -1e-12
    assert out["v"].max() <= model.v_max + 1e-12


@pytest.mark.parametrize("which", ["pid", "adrc"])
def test_saturating_setpoint_does_not_destabilise(which):
    """Command far beyond the reachable stroke; output must clamp, not diverge."""
    model = make_model()
    pid, adrc = tuned_pair(model)
    ctrl = pid if which == "pid" else adrc
    traj = TrajectoryGenerator("DC", amplitude_nm=10 * model.stroke_nm, frequency_hz=0.0)

    out = simulate(ctrl, SimulatedPlant(model), traj, DT, DURATION)

    assert np.all(np.isfinite(out["v"]))
    assert out["v"].max() <= model.v_max + 1e-12
    assert out["y"].max() <= model.stroke_nm * 1.05


# ─────────────────────────────────────────────────────────────────────────────
# The headline claim (meter/test_simulate.m:192)
# ─────────────────────────────────────────────────────────────────────────────

def test_preview_adrc_beats_pid_on_tracking_with_exact_model():
    model = make_model()
    pid, adrc = tuned_pair(model)
    traj = tracking_traj()

    pid_out = simulate(pid, SimulatedPlant(model), traj, DT, 8.0)
    adrc_out = simulate(adrc, SimulatedPlant(model), traj, DT, 8.0)

    assert adrc_out["iae"] < pid_out["iae"], (
        f"preview IAE {adrc_out['iae']:.1f} not better than PID {pid_out['iae']:.1f}"
    )
    assert settled_rmse(adrc_out) < 0.25 * settled_rmse(pid_out), (
        f"preview settled RMSE {settled_rmse(adrc_out):.2f} nm "
        f"vs PID {settled_rmse(pid_out):.2f} nm"
    )


def test_preview_adrc_rejects_hysteresis_disturbance():
    """The residual observer's whole purpose: cancel an additive disturbance."""
    model = make_model()
    pid, adrc = tuned_pair(model)
    traj = tracking_traj()

    adrc_out = simulate(adrc, SimulatedPlant(model, bw_enable=True, bw_D=100.0), traj, DT, 12.0)
    pid_out = simulate(pid, SimulatedPlant(model, bw_enable=True, bw_D=100.0), traj, DT, 12.0)

    # The observer estimates the additive disturbance and cancels it with
    # res1/K volts. The PI integrator cannot: it chases a residual that barely
    # responds to its own correction (meter/docs_why_preview_pid_diverges.md).
    assert settled_rmse(adrc_out) < 40.0, f"{settled_rmse(adrc_out):.1f} nm"
    assert settled_rmse(adrc_out) < 0.3 * settled_rmse(pid_out)


def test_preview_adrc_converges_on_a_step():
    model = make_model()
    _, adrc = tuned_pair(model)
    traj = TrajectoryGenerator("DC", amplitude_nm=500.0, frequency_hz=0.0)

    out = simulate(adrc, SimulatedPlant(model), traj, DT, DURATION)

    assert abs(out["e"][-1]) < 5.0, f"final error {out['e'][-1]:.2f} nm"


def test_both_controllers_track_a_static_setpoint():
    model = make_model()
    pid, adrc = tuned_pair(model)
    traj = TrajectoryGenerator("DC", amplitude_nm=500.0, frequency_hz=0.0)

    for ctrl in (pid, adrc):
        out = simulate(ctrl, SimulatedPlant(model), traj, DT, 15.0)
        assert abs(out["e"][-1]) < 20.0, f"{ctrl.name} settled at {out['e'][-1]:.2f} nm"


# ─────────────────────────────────────────────────────────────────────────────
# Preview structure — the three things that must not be "simplified"
# ─────────────────────────────────────────────────────────────────────────────

def test_observer_input_matrix_is_single_column():
    """No b0*u term: nothing but the exogenous disturbance drives this channel."""
    adrc = PreviewADRC(make_model(), w0=100.0, dt=DT)
    assert adrc.Ad_res.shape == (2, 2)
    assert adrc.Bd_res.shape == (2,)


def test_feedforward_alone_inverts_the_plant_exactly():
    """With w0 tiny the observer barely acts, so uFF must carry the tracking."""
    model = make_model()
    adrc = PreviewADRC(model, w0=1e-6, dt=DT)

    out = simulate(adrc, SimulatedPlant(model), tracking_traj(), DT, 8.0)

    # Exact model + exact inversion => the only error is the opening transient.
    assert settled_rmse(out) < 1e-6, f"settled RMSE {settled_rmse(out):.3g} nm"


def test_preview_horizon_grows_with_plant_dead_time():
    dt = 0.01
    assert PreviewADRC(make_model(theta_plant_s=0.0), 100.0, dt).d == 0
    assert PreviewADRC(make_model(theta_plant_s=0.03), 100.0, dt).d == 3

    a = PreviewADRC(make_model(theta_plant_s=0.03), 100.0, dt)
    t_d, t_d1 = a.preview_times(1.0)
    assert t_d == pytest.approx(1.03)
    assert t_d1 == pytest.approx(1.04)


def test_preview_survives_plant_dead_time():
    model = make_model(theta_plant_s=0.03)
    pid, adrc = tuned_pair(model)
    traj = tracking_traj()

    adrc_out = simulate(adrc, SimulatedPlant(model), traj, DT, 8.0)
    pid_out = simulate(pid, SimulatedPlant(model), traj, DT, 8.0)

    assert np.all(np.isfinite(adrc_out["v"]))
    # The feedforward reads the setpoint d+1 ticks ahead, so dead time costs it
    # much less than it costs the purely reactive PID.
    assert settled_rmse(adrc_out) < settled_rmse(pid_out)


def test_sensor_delay_buffer_length_tracks_theta_sensor():
    assert PreviewADRC(make_model(theta_sensor_s=0.0), 100.0, 0.01)._sensor_delay_steps == 0
    assert PreviewADRC(make_model(theta_sensor_s=0.05), 100.0, 0.01)._sensor_delay_steps == 5


def test_reset_clears_all_controller_state():
    model = make_model()
    pid, adrc = tuned_pair(model)
    traj = TrajectoryGenerator("Sine", amplitude_nm=500.0, frequency_hz=1.0)

    for ctrl in (pid, adrc):
        first = simulate(ctrl, SimulatedPlant(model), traj, DT, 2.0)
        second = simulate(ctrl, SimulatedPlant(model), traj, DT, 2.0)
        assert np.allclose(first["v"], second["v"]), f"{ctrl.name} kept stale state"


# ─────────────────────────────────────────────────────────────────────────────
# PID specifics
# ─────────────────────────────────────────────────────────────────────────────

def test_integral_anti_windup_clamps():
    model = make_model()
    pid = SoftwarePID(model, kp=0.0, ki=1.0, kd=0.0)
    for _ in range(500):
        pid.update(setpoint_nm=1e6, y_meas_nm=0.0, dt=DT)
    assert pid.diagnostics()["integral"] <= model.v_max / 1.0 + 1e-9


def test_derivative_acts_on_measurement_not_error():
    """A setpoint jump must not produce a derivative kick."""
    model = make_model()
    pid = SoftwarePID(model, kp=0.0, ki=0.0, kd=1.0, d_filter_n=20.0)
    pid.update(0.0, 0.0, DT)
    pid.update(0.0, 0.0, DT)
    assert pid.diagnostics()["d_filt"] == pytest.approx(0.0)

    pid.update(1000.0, 0.0, DT)         # setpoint jumps, measurement does not
    assert pid.diagnostics()["d_filt"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tuning and identification
# ─────────────────────────────────────────────────────────────────────────────

def test_imc_tune_matches_meter_formulas():
    model = make_model(tau_s=0.08, theta_plant_s=0.01, theta_sensor_s=0.0)
    dt, fs = 0.05, 1000.0
    g = imc_tune(model, dt, fs_sample_hz=fs)

    # tunedPID.m
    theta_eff = 0.01 + 0.0 + dt / 2 + 0.5 / fs
    denom = 0.08 + theta_eff / 2
    assert g["kp"] == pytest.approx(denom / (410.0 * (2 * theta_eff + theta_eff / 2)))
    assert g["ki"] == pytest.approx(g["kp"] / denom)

    # tunedADRC.m — theta_plant absent, the feedforward predicts it away
    theta_eff_adrc = 0.0 + dt / 2 + 0.5 / fs
    assert g["w0"] == pytest.approx(5.0 / max(0.08 + theta_eff_adrc, 1e-9))


def test_imc_gains_are_usable():
    g = imc_tune(make_model(), DT)
    assert all(g[k] > 0 for k in ("kp", "ki", "w0"))
    assert g["d_filter_n"] >= 1


def test_fit_fopdt_recovers_known_parameters():
    K, tau, theta, dv = 410.0, 0.08, 0.02, 1.0
    t = np.linspace(0, 1.0, 400)
    y = np.where(t > theta, K * dv * (1 - np.exp(-(t - theta) / tau)), 0.0)

    K_hat, tau_hat, theta_hat = fit_fopdt_step(t, y, dv)

    assert K_hat == pytest.approx(K, rel=0.02)
    assert tau_hat == pytest.approx(tau, rel=0.05)
    assert theta_hat == pytest.approx(theta, abs=0.01)


def test_fit_fopdt_survives_noise():
    K, tau, theta, dv = 410.0, 0.08, 0.01, 1.0
    t = np.linspace(0, 1.0, 400)
    clean = np.where(t > theta, K * dv * (1 - np.exp(-(t - theta) / tau)), 0.0)
    y = clean + np.random.default_rng(0).normal(0, 5.0, t.shape)

    K_hat, tau_hat, _ = fit_fopdt_step(t, y, dv)

    assert K_hat == pytest.approx(K, rel=0.05)
    assert tau_hat == pytest.approx(tau, rel=0.20)


def test_fit_fopdt_rejects_degenerate_input():
    t = np.linspace(0, 1, 100)
    with pytest.raises(ValueError):
        fit_fopdt_step(t, t, delta_v=0.0)
    with pytest.raises(ValueError):
        fit_fopdt_step(t[:3], t[:3], delta_v=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# display.py MQTT payload parsing (the prerequisite fix)
# ─────────────────────────────────────────────────────────────────────────────

def test_dn_payload_regex_matches_what_vb_publishes():
    """uMD_GUI publishes `"D:" & D & " N:" & N` (MainForm.vb)."""
    src = open("display.py", encoding="utf-8").read()
    pattern = re.search(r'_DN_RE = re\.compile\(r"([^"]+)"\)', src).group(1)
    dn_re = re.compile(pattern)

    m = dn_re.match("D:12345667 N:8901234")
    assert m is not None
    assert int(m.group(1)) == 12345667
    assert int(m.group(2)) == 8901234

    assert dn_re.match("D:-42 N:7") is not None
    assert dn_re.match("  D:1 N:2  ") is not None
    assert dn_re.match("1,2,3") is None
    assert dn_re.match("123.4") is None


# ─────────────────────────────────────────────────────────────────────────────
# app_settings — the JSON store behind "settings survive a restart"
# ─────────────────────────────────────────────────────────────────────────────

def test_settings_round_trip(tmp_path, monkeypatch):
    import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", lambda: tmp_path)

    payload = {"K_nm_per_V": 410.0, "waveform": "Sine", "log_enabled": True}
    assert app_settings.save("t.json", payload)
    assert app_settings.load("t.json") == payload


def test_settings_missing_file_is_empty(tmp_path, monkeypatch):
    import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", lambda: tmp_path)
    assert app_settings.load("nope.json") == {}


@pytest.mark.parametrize("content", ["{not json", "[1,2,3]", '"a string"', ""])
def test_settings_corrupt_file_is_empty_not_fatal(tmp_path, monkeypatch, content):
    """A bad settings file must never stop the app from starting."""
    import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", lambda: tmp_path)
    (tmp_path / "bad.json").write_text(content)
    assert app_settings.load("bad.json") == {}


def test_settings_creates_directory(tmp_path, monkeypatch):
    import app_settings
    nested = tmp_path / "a" / "b"
    monkeypatch.setattr(app_settings, "settings_dir", lambda: nested)
    assert app_settings.save("t.json", {"x": 1})
    assert (nested / "t.json").exists()


def test_settings_save_failure_is_reported_not_raised(tmp_path, monkeypatch):
    import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", lambda: tmp_path)
    # Not JSON-serialisable -- save must return False rather than propagate.
    assert app_settings.save("t.json", {"bad": object()}) is False


def test_settings_write_is_atomic(tmp_path, monkeypatch):
    """An interrupted write must not truncate the previous good file."""
    import app_settings
    monkeypatch.setattr(app_settings, "settings_dir", lambda: tmp_path)
    app_settings.save("t.json", {"keep": 1})

    real_replace = os.replace
    monkeypatch.setattr(os, "replace",
                        lambda *a: (_ for _ in ()).throw(OSError("boom")))
    assert app_settings.save("t.json", {"keep": 2}) is False
    monkeypatch.setattr(os, "replace", real_replace)

    assert app_settings.load("t.json") == {"keep": 1}


def test_feedforward_sine_inverts_first_order_plant():
    model = PlantModel(K_nm_per_V=410.0, tau_s=0.08)
    # ωτ = 1 → boost √2, lead 45 deg. Peak nm 410 → static peak 1 V, Vpp 2 V.
    freq = 1.0 / (2.0 * math.pi * model.tau_s)
    cmd = feedforward_command("Sine", 410.0, 820.0, freq, model, phase_deg=10.0)
    assert cmd.boost == pytest.approx(math.sqrt(2.0), rel=1e-6)
    assert cmd.amplitude_vpp == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-6)
    assert cmd.offset_v == pytest.approx(2.0, rel=1e-6)
    assert cmd.phase_deg == pytest.approx(55.0, abs=1e-6)


def test_feedforward_square_is_static_scale():
    model = PlantModel(K_nm_per_V=100.0, tau_s=0.08)
    cmd = feedforward_command("Square", 50.0, 100.0, 10.0, model)
    assert cmd.amplitude_vpp == pytest.approx(1.0)
    assert cmd.offset_v == pytest.approx(1.0)
    assert cmd.boost == 1.0


def test_feedforward_trim_updates_shift_and_scale_once():
    samples = [180.0, 20.0, 180.0, 20.0, 180.0, 20.0, 180.0, 20.0]
    offset, amp, changed = feedforward_trim(
        "Sine", 0.0, 100.0, samples, offset_tol_nm=25.0, amplitude_tol_frac=0.15
    )
    assert changed
    assert offset == pytest.approx(-100.0)
    assert amp == pytest.approx(100.0 * 100.0 / 80.0)


def test_feedforward_trim_holds_inside_tolerance():
    samples = [90.0, -90.0] * 8
    offset, amp, changed = feedforward_trim(
        "Sine", 0.0, 100.0, samples, offset_tol_nm=25.0, amplitude_tol_frac=0.15
    )
    assert not changed
    assert offset == 0.0
    assert amp == 100.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
