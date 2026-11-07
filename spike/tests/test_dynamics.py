"""The approximate dynamic model of the driver's execution modes: the impedance law holds the arm
under gravity compensation, its step response is stable with the overshoot its damping ratio
predicts, a slow move is tracked in stiff and position mode, nothing goes NaN, the reference and
the payload behave, and Puppy Mode adds no deviation inside PRECISION beyond the plain task's."""

import numpy as np
import pytest

from puppy_sim import checks
from puppy_sim import dynamics as dy
from puppy_sim.clips import registry
from puppy_sim.timeline import DT, constant_track, joint_move, scripted_track

HIDDEN = dy.Plant(armature=0.0)  # the plant the driver's model describes exactly (armature hidden)
STEP_RAD = 0.03
STEP_HOLD_S = 3.0


def overshoot(sc, plant, profile, joint, q0):
    """(measured overshoot, the damping ratio's prediction) of a STEP_RAD reference step on one
    joint at q0, positions only (no feedforward), as fractions of the step."""
    q1 = q0.copy()
    q1[joint] += STEP_RAD
    lead = int(round(0.5 / DT))
    track = scripted_track(np.vstack([np.tile(q0, (lead, 1)), np.tile(q1, (int(STEP_HOLD_S / DT), 1))]), 0.79)
    q = dy.simulate(track, dy.control(f"impedance:{profile}", feedforward=False), sc, plant)
    x = (q[:, joint] - q0[joint]) / STEP_RAD
    _, zeta = dy.modes(q0, dy.PROFILES[profile], plant, sc)
    z = zeta[joint]
    return x.max() - 1.0, np.exp(-np.pi * z / np.sqrt(1.0 - z * z)), x


@pytest.mark.parametrize("profile", ["soft", "default", "stiff"])
@pytest.mark.parametrize("plant", [dy.DEFAULT_PLANT, HIDDEN], ids=["menagerie", "hidden"])
def test_holds_at_rest_under_gravity_compensation(sc, home, run, profile, plant):
    """Gravity is compensated exactly for the arm and gripper, so the arm stays put at home and at
    a posture a third of the way through the reference run (no drift, no NaN)."""
    for q0 in (home, run.q[len(run.q) // 3]):
        q = dy.simulate(constant_track(q0, 0.79, 3.0), f"impedance:{profile}", sc, plant)
        assert np.all(np.isfinite(q))
        assert np.abs(q - q0).max() < 1e-4


def test_a_held_container_sags_and_settles(sc, home):
    """The driver's model has no container: its weight bends the springs, more on soft gains, and
    the arm settles there (the last second within 0.2 mrad: on 'soft' the wrist roll, lightly damped
    under the menagerie's armature, still rings a little 5 s after the weight came on)."""
    track = constant_track(home, 0.79, 6.0)
    track.held[:] = True
    sag = {}
    for profile in ("soft", "default", "stiff"):
        q = dy.simulate(track, f"impedance:{profile}", sc)
        late = q[-int(1.0 / dy.CONTROL_DT) :]
        assert np.abs(late - late[-1]).max() < 2e-4
        sag[profile] = np.abs(q[-1] - home).max()
    assert sag["soft"] > sag["default"] > sag["stiff"] > 0.005


@pytest.mark.parametrize("profile", ["default", "stiff"])
@pytest.mark.parametrize("joint", [1, 3])
def test_step_overshoot_matches_the_damping_ratio(sc, home, profile, joint):
    """A reference step on a big joint (2, 4): stable, settling on the step, and its overshoot
    within 4 points of exp(-pi zeta / sqrt(1 - zeta^2)), zeta the effective ratio (the driver's
    zeta on M_ii, lowered by armature it does not model)."""
    for plant in (HIDDEN, dy.DEFAULT_PLANT):
        measured, predicted, x = overshoot(sc, plant, profile, joint, home)
        assert abs(measured - predicted) < 0.04, (plant, measured, predicted)
        assert abs(x[-1] - 1.0) < 1e-3


def test_unmodelled_armature_lowers_the_damping(sc, home):
    """The wrist roll's own inertia is tiny, so the menagerie's armature dominates it: the driver's
    damping, set for M_ii, leaves it lightly damped (the overshoot says so, as predicted)."""
    _, zeta = dy.modes(home, dy.PROFILES["default"], dy.DEFAULT_PLANT, sc)
    assert zeta[6] < 0.1 and np.all(zeta <= 0.5 + 1e-12)
    _, zeta_hidden = dy.modes(home, dy.PROFILES["default"], HIDDEN, sc)
    assert np.allclose(zeta_hidden, 0.5)
    measured, predicted, _ = overshoot(sc, dy.DEFAULT_PLANT, "default", 6, home)
    assert measured > 0.7 and abs(measured - predicted) < 0.02


def test_slow_move_is_tracked_in_stiff_and_position_mode(sc, home, kin):
    """A slow move (0.3 rad on every joint over 4 s): position mode reproduces the command, and
    stiff impedance with the trajectory's feedforward stays within 1 mm at the head points."""
    track = joint_move(home, home + 0.3, 4.0, grip=0.79)
    pos = dy.achieved(track, "position", sc)
    assert np.abs(pos.q - track.q).max() < 1e-12
    stiff = dy.achieved(track, "impedance:stiff", sc)
    assert np.abs(stiff.q - track.q).max() < 1e-3
    assert dy.point_error_mm(kin, track.q, stiff.q).max() < 1.0
    assert stiff.commanded is track and stiff.control == "impedance:stiff"


def test_a_wag_stays_finite_in_every_mode(sc, kin):
    """The happy wag under every profile, positions only too: finite, within 5 cm of its command."""
    from puppy_sim.clips.impedance_wag import source

    track = source()
    for spec in ("impedance:soft", "impedance:default", "impedance:stiff"):
        for ff in (True, dy.VELOCITIES, False):
            ach = dy.achieved(track, dy.control(spec, feedforward=ff), sc)
            assert np.all(np.isfinite(ach.q)) and ach.n == track.n
            assert ach.func is not None and ach.func.commanded is track.func
            assert dy.point_error_mm(kin, track.q, ach.q).max() < 50.0


def test_the_reference_passes_through_the_samples(functional):
    """The quintic Hermite reference hits every 50 Hz sample and its finite-difference velocity
    there; positions only is the straight line between them with no profile."""
    track = scripted_track(functional.q[:200], 0.5)
    q, qd, qdd = dy.reference(track)
    knots = q[:: dy.STEPS_PER_SAMPLE]
    assert np.abs(knots - track.q).max() < 1e-12
    assert np.abs(qd[:: dy.STEPS_PER_SAMPLE] - np.gradient(track.q, DT, axis=0)).max() < 1e-9
    q_lin, qd_lin, qdd_lin = dy.reference(track, feedforward=False)
    assert np.abs(q_lin[:: dy.STEPS_PER_SAMPLE] - track.q).max() < 1e-12
    assert not qd_lin.any() and not qdd_lin.any()


def test_profiles_and_custom_gains_follow_the_driver():
    """The profiles are interface/gains.cpp's, 'default' is the session default (medium), and a
    custom spec is refused outside validate_custom's bounds."""
    assert dy.PROFILES["default"].kq == dy.PROFILES["medium"].kq == (80.0,) * 4 + (30.0,) * 3
    assert dy.PROFILES["soft"].zeta == 0.4 and dy.PROFILES["stiff"].leash == 0.25
    ctl = dy.control({"kq": [100] * 7, "zeta": 0.6, "torque_limit": [39] * 4 + [9] * 3})
    assert ctl.mode == dy.IMPEDANCE and ctl.gains.kq == (100.0,) * 7
    for bad in (
        {"kq": [500] * 7, "zeta": 0.5},
        {"kq": [80] * 7, "zeta": 3.0},
        {"kq": [80] * 7, "zeta": 0.5, "torque_limit": [1] * 7},
    ):
        with pytest.raises(ValueError):
            dy.control(bad)
    with pytest.raises(ValueError):
        dy.control("impedance:wobbly")
    assert dy.control("position").label == "position" and dy.slug("impedance:soft") == "impedance_soft"


def test_lock_on_adds_no_deviation_inside_precision(sc, kin):
    """lock_on_view under position and impedance 'default': inside PRECISION the arm with Puppy
    Mode puts the tool where the plain task's arm does on the same commands, never more than
    SETTLED_MM (half a millimetre) apart, though both stray far more while holding the container."""
    from puppy_sim.clips.impedance_lock_on import source

    track = source()
    exact = dy.response(track, "position", sc, kin=kin)
    assert exact.peak_mm == 0.0 and exact.precision.added_mm == 0.0 and exact.precision.samples > 100
    springy = dy.response(track, "impedance:default", sc, kin=kin)
    assert springy.precision.added_mm < dy.SETTLED_MM and springy.precision.settle_s == 0.0
    assert springy.precision.plain_max_mm > 10.0  # the held container's sag: the task's own
    lean = next(b for b in springy.beats if b.beat == "LOCK ON")
    assert 0.9 < lean.ratio < 1.1


@pytest.mark.parametrize("name", ["impedance_wag", "impedance_lock_on", "impedance_profiles", "impedance_friction"])
def test_compliance_clips_pass_every_check(name):
    """The impedance_* clips: check holds each side's command (every check passes) and notes how
    far the simulated arm strays from it."""
    rows = checks.check_named(name)
    assert [label for _, label, *_ in rows] == list(registry()[name].sides)
    for clip, label, _, results, _ in rows:
        failed = [(c.name, c.shown, c.detail) for c in results if not c.ok]
        assert not failed, (clip, label, failed)
        note = next(c for c in results if c.name == "achieved")
        assert note.detail.startswith(f"under {label} ")


def test_positions_and_velocities_are_a_third_trajectory_format(sc, home):
    """Positions + central-difference velocities (execute_joint_trajectory): its own label and slug,
    and its reference is neither the quintic one (accelerations) nor the straight line."""
    ctl = dy.control("impedance:default", feedforward=dy.VELOCITIES)
    assert ctl.label == "impedance:default (positions+velocities)" and dy.slug(ctl).endswith("_posvel")
    assert dy.control("impedance:default").label != ctl.label != dy.control("impedance:default", feedforward=False).label
    track = joint_move(home, home + 0.2, 2.0, grip=0.79)
    _, qd_vel, qdd_vel = dy.reference(track, dy.VELOCITIES)
    _, qd_full, qdd_full = dy.reference(track, True)
    assert not qdd_vel.any() and qdd_full.any() and np.abs(qd_vel).max() > 0.05
    assert np.abs(qd_vel - qd_full).max() < 0.02  # the same profile within a few rad/s^2 * DT


def test_overshoot_report_covers_wag_lock_on_and_bounce_in_all_three_formats():
    """The report's rows: finite, the tool stays within 5 cm of its command in every format, and
    feeding the profile forward (pos+vel+acc) does not track worse than positions only."""
    rows = dy.overshoot_report(profiles=("default",))
    assert {(r[0], r[1]) for r in rows} == {(b, f) for b in ("wag", "lock_on", "bounce") for f, _ in dy.FEEDFORWARDS}
    by = {(r[0], r[1]): r[3] for r in rows}
    for o in by.values():
        assert all(np.isfinite(v) for v in o.values()) and o["err_mm"] < 50.0
    for beat in ("wag", "bounce"):  # lock_on's reach and grip are the task's own, with the container
        assert by[beat, "pos+vel+acc"]["err_mm"] <= by[beat, "pos only"]["err_mm"] + 1e-6
    assert "over-travel" in dy.format_overshoot(rows)
