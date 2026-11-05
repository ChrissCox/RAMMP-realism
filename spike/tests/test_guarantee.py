"""Adversarial tests of the timing-only guarantee and the composition's safety rules: each one tries
to break a promise (precision identity at the same functional time, dtau/dt in [0, 1], gains that
never leak into PRECISION, bounded pauses, no motion toward the user, no gripper play while
holding or near the user, no new contact, fail-safe zones) and stays as a regression test."""

import numpy as np
import pytest

from puppy_sim import behaviors as B
from puppy_sim import checks, curves, layer, runs
from puppy_sim import director as D
from puppy_sim import timeline as tl
from puppy_sim.kin import ARM_PARTS
from puppy_sim.layer import FREE_JOINT_ACCEL, Contribution
from puppy_sim.timeline import DT, FREE, NEAR_USER, PRECISION

RUNS = ("grab_container_20261029_132745", "grab_container_20261029_131558", "grab_container_20261028_191809")
REST_GRIP = 0.79
GAIN_STEP_MAX = np.pi / (2.0 * round(layer.GAIN_RAMP_S / DT))  # steepest raised-cosine step per sample
HEAD = ARM_PARTS.index("head")


def _rest(q, seconds, grip=REST_GRIP, kin=None):
    track = tl.constant_track(q, grip, seconds)
    if kin is not None:
        runs.mark_near_user(track, kin)
    return track


def _failed(track, kin):
    return [(c.name, c.shown, c.detail) for c in checks.check_track(track, kin) if not c.ok]


def _warp_added_accel(func, warped):
    """(N, 7) acceleration the warp adds to the functional motion: |dq/dtau (tau)| * |d2tau/dt2|."""
    v = tl.interp_rows(np.gradient(func.q, DT, axis=0), warped.tau / DT)
    return np.abs(v) * np.abs(np.gradient(np.gradient(warped.tau, DT), DT))[:, None]


# ------------------------------------------------------------------------------- gain ramps
def _random_zones(rng, n=160):
    zone, i = np.empty(n, dtype=int), 0
    while i < n:
        run = int(rng.integers(1, 22))
        zone[i : i + run] = rng.choice([FREE, NEAR_USER, PRECISION], p=[0.5, 0.2, 0.3])
        i += run
    return zone


@pytest.mark.parametrize("seed", range(60))
def test_gain_never_leaks_into_precision_for_any_zone_pattern(seed):
    """Single-sample zones, zones at either end, gaps shorter than the ramp, NEAR_USER against
    PRECISION: the gain is exactly 0 on PRECISION and one sample either side, at most the
    NEAR_USER level on NEAR_USER, and never steps faster than a raised cosine."""
    zone = _random_zones(np.random.default_rng(seed))
    g = layer.zone_gain(zone)
    prec = zone == PRECISION
    guard = prec.copy()
    guard[1:] |= prec[:-1]
    guard[:-1] |= prec[1:]
    assert np.all(g[prec] == 0.0) and np.all(g[guard] == 0.0)
    assert np.all(g[zone == NEAR_USER] <= layer.GAINS[NEAR_USER] + 1e-12)
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert np.abs(np.diff(g)).max() <= GAIN_STEP_MAX + 1e-12


# ------------------------------------------------------------------------------- time-warp
@pytest.mark.parametrize("seed", range(40))
def test_random_pauses_and_slow_ins_change_only_the_timing(functional, seed):
    """Overlapping holds and slows, at the ends and past them: the warp never goes back or faster
    than 1, and every output sample is the functional path at its tau with the stricter zone."""
    rng = np.random.default_rng(100 + seed)
    end = functional.duration
    holds = [
        tl.Hold(float(rng.uniform(-0.5, end + 0.5)), float(rng.uniform(0.0, 2.0)), float(rng.uniform(0.05, 0.8)))
        for _ in range(int(rng.integers(0, 5)))
    ]
    slows = []
    for t0 in rng.uniform(-0.5, end, int(rng.integers(0, 3))):
        t1, factor, ramp_s = t0 + rng.uniform(0.0, 3.0), rng.uniform(0.2, 1.0), rng.uniform(0.05, 0.6)
        hold_s = float(rng.uniform(0.0, 1.5)) if rng.random() < 0.5 else 0.0
        slows.append(tl.Slow(float(t0), float(t1), float(factor), float(ramp_s), hold_s))
    u = tl.warp_indices(functional.n, holds, slows)
    warped = tl.warp(functional, holds, slows)
    s = tl.speed(warped)
    assert u[0] == 0.0 and u[-1] == functional.n - 1
    assert np.all(np.diff(u) >= 0.0) and s.min() >= 0.0 and s.max() <= 1.0 + 1e-12
    assert warped.q.tobytes() == tl.interp_rows(functional.q, u).tobytes()
    lo, hi = np.floor(u).astype(int), np.ceil(u).astype(int)
    assert np.array_equal(warped.zone, np.maximum(functional.zone[lo], functional.zone[hi]))


@pytest.mark.parametrize(
    "event",
    [
        lambda: tl.Slow(t0=7.0, t1=9.0, factor=1.5, ramp_s=0.3),  # a speed-up
        lambda: tl.Slow(t0=7.0, t1=9.0, factor=0.0, ramp_s=0.3),  # never arrives
        lambda: tl.Slow(t0=7.0, t1=9.0, factor=-0.5, ramp_s=0.3),  # goes back
        lambda: tl.Slow(t0=9.0, t1=7.0, factor=0.5, ramp_s=0.3),  # ends before it starts
        lambda: tl.Slow(t0=7.0, t1=9.0, factor=0.5, ramp_s=0.3, hold_s=-1.0),
        lambda: tl.Hold(at=5.0, hold_s=-1.0, ramp_s=0.3),
        lambda: tl.Hold(at=5.0, hold_s=1.0, ramp_s=0.0),  # a hold needs a ramp (bounded acceleration)
    ],
)
def test_the_time_warp_refuses_events_that_are_not_timing_only(event):
    with pytest.raises(ValueError):
        event()


@pytest.mark.parametrize("name", RUNS)
def test_a_hold_at_the_fastest_moment_adds_at_most_the_accel_cap(sc, name):
    """Stopping the arm mid-motion with the ramp hold_ramp_s asks for: the added acceleration
    stays under the cap (the ramp is never rounded shorter than asked)."""
    k = sc.kin()
    func = runs.functional_track(runs.load_run(name), k)
    at = int(np.argmax(tl.joint_speed(func))) * DT
    for cap in (FREE_JOINT_ACCEL, 1.0):
        holds = [tl.Hold(at, 1.0, tl.hold_ramp_s(func, at, cap, 0.05))]
        warped = tl.warp(func, holds)
        assert _warp_added_accel(func, warped).max() <= cap * (1.0 + 1e-6)
        assert warped.q.tobytes() == tl.interp_rows(func.q, tl.warp_indices(func.n, holds)).tobytes()


@pytest.mark.parametrize("lock_on", D.LOCK_ON_MODES)
@pytest.mark.parametrize("name", RUNS)
def test_the_directors_pauses_add_at_most_the_accel_cap_of_each_tier(sc, name, lock_on):
    k = sc.kin()
    func = runs.functional_track(runs.load_run(name), k)
    rules = [r for r in D.RULES["grab_container"] if r.lock_on in (None, lock_on)]
    planned = D.plan(func, D.find_events(func, k), rules, sc, k)
    warped = tl.warp(func, planned.holds, planned.slows)
    assert np.all(_warp_added_accel(func, warped) <= layer.sample_caps(warped.zone).joint_accel * (1.0 + 1e-6))


# --------------------------------------------------------------- precision identity, end to end
def test_precision_is_the_functional_path_at_tau_through_warp_and_compose(sc, functional):
    """A pause mid-motion, a pause inside 'detect', a stop-and-creep at the final approach, a tilt
    straddling the detect boundary and breathing everywhere: on every PRECISION sample the
    composite is bit for bit the functional path at its tau (computed here without the warped
    track compose was given), every sample touching a precise functional sample is PRECISION, and
    dtau/dt stays in [0, 1] there."""
    k = sc.kin()
    holds = [
        tl.Hold(at=3.5, hold_s=0.6, ramp_s=tl.hold_ramp_s(functional, 3.5, FREE_JOINT_ACCEL, 0.3)),
        tl.Hold(at=5.4, hold_s=0.8, ramp_s=0.3),
    ]
    slows = [tl.Slow(t0=8.84, t1=10.5, factor=0.6, ramp_s=0.4, hold_s=0.5)]
    warped = tl.warp(functional, holds, slows)
    ctx = B.Context.of(warped, sc, k)
    parts = [B.head_tilt(ctx, 4.5, 8.0, angle_rad=np.radians(10.0)), B.breathe(ctx, 0.0, 12.0)]
    puppy = layer.compose(warped, parts, k)
    u = tl.warp_indices(functional.n, holds, slows)
    prec = puppy.zone == PRECISION
    lo, hi = np.floor(u).astype(int), np.ceil(u).astype(int)
    assert np.all(prec[(functional.zone[lo] == PRECISION) | (functional.zone[hi] == PRECISION)])
    assert prec.sum() > 100 and (~prec).sum() > 100
    assert puppy.q[prec].tobytes() == tl.interp_rows(functional.q, u)[prec].tobytes()
    s = tl.speed(puppy)[prec]
    assert s.min() >= 0.0 and s.max() <= 1.0 + checks.DTAU_TOL  # (tau differences carry float noise)
    assert np.abs(puppy.q[~prec] - puppy.func.q[~prec]).max() > 1e-3  # the expression is there


def test_a_static_offset_across_a_precision_boundary_is_shrunk_by_compose(sc, home):
    """A look held while the zone gain ramps it out before PRECISION and back in after: the ramp
    alone moves the joints. compose must blame the look for it and shrink it to fit the caps."""
    k = sc.kin()
    rest = _rest(home, 10.0, kin=k)
    rest.zone[int(5.0 / DT) : int(6.0 / DT)] = PRECISION
    ctx = B.Context.of(rest, sc, k)
    look = B.gaze_to(ctx, ctx.container, 0.2, 7.2)  # arrives by ~3 s, turns back at 7.2 s
    out = layer.compose(rest, [look], k)
    assert look.scale < 1.0 and any(line.startswith("gaze: amplitude x") for line in out.log)
    assert not any("still over a cap" in line for line in out.log)
    assert not _failed(out, k)


# ------------------------------------------------------------------------------ the gripper
def _pinch_pose(sc, k, home):
    """The gripper pointing down over the counter, its open fingertips a few mm above it: closing
    moves the fingertips 13 mm further out (the 2F-85's parallel linkage), into the counter."""
    R = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    q, pos_err, rot_err = k.ik(home, np.array([0.62, -0.15, sc.placement.table_top_z + 0.045]), R, "tool")
    assert pos_err < 1e-5 and rot_err < 1e-4
    return q


def test_a_gripper_expression_cannot_hide_a_new_contact(sc, home):
    """A clack (and a full chomp) where the fingertips would hit the counter: compose evaluates the
    composite gripper too and shrinks the chomp, and the checks see the composite gripper."""
    k = sc.kin()
    rest = _rest(_pinch_pose(sc, k, home), 3.0, grip=0.0, kin=k)
    assert np.all(rest.zone == FREE) and not rest.held.any()
    func_clear = k.clearance_track(rest)
    assert 0.002 < func_clear[:, 1, HEAD].min() < 0.01  # clear, but only just
    floor = layer.clearance_floor(func_clear)
    ctx = B.Context.of(rest, sc, k)
    chomp = Contribution.empty("chomp", rest.n)
    chomp.grip[:] = curves.bumps(rest.t, 0.5, 1, 2.0)  # close fully and open again
    for c in (B.clack(ctx, 0.5), chomp):
        out = layer.compose(rest, [c], k)
        true = np.array([k.clearances(out.q[i], out.grip[i], out.held[i]) for i in range(out.n)])
        assert np.all(true >= floor - 1e-9), c.name
        assert c.scale < 1.0 and not _failed(out, k)
    doctored = layer.compose(rest, [], k)
    doctored.grip[:] = np.clip(rest.grip + chomp.grip, 0.0, 1.0)  # the fingertips into the counter
    assert "obstacle clr" in {name for name, _, _ in _failed(doctored, k)}


def test_a_gripper_jump_is_shrunk_to_the_grippers_speed(sc, home):
    """A gripper offset that steps half a stroke in one sample (14x the 2F-85's speed): compose
    shrinks it to the gripper's speed, and the checks catch the step if it is forced back in."""
    k = sc.kin()
    rest = _rest(home, 3.0, kin=k)
    snap = Contribution.empty("snap", rest.n)
    snap.grip[60:] = -0.5
    out = layer.compose(rest, [snap], k)
    assert snap.scale < 0.2 and layer.gripper_rate(out.grip - rest.grip).max() <= layer.EXPR_GRIP_SPEED
    assert not _failed(out, k)
    out.grip[60:] = rest.grip[60:] - 0.5
    assert "grip speed" in {name for name, _, _ in _failed(out, k)}


@pytest.mark.parametrize("make", [B.wiggle, B.clack])
@pytest.mark.parametrize("where", ["near the user", "holding", "precision"])
def test_wiggle_and_clack_stop_when_a_forbidden_span_starts_mid_beat(sc, home, make, where):
    """The user comes near, the gripper takes hold, or PRECISION starts in the middle of the beat:
    nothing on those samples (the gripper bit for bit functional), and still smooth."""
    k = sc.kin()
    rest = _rest(home, 5.0, kin=k)
    i = int(round(1.6 / DT))  # the beat runs from 1.0 s for about 1.5 s
    if where == "holding":
        rest.held[i:] = True
    else:
        rest.zone[i:] = NEAR_USER if where == "near the user" else PRECISION
    c = make(B.Context.of(rest, sc, k), 1.0)
    assert np.any(c.head[:i]) or np.any(c.joints[:i]) or np.any(c.grip[:i])
    assert not np.any(c.head[i:]) and not np.any(c.joints[i:]) and not np.any(c.grip[i:])
    out = layer.compose(rest, [c], k)
    assert out.grip[i:].tobytes() == rest.grip[i:].tobytes()
    assert not _failed(out, k)


# --------------------------------------------------------------------------- near the user
def test_a_near_user_offset_toward_the_person_is_removed(sc, home):
    """At a posture with the head truly within 0.35 m of the user: a head shift and a joint offset
    straight at them come out with no arm part closer to the user, while the sideways part of the
    shift survives (at the NEAR_USER gain)."""
    k = sc.kin()
    p, R = k.fk(home, "tool")
    flat = sc.person_head - p
    flat[2] = 0.0
    q, pos_err, _ = k.ik(home, p + 0.10 * flat / np.linalg.norm(flat), R, "tool")
    assert pos_err < 1e-5
    rest = _rest(q, 3.0, grip=0.5, kin=k)
    assert np.all(rest.zone == NEAR_USER)
    eye = k.fk(q, "eye")[0]
    toward = (sc.person_head - eye) / np.linalg.norm(sc.person_head - eye)
    grads = k.clearance_gradients(q, 0.5, False)[0]
    bump = curves.bumps(rest.t, 0.5, 1, 2.0)[:, None]
    c = Contribution.empty("at the user", rest.n)
    c.head[:, :3] = (0.04 * toward + np.array([0.0, 0.0, 0.02])) * bump
    c.joints[:] = -0.04 * grads[HEAD] / np.linalg.norm(grads[HEAD]) * bump
    out = layer.compose(rest, [c], k)
    func_clear = k.clearance_track(rest)[:, 0]
    comp_clear = k.clearance_track(out)[:, 0]
    assert np.all(out.gain <= layer.GAINS[NEAR_USER] + 1e-12)
    assert (comp_clear - func_clear).min() >= -2e-4  # first order: only a curvature residue
    assert np.abs(out.q - rest.q).max() > 1e-3  # the sideways part is still there
    assert not _failed(out, k)


# -------------------------------------------------------------------------------- zones
def test_unknown_phases_messages_and_capabilities_are_precision():
    assert runs.zone_of("grab_container", "view", "") == FREE
    assert runs.zone_of("grab_container", "view", "an unexpected message") == PRECISION
    assert runs.zone_of("grab_container", "home", "re-homing after a fault") == PRECISION
    assert runs.zone_of("grab_container", "approach", "a new message") == PRECISION
    assert runs.zone_of("grab_container", "canceled", "") == PRECISION
    assert runs.zone_of("button_press", "home", "") == PRECISION


def test_todays_grab_container_pairs_are_known():
    """grasp.py reports the approach under its own phase, and a dry run's planning: the approach is
    the same FREE approach (its final part PRECISION by geometry); a dry run moves nothing, so its
    planning is PRECISION and Puppy Mode does not move either."""
    assert runs.zone_of("grab_container", "approach", runs.APPROACH_MESSAGE) == FREE
    assert runs.zone_of("grab_container", "approach", runs.DRY_RUN_MESSAGE) == PRECISION
    assert runs.zone_of("grab_container", "raise", runs.DRY_RUN_MESSAGE) == PRECISION


def test_todays_approach_phase_gives_the_recorded_runs_zones(sc, run):
    """The recorded run with its approach reported as today's grasp.py does it: the same zones and
    phase names, so the director sees the same events."""
    import dataclasses

    k = sc.kin()
    renamed = tuple(
        (t, "approach" if (phase, msg) == ("grasp", runs.APPROACH_MESSAGE) else phase, msg)
        for t, phase, msg in run.feedback
    )
    today = runs.functional_track(dataclasses.replace(run, feedback=renamed), k)
    recorded = runs.functional_track(run, k)
    assert np.array_equal(today.zone, recorded.zone)
    assert [p.name for p in today.phases] == [p.name for p in recorded.phases]


def test_an_unknown_capability_is_replayed_exactly(sc, run):
    """Every phase unknown: every sample PRECISION, no beat, and Puppy Mode is the plain replay."""
    import dataclasses

    k = sc.kin()
    func = runs.functional_track(dataclasses.replace(run, capability="mystery"), k)
    assert np.all(func.zone == PRECISION)
    puppy, _ = D.direct_track(func, "mystery", sc, k)
    assert puppy.q.tobytes() == func.q.tobytes() and puppy.grip.tobytes() == func.grip.tobytes()
    assert not puppy.beats and not _failed(puppy, k)


@pytest.mark.parametrize("beat", ["curious", "proud"])
def test_moving_beats_put_on_precision_events_never_reach_precision(sc, functional, monkeypatch, beat):
    """A doctored table that plays a moving beat over every PRECISION run (instead of stillness):
    the director's fence and the layer's zero gain keep every PRECISION sample the functional one."""
    k = sc.kin()
    keep = tuple(r for r in D.RULES["grab_container"] if r.event != "precision")
    monkeypatch.setitem(D.RULES, "doctored", keep + (D.Rule("precision", beat),))
    puppy, parts = D.direct_track(functional, "doctored", sc, k, budget=None)
    prec = puppy.zone == PRECISION
    for c in parts:
        assert not np.any(c.head[prec]) and not np.any(c.joints[prec]), c.name
    assert puppy.q[prec].tobytes() == puppy.func.q[prec].tobytes()
    assert not _failed(puppy, k)


# --------------------------------------------------------------------- the checks themselves
def test_checks_hold_the_composite_to_the_source_path_at_tau(sc, functional):
    """The precision check must not trust the warped track compose was handed: a functional track
    edited after the warp (composite and its .func moved together) and a pause stopped too hard
    mid-motion are both caught."""
    k = sc.kin()
    puppy = layer.compose(tl.warp(functional, [tl.Hold(5.4, 0.5, 0.3)]), [], k)
    assert not _failed(puppy, k)
    prec = puppy.zone == PRECISION
    puppy.q[prec] += 0.01
    puppy.func.q[prec] += 0.01
    assert "path" in {name for name, _, _ in _failed(puppy, k)}
    at = int(np.argmax(tl.joint_speed(functional))) * DT
    abrupt = layer.compose(tl.warp(functional, [tl.Hold(at, 0.5, 0.04)]), [], k)
    assert "warp accel" in {name for name, _, _ in _failed(abrupt, k)}
