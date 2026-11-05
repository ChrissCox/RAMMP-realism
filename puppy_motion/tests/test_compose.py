"""Composition: PRECISION is untouched, gains ramp outside zones, caps shrink behaviors (tiered by
zone, gentler near the user), the 'snappy' preset only when asked, and near the user nothing moves
toward them. Ported from the spike's test_layer.py, test_guarantee.py and test_kin.py."""

import copy

import numpy as np
import pytest

from puppy_motion import caps as C
from puppy_motion import compose
from puppy_motion.caps import CAP_PRESETS, URDF_GUARDS
from puppy_motion.compose import Contribution, compose as compose_track, head_dq
from puppy_motion.geometry import PARTS
from puppy_motion.track import DT, Zone, constant_track, joint_move, resample_track

FREE, NEAR_USER, PRECISION = Zone.FREE, Zone.NEAR_USER, Zone.PRECISION
GAIN_STEP_MAX = np.pi / (2.0 * round(C.GAIN_RAMP_S / DT))  # steepest raised-cosine step per sample
MEASURED_PEAK = np.array([0.36, 1.12, 0.60, 0.65, 0.98, 0.87, 0.98])  # spike/puppy_sim/envelope.py, FREE moving samples
HEAD = PARTS.index("head")


def _big_offset(n):
    c = Contribution.empty("big", n)
    c.head[:] = [0.02, -0.02, 0.03, 0.4, 0.3, -0.5]  # 2-3 cm and ~20-30 deg, everywhere
    c.joints[:] = 0.2
    return c


def _failed(out, chain, scene, caps=None):
    """Every cap the composed track breaks, against its functional track."""
    return compose.cap_violations(chain, scene, out.func, out.q, out.grip, caps or out.caps)


def _clearance_track(chain, scene, track):
    return np.array([compose.clearances_of(chain, scene, q, h) for q, h in zip(track.q, track.held)])


# ------------------------------------------------------------------------------ test_layer.py
def test_precision_samples_are_bit_identical_under_a_large_offset(functional, chain, scene):
    track = resample_track(functional, np.arange(260, 560, dtype=float))  # detect .. close
    out = compose_track(track, [_big_offset(track.n)], chain, scene)
    prec = out.zone == PRECISION
    assert prec.any() and (~prec).any()
    assert np.all(out.gain[prec] == 0.0)
    assert out.q[prec].tobytes() == track.q[prec].tobytes()
    assert out.func.q.tobytes() == track.q.tobytes()  # the functional motion itself is never changed
    assert np.abs(out.q[~prec] - track.q[~prec]).max() > 1e-3  # elsewhere the offset does show


def test_gain_ramps_finish_before_a_lower_zone():
    zone = np.full(200, FREE)
    zone[80:120] = PRECISION
    zone[150:170] = NEAR_USER
    g = C.zone_gain(zone)
    assert np.all(g[79:121] == 0.0)  # already 0 one sample before the zone
    assert np.all(g[150:170] <= C.GAINS[NEAR_USER] + 1e-12)
    ramp = int(round(C.GAIN_RAMP_S / DT))
    assert g[79 - ramp - 1] == 1.0 and 0.0 < g[79 - ramp // 2] < 1.0
    assert np.all(np.diff(g[50:80]) <= 1e-12)  # a smooth monotone ramp down
    assert np.abs(np.diff(g)).max() < 0.15


def test_a_too_fast_behavior_scales_itself_down(home, chain, scene):
    rest = constant_track(home, 0.5, 3.0)
    wiggle = Contribution.empty("wiggle", rest.n)
    wiggle.head[:, 5] = np.radians(25.0) * np.sin(2 * np.pi * 2.0 * rest.t) * np.sin(np.pi * rest.t / 3.0)
    out = compose_track(rest, [wiggle], chain, scene)
    assert wiggle.scale < 0.5
    assert any(line.startswith("wiggle: amplitude x") for line in out.log)
    assert _failed(out, chain, scene) == []  # free speed, accel, head, person and obstacle clearance


def test_near_the_user_nothing_moves_toward_them(home, chain, scene):
    rest = constant_track(home, 0.5, 2.0, zone=NEAR_USER)
    toward = Contribution.empty("toward", rest.n)
    direction = scene.person_head() - chain.fk(home, "ee")[0]
    toward.head[:, :3] = 0.05 * direction / np.linalg.norm(direction) * np.sin(np.pi * rest.t / 2.0)[:, None]
    out = compose_track(rest, [toward], chain, scene)
    func = _clearance_track(chain, scene, rest)[:, 0]
    comp = _clearance_track(chain, scene, out)[:, 0]
    assert np.all(out.gain <= C.GAINS[NEAR_USER] + 1e-12)
    assert np.all(comp >= func - 1e-3)  # no arm part closer to the person (first order)
    assert comp[:, HEAD].max() >= func[:, HEAD].max()


def test_gripper_offsets_stop_while_holding(functional, chain, scene):
    track = resample_track(functional, np.arange(560, 700, dtype=float))  # close .. done
    track.held[track.n // 2 :] = True  # the recorded runs carry no held flag: the container rides in the gripper
    snap = Contribution.empty("snap", track.n)
    snap.grip[:] = -0.3
    out = compose_track(track, [snap], chain, scene)
    assert out.held.any()
    assert out.grip[out.held].tobytes() == track.grip[out.held].tobytes()


def test_caps_follow_the_most_restrictive_zone_across_a_ramp():
    """Wherever the gain ramps toward NEAR_USER or PRECISION, the gentle caps already apply."""
    zone = np.full(300, FREE)
    zone[100:140] = NEAR_USER
    zone[220:240] = PRECISION
    limits = C.sample_caps(zone)
    gain = C.zone_gain(zone)
    ramping = (gain < 1.0) & (zone == FREE)
    assert ramping.any() and np.all(limits.tier[ramping] != FREE)
    assert np.all(limits.head_speed[ramping] == C.NEAR_HEAD_SPEED)
    assert np.all(limits.tier[zone == NEAR_USER] == NEAR_USER) and np.all(limits.tier[:60] == FREE)
    free, gentle = CAP_PRESETS["default"][FREE], CAP_PRESETS["default"][NEAR_USER]
    assert np.array_equal(limits.joint_speed[0], free.joint_speed)
    assert np.array_equal(limits.joint_accel[120], gentle.joint_accel)
    assert gentle.head_speed < free.head_speed and max(gentle.joint_accel) < max(free.joint_accel)


def _wave(n, seconds, amplitude):
    t = np.arange(n) * DT
    return amplitude * np.sin(np.pi * t / seconds) ** 2


def test_a_held_offset_carried_into_near_user_moves_under_the_gentle_caps(home, chain, scene):
    """A roll held while the user comes near: the gain ramp alone moves it, already under the
    gentle caps (0.5 rad/s, 1.5 rad/s^2 per joint), which compose shrinks it to."""
    rest = constant_track(home, 0.5, 3.0)
    rest.zone[int(round(1.5 / DT)) :] = NEAR_USER
    roll = Contribution.empty("roll", rest.n)
    min_jerk = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
    roll.joints[:, 6] = np.radians(20.0) * min_jerk(rest.t / 1.0)  # eased in, then held
    out = compose_track(rest, [roll], chain, scene)
    limits = C.sample_caps(out.zone)
    gentle = limits.tier != FREE
    v = np.abs(np.gradient(out.q - rest.q, DT, axis=0))
    a = np.abs(np.gradient(np.gradient(out.q - rest.q, DT, axis=0), DT, axis=0))
    assert v[gentle].max() > 0.05  # the ramp moves it
    assert np.all(v[gentle] <= limits.joint_speed[gentle] + 1e-9)
    assert np.all(a[gentle] <= limits.joint_accel[gentle] + 1e-9)
    assert _failed(out, chain, scene) == []


def test_snappy_is_only_used_when_asked_and_only_loosens_the_wrist_in_free(home, chain, scene):
    """'snappy' differs from 'default' only in FREE's wrist acceleration; compose uses the default
    unless told, and a snappy track is checked against snappy's caps."""
    default, snappy = CAP_PRESETS["default"], CAP_PRESETS["snappy"]
    assert C.DEFAULT_CAPS == "default"
    for zone in (NEAR_USER, PRECISION):
        assert np.array_equal(snappy[zone].joint_speed, default[zone].joint_speed)
        assert np.array_equal(snappy[zone].joint_accel, default[zone].joint_accel)
        assert snappy[zone].head_speed == default[zone].head_speed
    assert np.array_equal(snappy[FREE].joint_speed, default[FREE].joint_speed)
    assert np.array_equal(snappy[FREE].joint_accel[:4], default[FREE].joint_accel[:4])
    assert min(snappy[FREE].joint_accel[4:]) == C.SNAPPY_WRIST_ACCEL > max(default[FREE].joint_accel)
    rest = constant_track(home, 0.5, 2.0)
    wag = Contribution.empty("wag", rest.n)
    wag.joints[:, 6] = _wave(rest.n, 0.8, np.radians(4.0)) * np.sin(2 * np.pi * 1.5 * rest.t)
    quick = compose_track(rest, [copy.deepcopy(wag)], chain, scene, caps="snappy")
    assert quick.caps == "snappy" and _failed(quick, chain, scene) == []
    assert compose_track(rest, [wag], chain, scene).caps == "default" and wag.scale < 1.0
    # the same motion held to the default caps fails
    assert "expressive joint accel" in {v.kind for v in _failed(quick, chain, scene, "default")}


# --------------------------------------------------------------------------- test_guarantee.py
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
    g = C.zone_gain(zone)
    prec = zone == PRECISION
    guard = prec.copy()
    guard[1:] |= prec[:-1]
    guard[:-1] |= prec[1:]
    assert np.all(g[prec] == 0.0) and np.all(g[guard] == 0.0)
    assert np.all(g[zone == NEAR_USER] <= C.GAINS[NEAR_USER] + 1e-12)
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert np.abs(np.diff(g)).max() <= GAIN_STEP_MAX + 1e-12


# -------------------------------------------------------------------------------- test_kin.py
def test_head_offset_is_realised_by_the_iterated_dls(chain, home):
    offset = np.array([0.01, -0.005, 0.008, np.radians(15), np.radians(-10), np.radians(20)])
    p_t, R_t = compose.head_target(chain, *chain.fk(home, "ee"), offset)
    p, R = chain.fk(home + head_dq(chain, home, offset), "ee")
    assert np.linalg.norm(p - p_t) < 1e-3
    assert np.degrees(np.linalg.norm(compose.rot_err(R_t, R))) < 0.5


def test_zero_offset_gives_exactly_zero(chain, home):
    assert np.array_equal(head_dq(chain, home, np.zeros(6)), np.zeros(7))


# ------------------------------------------------------------------------------------- new
def test_free_caps_stay_inside_the_measured_envelope():
    assert (CAP_PRESETS["default"][FREE].joint_speed <= MEASURED_PEAK).all()
    assert CAP_PRESETS["default"][FREE].joint_speed.tolist() == [0.36, 0.85, 0.59, 0.64, 0.85, 0.85, 0.85]
    assert CAP_PRESETS["default"][FREE].joint_accel.tolist() == [3.0] * 7
    assert CAP_PRESETS["default"][FREE].head_speed == 0.40
    assert CAP_PRESETS["default"][NEAR_USER].head_speed == 0.16
    assert (CAP_PRESETS["default"][NEAR_USER].joint_speed <= 0.5).all()
    assert CAP_PRESETS["default"][NEAR_USER].joint_accel.tolist() == [1.5] * 7


def test_snappy_only_raises_wrist_accel_in_free():
    assert CAP_PRESETS["snappy"][FREE].joint_accel.tolist() == [3.0] * 4 + [9.0] * 3


def test_guards_are_the_rammp_limits():
    assert URDF_GUARDS == {1: 2.24, 3: pytest.approx(np.radians(144.0)), 5: pytest.approx(np.radians(115.0))}


def test_the_result_is_a_track_carrying_gain_log_and_labels(home, chain, scene):
    from puppy_motion.track import Label

    rest = constant_track(home, 0.5, 1.0)
    nod = Contribution.empty("nod", rest.n, labels=[Label("nod", 0.2, 0.8)])
    nod.joints[:, 6] = _wave(rest.n, 1.0, 0.02)
    out = compose_track(rest, [nod], chain, scene)
    assert out.gain.shape == (rest.n,) and isinstance(out.log, list)
    assert [lb.name for lb in out.labels] == ["rest", "nod"]
    assert out.q.shape == rest.q.shape and np.array_equal(out.tau, rest.tau)


# ------------------------------------------------------------------ fix round 1: regressions
def _mj(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def _fwd_speed(q):
    return np.abs(np.diff(q, axis=0)) / DT


def test_a_held_head_tilt_over_a_moving_task_is_fitted(functional, chain, scene):
    """An offset held while the task moves has no activity of its own, but head_dq at the moving
    posture gives large joint rates: the fit must blame its realised joint offsets."""
    tr = resample_track(functional, np.arange(285, 542, dtype=float))  # a FREE window with the arm moving
    tilt = Contribution.empty("held_tilt", tr.n)
    tilt.head[:, 3] = np.radians(12) * _mj(tr.t / 1.0)
    tilt.head[:, 4] = np.radians(-8) * _mj(tr.t / 1.0)
    out = compose_track(tr, [tilt], chain, scene)
    assert tilt.scale < 1.0 and any(line.startswith("held_tilt: amplitude x") for line in out.log)
    assert _failed(out, chain, scene) == []
    assert (_fwd_speed(out.q) <= chain.vel_limits + 1e-9).all()  # the composite stays under the URDF limits
    assert out.log[-1].startswith("held_tilt") or "still over" not in " ".join(out.log)


def test_a_head_offset_held_over_the_whole_run_stays_inside_every_cap(functional, chain, scene):
    tr = functional
    gaze = Contribution.empty("gaze", tr.n)
    gaze.head[:, 5] = 0.3 * _mj(tr.t / 1.0)
    gaze.head[:, 4] = -0.15 * _mj(tr.t / 1.0)
    out = compose_track(tr, [gaze], chain, scene)
    assert gaze.scale < 1.0
    assert _failed(out, chain, scene) == []


def test_a_static_joint_offset_over_the_guard_is_fitted(home, chain, scene):
    q = home.copy()
    q[3] = -(URDF_GUARDS[3] - 0.05)  # 5 cm inside joint 4's guard
    rest = constant_track(q, 0.5, 2.0)
    c = Contribution.empty("static", rest.n)
    c.joints[:, 3] = -0.1
    out = compose_track(rest, [c], chain, scene)
    assert 0.0 < c.scale < 1.0
    assert np.abs(out.q[:, 3]).max() <= URDF_GUARDS[3] + 1e-9
    assert _failed(out, chain, scene) == []


def test_an_offset_held_while_the_task_drives_a_joint_toward_its_guard(home, chain, scene):
    from puppy_motion.track import concat

    q0, q1 = home.copy(), home.copy()
    q0[3], q1[3] = -1.9, -(URDF_GUARDS[3] - 0.03)
    task = concat([constant_track(q0, 0.5, 1.5), joint_move(q0, q1, 2.0, grip=0.5), constant_track(q1, 0.5, 1.0)])
    c = Contribution.empty("held", task.n)
    c.joints[:, 3] = -0.1 * _mj(task.t / 1.0)  # eased in, then held
    out = compose_track(task, [c], chain, scene)
    assert c.scale < 1.0
    assert np.abs(out.q[:, 3]).max() <= URDF_GUARDS[3] + 1e-9
    assert _failed(out, chain, scene) == []


def test_the_composite_urdf_speed_is_checked_and_held(home, chain, scene):
    task = joint_move(home, home + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55]), 0.9, grip=0.5)  # ~1.15 rad/s peak on joint 7
    fast = task.q.copy()
    fast[:, 6] += 0.5 * (-1.0) ** np.arange(task.n) * np.sin(np.pi * task.t / task.t[-1]) ** 2
    kinds = {v.kind for v in compose.cap_violations(chain, scene, task, fast, task.grip)}
    assert "composite joint speed" in kinds
    c = Contribution.empty("push", task.n)
    c.joints[:, 6] = 0.4 * _mj(task.t / task.t[-1])  # under its own 0.85 rad/s cap, but adds to the task's speed
    out = compose_track(task, [c], chain, scene)
    assert 0.0 < c.scale < 1.0 and any("composite joint speed" in line for line in out.log)
    assert (_fwd_speed(out.q) <= chain.vel_limits + 1e-9).all()
    assert _failed(out, chain, scene) == []


def test_clearance_broken_by_several_contributions_together_is_fitted(home, chain, scene):
    """Each offset alone eats 4 mm of the 5 mm the person floor allows; together they break it."""
    base = compose.clearances_of(chain, scene, home, False)
    floor = compose.clearance_floor(base)
    part = int(np.argmin(base[0]))
    grad = np.zeros(7)
    for k in range(4):
        step = np.zeros(7)
        step[k] = 1e-5
        grad[k] = (compose.clearances_of(chain, scene, home + step, False)[0, part] - base[0, part]) / 1e-5
    assert np.linalg.norm(grad) > 0.1
    each = -0.004 * grad / (grad @ grad)  # lowers that part's clearance by about 4 mm
    alone = compose.clearances_of(chain, scene, home + each, False)[0]
    both = compose.clearances_of(chain, scene, home + 2 * each, False)[0]
    assert (alone >= floor[0]).all() and (both < floor[0]).any()
    rest = constant_track(home, 0.5, 1.0)
    parts = []
    for name in "AB":
        c = Contribution.empty(name, rest.n)
        c.joints[:] = each
        parts.append(c)
    out = compose_track(rest, parts, chain, scene)
    assert all(c.scale < 1.0 for c in parts)
    assert _failed(out, chain, scene) == []
    clr = np.array([compose.clearances_of(chain, scene, q, False)[0] for q in out.q])
    assert (clr >= floor[0]).all()


def test_a_sample_to_sample_alternation_is_seen(home, chain, scene):
    rest = constant_track(home, 0.5, 3.0)
    buzz = Contribution.empty("buzz", rest.n)
    env = np.sin(np.pi * rest.t / rest.t[-1]) ** 2
    buzz.joints[:, 6] = 0.04 * env * (-1.0) ** np.arange(rest.n)
    raw = rest.q + buzz.joints
    assert "expressive joint speed" in {v.kind for v in compose.cap_violations(chain, scene, rest, raw)}
    out = compose_track(rest, [buzz], chain, scene)
    assert buzz.scale < 0.2
    assert _fwd_speed(out.q - rest.q).max() <= CAP_PRESETS["default"][FREE].joint_speed.max()
    assert _failed(out, chain, scene) == []


def test_a_contribution_that_cannot_be_fitted_ends_at_scale_zero(home, chain, scene, monkeypatch):
    rest = constant_track(home, 0.5, 2.0)
    bad = Contribution.empty("bad", rest.n)
    bad.joints[:, 6] = 0.04 * (-1.0) ** np.arange(rest.n)
    ok = Contribution.empty("ok", rest.n)
    ok.joints[:, 0] = _wave(rest.n, 2.0, 0.01)
    monkeypatch.setattr(compose, "FIT_ROUNDS", 0)  # no shrinking at all: only the fail-safe is left
    out = compose_track(rest, [bad, ok], chain, scene)
    assert bad.scale == 0.0
    assert any("switched off" in line and line.startswith("bad") for line in out.log)
    assert _failed(out, chain, scene) == []
    assert np.array_equal(out.q[:, 6], rest.q[:, 6])


def test_when_nothing_can_be_blamed_every_contribution_is_switched_off(home, chain, scene, monkeypatch):
    rest = constant_track(home, 0.5, 2.0)
    parts = []
    for name in ("a", "b"):
        c = Contribution.empty(name, rest.n)
        c.joints[:, 6] = 0.04 * (-1.0) ** np.arange(rest.n)
        parts.append(c)
    monkeypatch.setattr(compose._Fit, "blamed", lambda self, v: [])
    out = compose_track(rest, parts, chain, scene)
    assert all(c.scale == 0.0 for c in parts)
    assert np.array_equal(out.q, rest.q)
    assert _failed(out, chain, scene) == []
