"""The beat validator: a composed head_tilt at home passes, and each rule has a minimal mutation of
that beat that breaks exactly that rule (and nothing else)."""

import copy

import numpy as np
import pytest

from puppy_motion import behaviors as B
from puppy_motion import caps as C
from puppy_motion import validate as V
from puppy_motion.behaviors import Context, Targets
from puppy_motion.caps import CAP_PRESETS, CapSet
from puppy_motion.compose import compose
from puppy_motion.geometry import Scene, Sphere, clearances
from puppy_motion.track import DT, Zone, constant_track
from puppy_motion.validate import Violation, check_beat

REST_GRIP = 0.79
BEAT_S = 7.0
REST_AT = 3.0  # s: the head_tilt is done; the arm rests from here, room for a mutation's bump
TILT = (0.5, 1.5)


def _points(q):
    """A beat's points from its positions: velocities are the forward differences (zero at the end)."""
    dq = np.vstack([np.diff(q, axis=0) / DT, np.zeros((1, q.shape[1]))])
    return [(k * DT, q[k].copy(), dq[k]) for k in range(len(q))]


def _rules(points, q_start, chain, scene, zone=Zone.FREE, **kw):
    return [v.rule for v in check_beat(points, q_start, chain, scene, zone, **kw)]


def _beat_q(q_home, chain, scene, zone=Zone.FREE):
    track = constant_track(q_home, REST_GRIP, BEAT_S, zone)
    ctx = Context.of(track, chain, scene, Targets(scene.person_head()))
    return compose(track, [B.head_tilt(ctx, *TILT)], chain, scene).q


@pytest.fixture(scope="module")
def q_beat(home, chain, scene):
    return _beat_q(home, chain, scene)


@pytest.fixture
def beat(q_beat):
    return q_beat.copy()


def _bump(q, joint, peak_speed, seconds, at=REST_AT + 0.5):
    """Adds a (1 - cos) bump to one joint over [at, at + seconds]: peak speed peak_speed, peak
    acceleration 2 pi peak_speed / seconds, zero speed at both ends."""
    q = q.copy()
    k0, m = int(round(at / DT)), int(round(seconds / DT))
    x = np.arange(m + 1) / m
    amp = peak_speed * seconds / np.pi
    q[k0 : k0 + m + 1, joint] += amp / 2 * (1 - np.cos(2 * np.pi * x))
    return q


def test_the_composed_beat_passes(beat, home, chain, scene):
    assert np.abs(beat - home).max() > 0.01  # it does move
    assert check_beat(_points(beat), home, chain, scene, Zone.FREE) == []


def test_a_beat_composed_in_near_user_passes_the_near_user_caps(home, chain, scene):
    q = _beat_q(home, chain, scene, Zone.NEAR_USER)
    assert check_beat(_points(q), home, chain, scene, Zone.NEAR_USER) == []
    assert not np.allclose(q, home)


def test_the_zone_picks_the_caps(beat, home, chain, scene):
    # the FREE beat's own motion is over the gentler NEAR_USER caps' accel or head speed
    assert set(_rules(_points(beat), home, chain, scene, Zone.NEAR_USER)) & {"accel", "head_speed", "speed"}


# ------------------------------------------------------------------ one mutation per rule
def test_start(beat, home, chain, scene):
    beat[0] += 1e-6  # the first point is off q_start
    assert _rules(_points(beat), home, chain, scene) == ["start"]


def test_start_needs_zero_velocity(beat, chain, scene, home):
    pts = _points(beat)
    pts[0] = (pts[0][0], pts[0][1], pts[0][2] + 1e-6)
    assert _rules(pts, home, chain, scene) == ["start"]


def test_end(beat, home, chain, scene):
    pts = _points(beat)
    pts[-1] = (pts[-1][0], pts[-1][1], pts[-1][2] + 1e-6)  # still moving at the end
    assert _rules(pts, home, chain, scene) == ["end"]
    beat[-1] += 1e-6  # or somewhere else
    assert _rules(_points(beat), home, chain, scene) == ["end"]


def test_dt(beat, home, chain, scene):
    pts = [(t * 1.01, q, dq) for t, q, dq in _points(beat)]
    assert _rules(pts, home, chain, scene) == ["dt"]


def test_velocity_the_points_carry_must_agree_with_the_positions(beat, home, chain, scene):
    pts = _points(beat)
    k = int(2.0 / DT)
    pts[k] = (pts[k][0], pts[k][1], pts[k][2] + 0.2)
    assert _rules(pts, home, chain, scene) == ["velocity"]


def test_speed(beat, home, chain, scene):
    # joint 7 at 1.0 rad/s: over its 0.85 cap, under the URDF's 1.22, and gentle enough on accel
    assert _rules(_points(_bump(beat, 6, 1.0, 2.5)), home, chain, scene) == ["speed"]


def test_accel(beat, home, chain, scene):
    # 0.6 rad/s is fine, but reached in 0.8 s it needs 4.7 rad/s^2 against the 3.0 cap
    assert _rules(_points(_bump(beat, 6, 0.6, 0.8)), home, chain, scene) == ["accel"]


def test_head_speed(beat, home, chain, scene):
    # joints 2, 4 and 6 each inside their caps, together the eye/tool point is over 0.40 m/s
    q = _bump(_bump(_bump(beat, 1, 0.8, 2.0), 3, 0.6, 2.0), 5, 0.8, 2.0)
    assert _rules(_points(q), home, chain, scene) == ["head_speed"]


def test_urdf_speed(beat, home, chain, scene, monkeypatch):
    # the URDF limits sit above every cap, so only a looser preset isolates them
    loose = CapSet(np.full(7, 10.0), np.full(7, 1e3), 10.0)
    monkeypatch.setitem(CAP_PRESETS, "loose", {z: loose for z in Zone})
    pts = _points(_bump(beat, 6, 1.3, 2.5))  # joint 7: over 1.2218
    assert _rules(pts, home, chain, scene, caps="loose") == ["urdf_speed"]
    assert set(_rules(pts, home, chain, scene)) >= {"speed", "urdf_speed"}  # the default caps catch it too


def test_limits(beat, home, chain, scene):
    shift = np.zeros(7)
    shift[3] = -0.33  # joint 4 from -2.27 to -2.60, past its +-2.57 range
    q = beat + shift
    assert _rules(_points(q), home + shift, chain, scene) == ["limits"]


def test_guards(beat, home, chain, scene):
    shift = np.zeros(7)
    shift[3] = -0.28  # joint 4 to -2.55: inside the +-2.57 range, past RAMMP's 144 degrees (2.513)
    q = beat + shift
    assert abs(q[:, 3]).max() < chain.pos_limits[3, 1]
    assert _rules(_points(q), home + shift, chain, scene) == ["guards"]


def test_person(beat, home, chain, scene):
    task = clearances(chain, scene, home, False)
    task[0] = 0.2  # the task had 0.2 m to the person; the beat's arm sits at 0.07
    assert _rules(_points(beat), home, chain, scene, task_clearance=task) == ["person"]


def test_person_floor_is_5mm_under_the_task_clearance(beat, home, chain, scene):
    task = clearances(chain, scene, home, False)
    pts = _points(beat)
    seen = np.array([clearances(chain, scene, q, False)[0] for _, q, _ in pts]).min(axis=0)
    assert _rules(pts, home, chain, scene, task_clearance=task) == []
    task[0] = seen + V.PERSON_SLACK_M - 1e-6
    assert _rules(pts, home, chain, scene, task_clearance=task) == []
    task[0] = seen + V.PERSON_SLACK_M + 1e-3
    assert _rules(pts, home, chain, scene, task_clearance=task) == ["person"]


def _obstacle_gap(chain, scene, home, gap):
    """A scene with a point obstacle `gap` m from the head capsule's side (the other parts are far)."""
    ee, tool = chain.fk(home, "ee")[0], chain.fk(home, "tool")[0]
    axis = (tool - ee) / np.linalg.norm(tool - ee)
    side = np.cross(axis, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    centre = (ee + tool) / 2 + side * (0.065 + gap)
    return Scene(scene.person, [Sphere(centre, 0.0)])


def test_obstacle(beat, home, chain, scene):
    scene2 = _obstacle_gap(chain, scene, home, 0.01)
    task = clearances(chain, scene2, home, False)
    task[1] = 0.05  # the task had 5 cm to it; the head sits 1 cm from it
    assert _rules(_points(beat), home, chain, scene2, task_clearance=task) == ["obstacle"]


def test_obstacle_floor_has_the_composers_half_millimetre_tolerance(beat, home, chain, scene):
    assert V.OBSTACLE_TOL_M == 0.0005 and V.OBSTACLE_KEEP_M == 0.03
    still = _points(np.tile(home, (20, 1)))
    task = clearances(chain, scene, home, False)
    task[1] = 0.03
    for gap, expect in ((0.0296, []), (0.0294, ["obstacle"])):
        scene2 = _obstacle_gap(chain, scene, home, gap)
        assert clearances(chain, scene2, home, False)[1].min() == pytest.approx(gap, abs=1e-6)
        assert _rules(still, home, chain, scene2, task_clearance=task) == expect


# ------------------------------------------------------------------------- safe failure
def test_a_zig_zag_between_samples_is_seen(beat, home, chain, scene):
    q = beat.copy()
    k = int(3.5 / DT)
    q[k : k + 20, 6] += 0.01 * (-1.0) ** np.arange(20)  # central differences would cancel this
    q[k + 20 :, 6] = beat[k + 20 :, 6]
    rules = _rules(_points(q), home, chain, scene)
    assert "speed" in rules and "accel" in rules


def test_no_beat_plays_in_precision(home, chain, scene):
    still = _points(np.tile(home, (20, 1)))
    assert _rules(still, home, chain, scene, Zone.PRECISION) == ["zone"]
    assert _rules(still, home, chain, scene, Zone.FREE) == []


def test_malformed_beats_fail_instead_of_raising(home, chain, scene, beat):
    assert _rules([], home, chain, scene) == ["shape"]
    assert _rules(_points(beat)[:1], home, chain, scene) == ["shape"]
    pts = _points(beat)
    pts[10] = (pts[10][0], pts[10][1] * np.nan, pts[10][2])
    assert _rules(pts, home, chain, scene) == ["shape"]
    assert _rules([(0.0, np.zeros(6), np.zeros(6))] * 3, home, chain, scene) == ["shape"]


def test_unknown_preset_is_an_error(beat, home, chain, scene):
    with pytest.raises(ValueError):
        check_beat(_points(beat), home, chain, scene, Zone.FREE, caps="wild")


def test_violation_carries_rule_sample_and_detail(beat, home, chain, scene):
    (v,) = check_beat(_points(_bump(beat, 6, 0.6, 0.8)), home, chain, scene, Zone.FREE)
    assert isinstance(v, Violation) and v.rule == "accel" and 0 < v.sample < len(beat) and v.detail
