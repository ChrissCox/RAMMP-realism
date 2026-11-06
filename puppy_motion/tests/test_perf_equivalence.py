"""The vectorized clearances and the batched compose gradients give what the original scalar code gave.

The reference below is a frozen copy of the pre-optimization implementation (geometry.distance,
arm_capsules, clearances and compose._Clearance); the live code must match it to 1e-9 m on random
postures, scenes, held on/off and NaN shapes, and compose must give the same track to 1e-9 rad."""

import os

import numpy as np
import pytest

import runs_helper
from puppy_motion import behaviors as B
from puppy_motion import compose as compose_mod
from puppy_motion import geometry
from puppy_motion.flatbot import chair_scene
from puppy_motion.geometry import PARTS, RADII, HELD_OFFSET, HELD_RADIUS, PART_LINKS, Box, Capsule, Scene, Sphere
from puppy_motion.stage import find_stage_poses
from puppy_motion.track import constant_track
from puppy_motion.compose import GRADIENT_STEP, CLEARANCE_CUTOFF_M

ATOL = 1e-9
LOOK_TARGET = np.array([1.2, -0.36, 0.9])  # standing head height in front of the seat (idle.LOOK_TARGET)


# ====================================================================== frozen reference
def _ref_point_box(points, box):
    local = (points - box.center) @ box.R
    out = np.maximum(np.abs(local) - box.half, 0.0)
    return np.linalg.norm(out, axis=1)


def _ref_seg_seg(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps * a * e else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _ref_seg_box(a, b, box, rounds=6, n=17):
    lo, hi = 0.0, 1.0
    best = np.inf
    for _ in range(rounds):
        ts = np.linspace(lo, hi, n)
        d = _ref_point_box(a + ts[:, None] * (b - a), box)
        i = int(np.argmin(d))
        best = min(best, float(d[i]))
        step = ts[1] - ts[0]
        lo, hi = max(0.0, ts[i] - step), min(1.0, ts[i] + step)
        if best == 0.0:
            break
    return best


def _ref_as_capsule(s):
    return Capsule(s.c, s.c, s.r) if isinstance(s, Sphere) else s


def _ref_distance(x, y):
    if isinstance(x, Box):
        x, y = y, x
    if isinstance(y, Box):
        c = _ref_as_capsule(x)
        return _ref_seg_box(c.a, c.b, y) - c.r
    cx, cy = _ref_as_capsule(x), _ref_as_capsule(y)
    return _ref_seg_seg(cx.a, cx.b, cy.a, cy.b) - cx.r - cy.r


def _ref_arm_capsules(chain, q, held):
    out = {}
    for part, links in PART_LINKS.items():
        pts = [chain.link_pose(q, name)[0] for name in links]
        out[part] = [Capsule(p0, p1, RADII[part]) for p0, p1 in zip(pts[:-1], pts[1:])]
    ee, R = chain.fk(q, "ee")
    tool, _ = chain.fk(q, "tool")
    head = [Capsule(ee, tool, RADII["head"])]
    if held:
        c = ee + HELD_OFFSET * R[:, 2]
        head.append(Capsule(c, c, HELD_RADIUS))
    out["head"] = head
    return out


def ref_clearances(chain, scene, q, held):
    caps = _ref_arm_capsules(chain, q, held)
    out = np.full((2, len(PARTS)), np.inf)
    for j, part in enumerate(PARTS):
        for row, shapes in enumerate((scene.person, scene.obstacles)):
            for c in caps[part]:
                for s in shapes:
                    d = _ref_distance(c, s)
                    out[row, j] = np.nan if np.isnan(d) or np.isnan(out[row, j]) else min(out[row, j], d)
    return out


def ref_clearances_of(chain, scene, q, held):
    return ref_clearances(chain, scene, np.asarray(q, dtype=float), bool(held))


class RefClearance:
    """compose._Clearance as it was: a memo, and 7 forward differences per gradient."""

    def __init__(self, chain, scene):
        self.chain, self.scene, self._memo = chain, scene, {}

    def __call__(self, q, held):
        key = (np.asarray(q, dtype=float).tobytes(), bool(held))
        if key not in self._memo:
            self._memo[key] = ref_clearances_of(self.chain, self.scene, q, held)
        return self._memo[key]

    def prefetch(self, qs, helds):  # the live class batches here; the original did not
        pass

    def prefetch_gradients(self, qs, helds):
        pass

    def gradients(self, q, held):
        base = self(q, held)[0]
        out = np.zeros((len(PARTS), 7))
        for k in range(7):
            step = np.zeros(7)
            step[k] = GRADIENT_STEP
            out[:, k] = (ref_clearances_of(self.chain, self.scene, q + step, held)[0] - base) / GRADIENT_STEP
        out[~(base < CLEARANCE_CUTOFF_M)] = 0.0
        return out


# ============================================================================== fixtures
def _spike():
    with open(os.path.join(runs_helper.DATA, "spike_person.json")) as f:
        return Scene.from_param(f.read())


SCENES = {"spike": _spike, "chair": chair_scene, "empty": lambda: Scene(person=[], obstacles=[])}


def _postures(n, seed):
    rng = np.random.default_rng(seed)
    near = np.array(runs_helper.HOME_JOINTS) + rng.normal(0, 0.5, (n // 2, 7))  # around rest, in reach of the person
    wide = rng.uniform(-np.pi, np.pi, (n - n // 2, 7))
    return np.vstack([near, wide])


@pytest.fixture(scope="module")
def postures():
    return _postures(300, 7)


# ================================================================================ clearances
@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("name", list(SCENES))
def test_clearances_match_the_frozen_implementation(chain, postures, name, held):
    scene = SCENES[name]()
    ref = np.array([ref_clearances(chain, scene, q, held) for q in postures])
    got = np.array([geometry.clearances(chain, scene, q, held) for q in postures])
    np.testing.assert_allclose(got, ref, rtol=0, atol=ATOL)
    assert np.array_equal(np.isinf(got), np.isinf(ref))
    batch = geometry.clearances_batch(chain, scene, postures, held)
    np.testing.assert_allclose(batch, ref, rtol=0, atol=ATOL)
    assert batch.shape == (len(postures), 2, len(PARTS))


def test_clearances_cover_contact_and_penetration(chain):
    """Postures that touch or sink into the shapes (negative and near-zero gaps) match too."""
    scene = chair_scene()
    rng = np.random.default_rng(3)
    qs = np.array(runs_helper.HOME_JOINTS) + rng.normal(0, 0.9, (150, 7))
    for held in (False, True):
        ref = np.array([ref_clearances(chain, scene, q, held) for q in qs])
        assert (ref < 0.0).any() and (ref > 0.0).any()
        np.testing.assert_allclose(geometry.clearances_batch(chain, scene, qs, held), ref, rtol=0, atol=ATOL)


def _mixed(person=(), obstacles=()):
    return Scene(person=list(person), obstacles=list(obstacles))


NAN = float("nan")
NAN_SCENES = {
    "nan radius": lambda: _mixed([Sphere([0.3, 0.2, 0.5], NAN)]),
    "nan sphere centre": lambda: _mixed([Sphere([NAN, 0.2, 0.5], 0.1)], [Sphere([0.3, 0.3, 0.3], 0.1)]),
    "nan capsule end": lambda: _mixed([Sphere([0.3, 0.2, 0.5], 0.1)], [Capsule([0, 0, 0], [0.1, NAN, 0.2], 0.05)]),
    "nan capsule radius": lambda: _mixed([Capsule([0, 0, 0], [0.1, 0.1, 0.2], NAN)]),
    "nan box centre": lambda: _mixed([Box([NAN, 0, 0.3], [0.1, 0.1, 0.1])], [Sphere([0.3, 0.3, 0.3], 0.1)]),
    "nan box half": lambda: _mixed([Sphere([0.3, 0.2, 0.5], 0.1)], [Box([0.2, 0, 0.3], [0.1, NAN, 0.1])]),
    "nan beside good shapes": lambda: _mixed(
        [Sphere([0.3, 0.2, 0.5], 0.1), Box([0.0, 0.3, 0.5], [0.1, 0.1, 0.1]), Sphere([NAN, 0, 0], 0.1)]
    ),
}


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("name", list(NAN_SCENES))
def test_nan_shapes_behave_as_before(chain, name, held):
    scene = NAN_SCENES[name]()
    qs = _postures(24, 11)
    ref = np.array([ref_clearances(chain, scene, q, held) for q in qs])
    got = np.array([geometry.clearances(chain, scene, q, held) for q in qs])
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got, ref, rtol=0, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(geometry.clearances_batch(chain, scene, qs, held), ref, rtol=0, atol=ATOL, equal_nan=True)


def test_a_nan_posture_gives_what_it_gave(chain):
    scene = chair_scene()
    q = np.array(runs_helper.HOME_JOINTS)
    q[2] = np.nan
    ref = ref_clearances(chain, scene, q, False)
    got = geometry.clearances(chain, scene, q, False)
    np.testing.assert_allclose(got, ref, rtol=0, atol=ATOL, equal_nan=True)
    assert np.array_equal(np.isnan(got), np.isnan(ref))


def test_a_scene_edited_in_place_is_not_served_stale(chain):
    scene = _mixed([Sphere([0.3, 0.2, 0.5], 0.1)])
    q = np.array(runs_helper.HOME_JOINTS)
    first = geometry.clearances(chain, scene, q, False)
    scene.person[0].c[:] = [0.1, 0.1, 0.4]
    scene.person.append(Sphere([0.0, 0.0, 0.8], 0.2))
    got = geometry.clearances(chain, scene, q, False)
    np.testing.assert_allclose(got, ref_clearances(chain, scene, q, False), rtol=0, atol=ATOL)
    assert not np.allclose(got, first)


def test_a_wrong_shaped_q_still_raises(chain):
    with pytest.raises(ValueError):
        geometry.clearances(chain, chair_scene(), np.zeros(6), False)


# ==================================================================================== compose
LENGTH = {"head_tilt": 5.0, "shake_off": 6.0, "wiggle": 5.0, "startle": 3.5}
BUILD = {
    "head_tilt": lambda c: [B.head_tilt(c, 0.3, 1.8, side=1)],
    "shake_off": lambda c: [B.shake_off(c, 0.3)],
    "wiggle": lambda c: [B.wiggle(c, 0.3)],
    "startle": lambda c: [B.startle(c, 0.3)],
}


def _compose(name, q0, scene, chain, zone):
    track = constant_track(q0, 0.0, LENGTH[name], zone, name)
    ctx = B.Context.of(track, chain, scene, B.Targets(person_head=LOOK_TARGET), 3)
    parts = BUILD[name](ctx)
    return compose_mod.compose(track, parts, chain, scene), parts


def _frozen(monkeypatch):
    monkeypatch.setattr(compose_mod, "_Clearance", RefClearance)
    monkeypatch.setattr(compose_mod, "clearances_of", ref_clearances_of)


def _stage_pose(chain):
    poses = find_stage_poses(chain, chair_scene(), n=1, seed=0)
    assert poses
    return poses[0]


@pytest.mark.parametrize("name", list(BUILD))
@pytest.mark.parametrize("where", ["rest", "stage"])
def test_compose_matches_the_frozen_implementation(chain, home, monkeypatch, name, where):
    from puppy_motion.track import Zone

    scene = chair_scene()
    q0 = home if where == "rest" else _stage_pose(chain)
    zone = Zone.NEAR_USER if geometry.clearances(chain, scene, q0, False)[0, 3] < 0.35 else Zone.FREE
    if where == "rest":
        zone = Zone.NEAR_USER  # the slow case, whatever the head reads
    new, new_parts = _compose(name, q0, scene, chain, zone)
    with monkeypatch.context() as m:
        _frozen(m)
        old, old_parts = _compose(name, q0, scene, chain, zone)
    np.testing.assert_allclose(new.q, old.q, rtol=0, atol=1e-9)
    np.testing.assert_allclose(new.grip, old.grip, rtol=0, atol=1e-9)
    assert [p.scale for p in new_parts] == pytest.approx([p.scale for p in old_parts], abs=1e-9)
    assert new.log == old.log
    if name != "wiggle":  # wiggle never acts near the user: its rest case is a (still equal) no-op
        assert np.abs(new.q - new.func.q).max() > 1e-4  # it really moved


def test_compose_near_the_user_matches_with_a_person_in_reach(chain, home, monkeypatch):
    from puppy_motion.track import Zone

    scene = _spike()
    new, _ = _compose("wiggle", home, scene, chain, Zone.NEAR_USER)
    with monkeypatch.context() as m:
        _frozen(m)
        old, _ = _compose("wiggle", home, scene, chain, Zone.NEAR_USER)
    np.testing.assert_allclose(new.q, old.q, rtol=0, atol=1e-9)
    assert new.log == old.log
