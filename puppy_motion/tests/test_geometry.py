import json
import os

import numpy as np
import pytest

import runs_helper
from puppy_motion.geometry import (
    PARTS, Box, Capsule, Scene, Sphere, arm_capsules, clearances, distance,
)
from puppy_motion.urdf import Chain

DATA = os.path.join(os.path.dirname(__file__), "data")
SPIKE_MIN = {"upper_arm": 0.068, "forearm": 0.108, "wrist": 0.246, "head": 0.377}


def _chain():
    with open(os.path.join(DATA, runs_helper.RUN_NAMES[0] + ".json")) as f:
        return Chain.from_urdf(json.load(f)["urdf"])


def test_capsule_sphere_distance_closed_form():
    cap = Capsule([0, 0, 0], [1, 0, 0], 0.1)
    cases = [
        (Sphere([0.5, 1.0, 0], 0.2), 0.7),                 # beside the middle
        (Sphere([2.0, 0, 0], 0.25), 0.65),                 # beyond end b
        (Sphere([-1.0, 0, 0], 0.4), 0.5),                  # beyond end a
        (Sphere([0.5, 0.3, 0], 0.2), 0.0),                 # touching
        (Sphere([0.5, 0.2, 0], 0.2), -0.1),                # penetrating
        (Sphere([4.0, 4.0, 0], 0.0), 5.0 - 0.1),          # 3-4-5 from end b
    ]
    for sph, expect in cases:
        assert abs(distance(cap, sph) - expect) < 1e-12
        assert abs(distance(sph, cap) - expect) < 1e-12


def test_capsule_capsule_distance():
    a = Capsule([0, 0, 0], [1, 0, 0], 0.1)
    b = Capsule([0.5, 1, -1], [0.5, 1, 1], 0.2)  # crosses above at y=1
    assert abs(distance(a, b) - 0.7) < 1e-12
    c = Capsule([2, 0, 0], [3, 0, 0], 0.1)       # collinear, gap 1
    assert abs(distance(a, c) - 0.8) < 1e-12


def _box_surface(box, n=61):
    g = np.linspace(-1, 1, n)
    u, v = np.meshgrid(g, g)
    u, v = u.ravel(), v.ravel()
    one = np.ones_like(u)
    pts = []
    for sgn in (-1, 1):
        pts += [np.c_[sgn * one, u, v], np.c_[u, sgn * one, v], np.c_[u, v, sgn * one]]
    local = np.vstack(pts) * box.half
    return box.center + local @ box.R.T


def test_capsule_box_distance_matches_brute_force():
    rng = np.random.default_rng(3)
    tested = 0
    while tested < 200:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        box = Box(rng.uniform(-0.3, 0.3, 3), rng.uniform(0.05, 0.15, 3), R)
        cap = Capsule(rng.uniform(-0.8, 0.8, 3), rng.uniform(-0.8, 0.8, 3), rng.uniform(0.02, 0.08))
        d = distance(cap, box)
        if d <= 0.02:
            continue  # penetrating or touching pairs are not what the surface sampling measures
        tested += 1
        surf = _box_surface(box)
        a, b = cap.a, cap.b
        ab = b - a
        t = np.clip(((surf - a) @ ab) / (ab @ ab), 0, 1)
        d_brute = np.min(np.linalg.norm(surf - (a + t[:, None] * ab), axis=1)) - cap.r
        assert abs(d - d_brute) < 2e-3


def test_capsule_box_penetrating_is_negative():
    box = Box([0, 0, 0], [0.1, 0.1, 0.1])
    assert distance(Capsule([-1, 0, 0], [1, 0, 0], 0.05), box) < 0
    assert distance(Capsule([0.3, 0, 0], [0.3, 1, 0], 0.05), box) == pytest.approx(0.2 - 0.05)


def test_scene_param_without_person_raises():
    for text in ('{}', '{"obstacles": {"spheres": [[0,0,0,1]]}}', '{"person": {}}',
                 '{"person": {"spheres": [], "capsules": [], "boxes": []}}'):
        with pytest.raises(ValueError):
            Scene.from_param(text)


def test_scene_param_parses_and_finds_head():
    scene = Scene.from_param(json.dumps({
        "person": {"spheres": [[0, 0, 0.2, 0.1], [0, 0, 0.6, 0.09]], "capsules": [[0, 0, 0, 0, 0, 0.3, 0.1]],
                   "boxes": [[0, 0, 0, 0.1, 0.1, 0.1]]},
        "obstacles": {"boxes": [[1, 0, 0, 0.1, 0.1, 0.1]]}}))
    assert len(scene.person) == 4 and len(scene.obstacles) == 1
    assert np.allclose(scene.person_head(), [0, 0, 0.6])


def test_link_pose_agrees_with_fk_and_names():
    chain = _chain()
    q = np.array(runs_helper.HOME_JOINTS)
    p, R = chain.link_pose(q, "arm_end_effector_link")
    p2, R2 = chain.fk(q, "ee")
    assert np.allclose(p, p2) and np.allclose(R, R2)
    p3, _ = chain.link_pose(q, "bracelet_link")
    assert np.linalg.norm(p3 - p) == pytest.approx(0.061525)
    with pytest.raises(ValueError):
        chain.link_pose(q, "no_such_link")


def test_arm_capsules_structure_and_held_sphere():
    chain = _chain()
    q = np.array(runs_helper.HOME_JOINTS)
    free = arm_capsules(chain, q, held=False)
    held = arm_capsules(chain, q, held=True)
    assert tuple(free) == PARTS
    assert [c.r for c in free["forearm"]] == [0.055]
    assert all(c.r == 0.06 for c in free["upper_arm"]) and all(c.r == 0.05 for c in free["wrist"])
    assert len(held["head"]) == len(free["head"]) + 1
    ee, R = chain.fk(q, "ee")
    tool, _ = chain.fk(q, "tool")
    h = held["head"][-1]
    assert h.r == 0.09 and np.allclose(h.a, ee + 0.17 * R[:, 2]) and np.allclose(h.b, h.a)
    assert free["head"][0].r == 0.06
    assert np.allclose(free["head"][0].a, ee) and np.allclose(free["head"][0].b, tool)


def test_clearances_shape_and_no_obstacles_is_inf():
    chain = _chain()
    with open(os.path.join(DATA, "spike_person.json")) as f:
        scene = Scene.from_param(f.read())
    c = clearances(chain, scene, np.array(runs_helper.HOME_JOINTS), False)
    assert c.shape == (2, 4) and np.all(np.isfinite(c[0])) and np.all(np.isposinf(c[1]))


def test_spike_person_clearances_reproduced():
    chain = _chain()
    with open(os.path.join(DATA, "spike_person.json")) as f:
        scene = Scene.from_param(f.read())
    track = runs_helper.functional_track(runs_helper.RUN_NAMES[0])
    mins = np.full(4, np.inf)
    for q, held in list(zip(track.q, track.held))[::4]:
        mins = np.minimum(mins, clearances(chain, scene, q, bool(held))[0])
    for part, m in zip(PARTS, mins):
        assert abs(m - SPIKE_MIN[part]) < 0.03, (part, m, SPIKE_MIN[part])
