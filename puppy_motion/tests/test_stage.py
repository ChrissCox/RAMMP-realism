import numpy as np
import pytest

from puppy_motion.flatbot import chair_scene
from puppy_motion.stage import (
    REST_Q, find_stage_poses, min_jerk_path, path_violations, points_from_q, retime_path, stage_limits,
    stage_violations,
)
from puppy_motion.track import DT


def rules(vs):
    return {v.rule for v in vs}


@pytest.fixture(scope="module")
def poses(chain):
    return find_stage_poses(chain, chair_scene(), n=5, seed=0)


def test_found_stage_poses_meet_every_rule(chain, poses):
    assert poses and all(stage_violations(chain, chair_scene(), q) == [] for q in poses)


def test_stage_poses_are_sorted_by_head_travel_and_distinct(chain, poses):
    rest = chain.fk(REST_Q, "tool")[0]
    d = [np.linalg.norm(chain.fk(q, "tool")[0] - rest) for q in poses]
    assert d == sorted(d)
    assert all(np.abs(a - b).max() > 0.15 for i, a in enumerate(poses) for b in poses[:i])


def test_search_is_deterministic(chain, poses):
    again = find_stage_poses(chain, chair_scene(), n=5, seed=0)
    assert len(again) == len(poses) and all(np.array_equal(a, b) for a, b in zip(again, poses))


def test_rest_is_not_a_stage_pose(chain, home):
    assert "stage_person" in rules(stage_violations(chain, chair_scene(), home))


def test_each_stage_rule_can_fire(chain, poses):
    q = poses[0].copy()
    lim = stage_limits(chain)
    q[1] = lim[1] + 0.05
    assert "stage_guard" in rules(stage_violations(chain, chair_scene(), q))
    q = poses[0].copy()
    q[5] += 0.9
    assert rules(stage_violations(chain, chair_scene(), q)) & {"stage_gaze", "stage_up"}
    # malformed input is a violation, not a crash
    assert stage_violations(chain, chair_scene(), np.zeros(3))
    assert stage_violations(chain, chair_scene(), np.full(7, np.nan))


def test_gaze_and_up_rules_name_themselves(chain):
    # the home pose looks somewhere else entirely; a random joint-space pose breaks gaze or up
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        seen |= rules(stage_violations(chain, chair_scene(), rng.uniform(-1.5, 1.5, 7)))
    assert {"stage_gaze", "stage_up", "stage_person"} <= seen


@pytest.fixture(scope="module")
def slow0(chain, poses):
    return retime_path(min_jerk_path(REST_Q, poses[0], 1.0), chain)


@pytest.fixture(scope="module")
def slow0_violations(chain, slow0):
    return path_violations(slow0, chain, chair_scene())  # ~1000 samples of clearances: compute once


def test_retime_only_slows_and_meets_the_caps(chain, poses, slow0, slow0_violations):
    fast = min_jerk_path(REST_Q, poses[0], 1.0)
    assert slow0[-1][0] > fast[-1][0]
    assert not rules(slow0_violations) & {"path_caps", "path_ends"}
    assert np.allclose(slow0[0][1], REST_Q) and np.allclose(slow0[-1][1], poses[0])
    assert "path_caps" in rules(path_violations(fast, chain, chair_scene()))  # the 1 s move really is too fast


def test_retime_leaves_a_slow_enough_path_alone(chain, slow0):
    again = retime_path(slow0, chain)
    assert len(again) == len(slow0)
    assert np.allclose([p[1] for p in again], [p[1] for p in slow0], atol=1e-9)


def test_retime_rejects_an_unknown_preset(chain, poses):
    with pytest.raises(ValueError):
        retime_path(min_jerk_path(REST_Q, poses[0], 1.0), chain, caps="nope")


def test_a_path_that_dips_back_toward_the_person_is_refused(chain, poses):
    # rest -> stage -> back toward rest halfway -> stage again
    mid = 0.5 * (REST_Q + poses[0])
    qs = np.vstack([
        [p[1] for p in min_jerk_path(REST_Q, poses[0], 1.0)],
        [p[1] for p in min_jerk_path(poses[0], mid, 1.0)][1:],
        [p[1] for p in min_jerk_path(mid, poses[0], 1.0)][1:],
    ])
    assert "path_person_monotone" in rules(path_violations(points_from_q(qs), chain, chair_scene()))


def test_a_path_that_heads_toward_the_person_is_refused(chain, poses):
    # stage -> rest: the clearance drops below its first sample, and the head nears the chair
    v = rules(path_violations(min_jerk_path(poses[0], REST_Q, 0.4), chain, chair_scene()))
    assert "path_person_floor" in v


def test_path_ends_must_be_at_rest(chain, poses):
    pts = min_jerk_path(REST_Q, poses[0], 0.4)
    pts[0] = (pts[0][0], pts[0][1], np.full(7, 0.01))
    assert "path_ends" in rules(path_violations(pts, chain, chair_scene()))
    assert "path_ends" in rules(path_violations([], chain, chair_scene()))
    assert "path_ends" in rules(path_violations([(0.0, REST_Q, np.zeros(7))], chain, chair_scene()))


def test_a_min_jerk_path_from_rest_to_a_found_stage_pose_is_accepted_or_names_its_rule(slow0_violations):
    known = {"path_person_monotone", "path_person_floor", "path_obstacle", "path_caps", "path_ends"}
    assert rules(slow0_violations) <= known
    for v in slow0_violations:
        print("named:", v.rule, v.sample, v.detail)


def test_min_jerk_path_is_on_the_clock(poses):
    pts = min_jerk_path(REST_Q, poses[0], 2.0)
    assert len(pts) == 101 and abs(pts[1][0] - DT) < 1e-12
    assert not pts[0][2].any() and not pts[-1][2].any()
