import numpy as np
import pytest

from puppy_motion.escape import escape_path, unwrap_to
from puppy_motion.flatbot import chair_scene
from puppy_motion.geometry import PARTS, clearances
from puppy_motion.stage import find_stage_poses, path_violations, points_from_q

# Of find_stage_poses(n=3, seed=0), only the third has a route from rest that keeps the forearm,
# wrist and head 0.35 m from the person all the way (the first two swing the shoulder over the
# occupant); escape_path raises ValueError for those rather than return a path the rules refuse.
REACHABLE = 2


@pytest.fixture(scope="module")
def poses(chain):
    return find_stage_poses(chain, chair_scene(), n=3, seed=0)


@pytest.fixture(scope="module")
def path(chain, home, poses):
    return escape_path(chain, chair_scene(), home, poses[REACHABLE])


def test_escape_path_from_rest_to_a_stage_pose_passes_the_path_rules(chain, home, poses, path):
    assert path_violations(path, chain, chair_scene()) == []
    assert path[0][0] == 0.0 and np.abs(path[0][1] - home).max() == 0.0
    assert np.abs(path[-1][1] - unwrap_to(chain, home, poses[REACHABLE])).max() == 0.0
    assert np.allclose(np.sin(path[-1][1]), np.sin(poses[REACHABLE]))  # the same posture, whole turns apart


def test_escape_never_lowers_a_part_and_ends_clear(chain, path):
    scene = chair_scene()
    c = np.array([clearances(chain, scene, q, False)[0] for _, q, _ in path])
    assert np.all(c >= c[0])
    assert np.all(c[-1, 1:] >= 0.35)
    print("duration s", path[-1][0], "min person clearance per part", dict(zip(PARTS, c.min(axis=0).round(4))))


def test_the_other_stage_poses_have_no_route_and_raise(chain, home, poses):
    # one is enough to show the failure is an error, not a bad path (each failed search takes ~1 min)
    with pytest.raises(ValueError):
        escape_path(chain, chair_scene(), home, poses[0])


def test_a_reversed_escape_path_keeps_caps_and_ends_but_not_the_forward_floors(chain, path):
    """rest replays the stored path backwards: the same postures, in reverse. The floor rules are
    forward-only (the reversed path ends back in the torso box); caps and ends still hold."""
    rev = points_from_q(np.array([q for _, q, _ in path])[::-1])
    rules = {v.rule for v in path_violations(rev, chain, chair_scene())}
    assert not rules & {"path_caps", "path_ends"}
    assert rules <= {"path_person_floor", "path_person_monotone", "path_obstacle"}
    assert len(rev) == len(path)


def test_the_shipped_stage_pose_has_a_checked_escape_path_with_the_candidate_margin(chain, home):
    import json
    from pathlib import Path

    from puppy_motion.escape import grown_scene
    from puppy_motion.stage import stage_violations

    doc = json.loads((Path(__file__).resolve().parents[2] / "puppy_playground/puppy_playground/data/stage_pose.json").read_text())
    q = np.array(doc["q"])
    assert stage_violations(chain, chair_scene(), q) == []
    p = escape_path(chain, chair_scene(), home, q)
    assert path_violations(p, chain, grown_scene(chair_scene(), 0.002)) == []
    assert np.abs(p[-1][1] - q).max() < 1e-6 and "search_index" in doc


def test_grown_scene_never_reads_more_clearance_than_the_margin_allows(chain, path):
    from puppy_motion.escape import grown_scene

    scene, grown = chair_scene(), grown_scene(chair_scene(), 0.002)
    for _, q, _ in path[::200]:
        a, b = clearances(chain, scene, q, False), clearances(chain, grown, q, False)
        assert np.all(b <= a - 0.002 + 1e-6)
