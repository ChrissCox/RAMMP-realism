"""The playground previews: flatbot's chair scene, the stage still, and the driver model's
positions+velocities execution of every expression at size 1.0."""

import hashlib
import json

import numpy as np
import pytest

from puppy_sim import checks, dynamics, render, scene
from puppy_sim.clips import playground_expressions as pe
from puppy_sim.clips import playground_stage as ps
from puppy_sim.clips import registry
from puppy_sim.clips.playground_common import STAGE_POSE_JSON, Assembler, chain, chair, points_q, stage_pose, urdf_text
from puppy_sim.flatbot_scene import FLATBOT

CLIPS = ("playground_stage", "playground_expressions", "playground_auto")


def test_clips_registered_and_pass_check():
    clips = registry()
    for name in CLIPS:
        assert name in clips and clips[name].run == FLATBOT
        for _, _, _, results, _ in checks.check_named(name):
            assert checks.passed(results), [c for c in results if not c.ok]


def test_stage_still_exists_and_pose_meets_the_rules():
    from puppy_motion import stage

    assert ps.STILL_PATH.exists() and ps.STILL_PATH.stat().st_size > 20_000
    doc = json.loads(STAGE_POSE_JSON.read_text())
    q = np.array(doc["q"])
    assert q.shape == (7,)
    assert doc["urdf_sha256"] == hashlib.sha256(urdf_text().encode("utf-8")).hexdigest()
    assert doc["rules"].startswith("stage.py@")
    assert np.allclose(q, stage_pose(), atol=1e-12)  # the shipped pose is what the clips use
    assert stage.stage_violations(chain(), chair(), q) == []


def test_flatbot_scene_draws_the_layer_boxes_without_a_mannequin():
    sc = scene.build(FLATBOT)
    m = sc.model
    names = {m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)}
    assert "person" in names and "chair" in names
    assert len(sc.groups["person"]) == len(chair().person) and len(sc.groups["chair"]) == len(chair().obstacles)
    assert not any(m.geom(g).name in ("head", "torso") for g in range(m.ngeom))
    assert (m.mat_rgba[m.geom_matid[sc.groups["person"]]][:, 3] < 0.5).all()  # translucent


def test_positions_velocities_reference_is_cubic_hermite_with_central_velocities():
    q = np.cumsum(np.random.default_rng(1).normal(0, 0.01, (12, 7)), axis=0)
    ref, qd, qdd = dynamics.reference(type("T", (), {"q": q})(), dynamics.VELOCITIES)
    assert np.allclose(ref[:: dynamics.STEPS_PER_SAMPLE], q)  # passes through every waypoint
    assert not qdd.any()
    mid = 5 * dynamics.STEPS_PER_SAMPLE
    assert np.allclose(qd[mid], (q[6] - q[4]) / (2 * dynamics.DT), atol=1e-6)
    assert not qd[0].any() and not qd[-1].any()


def _head_floor_margins(track_q, q_cmd):
    """Least margin (m) of the achieved q over check_beat's clearance floors (set by the start)."""
    from puppy_motion.geometry import clearances
    from puppy_motion.validate import NEAR_USER_M, OBSTACLE_KEEP_M, OBSTACLE_TOL_M, PERSON_SLACK_M

    ch, sc = chain(), chair()
    c0 = clearances(ch, sc, q_cmd[0], False)
    floor_p = np.minimum(c0[0], NEAR_USER_M) - PERSON_SLACK_M
    floor_o = np.minimum(c0[1], OBSTACLE_KEEP_M) - OBSTACLE_TOL_M
    person, obstacle = np.inf, np.inf
    for q in track_q:
        c = clearances(ch, sc, q, False)
        person, obstacle = min(person, float((c[0] - floor_p).min())), min(obstacle, float((c[1] - floor_o).min()))
    return person, obstacle


def test_driver_model_overshoot_with_positions_and_velocities_is_within_the_checks():
    """Every expression at size 1.0 under impedance 'default', sent as positions + central-difference
    velocities (no accelerations): the arm's own clearance never drops under check_beat's floors."""
    sc = scene.build(FLATBOT)
    ctl = dynamics.control("impedance:default", feedforward=dynamics.VELOCITIES)
    beats, _ = pe.expression_beats(1.0)
    assert all(b is not None for b in beats.values())
    for name, (points, _) in beats.items():
        a = Assembler(points[0][1])
        a.add(points_q(points))
        track = a.track(name)
        ach = dynamics.achieved(track, ctl, sc)
        person, obstacle = _head_floor_margins(ach.q[:: 2], track.q[:: 2])
        assert person >= 0.0 and obstacle >= 0.0, (name, person, obstacle)


def _candidate_file(tmp_path, mutate=None):
    """A candidate written by stage_store.save from the repo's own path (optionally mutated)."""
    from puppy_sim.clips.playground_common import stage_store

    _, q_stage, points, _, _ = ps.plan()
    points = [(t, q.copy(), dq.copy()) for t, q, dq in points]
    if mutate:
        mutate(points)
    path = tmp_path / "stage_path.candidate.json"
    stage_store().save(path, points, urdf_text(), q_stage, method="escape")
    return path


def test_candidate_file_renders_for_approval(tmp_path):
    clip = ps.make_clip(_candidate_file(tmp_path))
    assert "REVIEW FOR APPROVAL" in clip.title and "escape" in clip.title and "NOT APPROVABLE" not in clip.title
    assert clip.candidate.violations == []
    import hashlib

    sha = hashlib.sha256(_candidate_file(tmp_path).read_bytes()).hexdigest()
    assert sha[:12] in clip.title and len(clip.candidate.sha256) == 64
    assert clip.name == "playground_stage_candidate"  # not the committed clip's output name
    track = clip.build()
    assert track.log[-1].startswith("path by candidate file")
    assert "NOT APPROVABLE" not in " ".join(clip.subtitle_for([track]))


def test_candidate_with_a_violating_path_is_not_approvable(tmp_path):
    def into_the_person(points):  # swing the middle of the path far from the planned route
        mid = len(points) // 2
        t, q, dq = points[mid]
        points[mid] = (t, q + 1.5, dq)

    clip = ps.make_clip(_candidate_file(tmp_path, into_the_person))
    assert "NOT APPROVABLE" in clip.title and "REVIEW FOR APPROVAL" not in clip.title
    assert clip.candidate.violations
    assert "NOT APPROVABLE" in " ".join(clip.subtitle_for([clip.build()]))


def test_unreadable_candidate_is_not_approvable(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    clip = ps.make_clip(bad)
    assert "NOT APPROVABLE" in clip.title
    with pytest.raises(ValueError):
        clip.build()


USERS_POSE = [0.0531, 0.1366, -3.1161, -2.3040, -0.6222, 1.3713, 1.7233]  # upper arm 44 mm from the occupant box


def _pose_candidate(tmp_path, cap, name="stage_path.candidate.json"):
    """A min_jerk path from the user's pose to the stage pose, saved with person_floor_cap_m=cap."""
    import numpy as np
    from puppy_motion import stage
    from puppy_motion.joints import shortest_delta
    from puppy_sim.clips.playground_common import stage_store

    ch, q0, q_stage = chain(), np.array(USERS_POSE), stage_pose()
    target = q0 + shortest_delta(ch, q0, q_stage)
    pts = stage.retime_path(stage.min_jerk_path(q0, target, max(2.0, float(np.abs(target - q0).max()) / 0.2)), ch)
    path = tmp_path / name
    stage_store().save(path, pts, urdf_text(), q_stage, method="min_jerk", person_floor_cap_m=cap)
    return path


def test_floor_cap_candidate_is_checked_with_the_cap_and_says_so(tmp_path):
    clip = ps.make_clip(_pose_candidate(tmp_path, 0.02))
    assert clip.candidate.floor_cap_m == 0.02 and clip.candidate.violations == []
    assert "REVIEW FOR APPROVAL" in clip.title
    assert any("ATTENDED FLOOR CAP 20 mm" in line for line in clip.subtitle_for([clip.build()]))


def test_the_same_path_without_the_cap_is_not_approvable_and_does_not_say_cap(tmp_path):
    clip = ps.make_clip(_pose_candidate(tmp_path, 0.0))
    assert clip.candidate.floor_cap_m == 0.0 and clip.candidate.violations
    assert "NOT APPROVABLE" in clip.title
    assert not any("FLOOR CAP" in line for line in clip.subtitle_for([clip.build()]))


def test_a_candidate_without_the_cap_key_reads_as_off(tmp_path):
    import json

    f = _pose_candidate(tmp_path, 0.0)
    doc = json.loads(f.read_text())
    del doc["person_floor_cap_m"]
    f.write_text(json.dumps(doc))
    assert ps.make_clip(f).candidate.floor_cap_m == 0.0


def test_a_candidate_with_a_bad_cap_is_unreadable(tmp_path):
    import json

    f = _pose_candidate(tmp_path, 0.0)
    for bad in (0.05, True, "0.02", None):
        doc = json.loads(f.read_text())
        doc["person_floor_cap_m"] = bad
        f.write_text(json.dumps(doc))
        clip = ps.make_clip(f)
        assert clip.candidate.error and "NOT APPROVABLE" in clip.title, bad


def test_floor_cap_candidate_title_says_so(tmp_path):
    assert "ATTENDED FLOOR CAP 20 mm" in ps.make_clip(_pose_candidate(tmp_path, 0.02)).title
    assert "FLOOR CAP" not in ps.make_clip(_pose_candidate(tmp_path, 0.0, "off.json")).title


def test_a_huge_or_non_finite_cap_fails_closed_with_a_message(tmp_path):
    f = _pose_candidate(tmp_path, 0.0)
    text = f.read_text()
    for bad in ("1" + "0" * 400, "NaN", "Infinity", "-Infinity"):
        f.write_text(text.replace('"person_floor_cap_m": 0.0', f'"person_floor_cap_m": {bad}'))
        clip = ps.make_clip(f)  # must not raise
        assert clip.candidate.error and "person_floor_cap_m" in clip.candidate.error, bad
        assert "NOT APPROVABLE" in clip.title
