"""The opt-in attended floor cap (playground.stage_floor_cap_m): off by default, 0.0 or 0.02 only, recorded
in the stage file, and a file only replays under the setting it was approved with."""

import json

import numpy as np
import pytest
from rammp_orchestrator_interfaces.msg import Result
from test_stage_actions import HOME, _env, approve, path_points, stage_q, urdf_text  # noqa: F401 -- fixtures

from puppy_motion.escape import grown_scene
from puppy_motion.flatbot import chair_scene
from puppy_motion.stage import path_violations
from puppy_motion.urdf import Chain
from puppy_playground import stage_store
from puppy_playground.engine import ATTENDED_FLOOR_CAP_M

# the pose the user parked at on 2026-11-06 (upper arm 44 mm from the occupant box)
USERS_POSE = np.array([0.0531, 0.1366, -3.1161, -2.3040, -0.6222, 1.3713, 1.7233])


@pytest.fixture
def env(tmp_path):  # the engine with the parameter not set at all
    yield from _env(tmp_path)


@pytest.fixture
def env_cap(tmp_path):
    yield from _env(tmp_path, ATTENDED_FLOOR_CAP_M)


@pytest.fixture
def env_bad_cap(request, tmp_path):
    yield from _env(tmp_path, request.param)


def test_the_constant_is_20_mm():
    assert ATTENDED_FLOOR_CAP_M == 0.020 and stage_store.ATTENDED_FLOOR_CAP_M == 0.020


def test_off_by_default_the_users_pose_has_no_plan_and_nothing_is_written(env):
    assert env.engine.stage_floor_cap_m == 0.0
    env.at(USERS_POSE)
    res = env.run("plan_stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "no_candidate"), (res.error_code, res.payload_json)
    assert "path_person_floor" in json.loads(res.payload_json)["message"]
    assert not env.cand_file.exists()


def test_with_the_cap_the_users_pose_plans_min_jerk_and_the_file_loads_only_with_the_cap(env_cap, urdf_text, stage_q):
    env = env_cap
    env.at(USERS_POSE)
    res = env.run("plan_stage")
    assert res.status == Result.STATUS_SUCCESS, (res.error_code, res.payload_json)
    assert json.loads(res.payload_json)["method"] == "min_jerk"
    doc = json.loads(env.cand_file.read_text())
    assert doc["person_floor_cap_m"] == 0.02
    chain = Chain.from_urdf(urdf_text)
    sp = stage_store.load(env.cand_file, urdf_text, stage_q, chain, floor_cap_m=0.02)
    assert sp.person_floor_cap_m == 0.02
    for off in (0.0, None):  # the default is off
        kwargs = {} if off is None else {"floor_cap_m": off}
        with pytest.raises(stage_store.StageStoreError) as e:
            stage_store.load(env.cand_file, urdf_text, stage_q, chain, **kwargs)
        assert e.value.code == "bad_stage_path" and "person_floor_cap_m" in str(e.value)
    # the unrelaxed rule still rejects this path
    assert path_violations(sp.points, chain, grown_scene(chair_scene(), stage_store.MARGIN_M))
    # approve it: Stage replays under the cap (reload check and blended check), and Rest returns
    env.cand_file.replace(env.stage_file)
    env.at(USERS_POSE)
    res = env.run("stage")
    assert res.status == Result.STATUS_SUCCESS, (res.error_code, res.payload_json)
    assert len(env.fake.goals["execute_joint_trajectory"]) == 1
    # Rest returns along the same file (it checks caps and ends only)
    env.at(stage_q)
    res = env.run("rest")
    assert res.status == Result.STATUS_SUCCESS, (res.error_code, res.payload_json)
    assert len(env.fake.goals["execute_joint_trajectory"]) == 2


def test_the_cap_and_plan_only_are_read_only_at_run_time(env_cap):
    from rclpy.parameter import Parameter

    for name, value, attr, kept in (
        ("playground.stage_floor_cap_m", 0.0, "stage_floor_cap_m", 0.02),
        ("plan_only", True, "plan_only", False),
    ):
        r = env_cap.engine.set_parameters([Parameter(name, value=value)])[0]
        assert not r.successful, name
        assert getattr(env_cap.engine, attr) == kept
        assert env_cap.engine.get_parameter(name).value == kept  # ROS never reports a value the engine isn't using


def test_launch_turns_the_argument_into_a_float_for_both_nodes():
    import importlib.util
    from pathlib import Path

    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.parameter_descriptions import ParameterValue

    path = Path(__file__).resolve().parents[1] / "launch" / "playground.launch.py"
    spec = importlib.util.spec_from_file_location("playground_launch2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = []

    def walk(x, depth=0):
        if depth > 6:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                name = k if isinstance(k, str) else "".join(getattr(t, "text", "") for t in k)
                if name == "playground.stage_floor_cap_m":
                    found.append(v)
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)
        elif hasattr(x, "__dict__") and not isinstance(x, (str, bytes)):
            walk(vars(x), depth + 1)

    for e in module.generate_launch_description().entities:
        if type(e).__name__ == "Node":
            walk(vars(e))
    assert len(found) == 2, found  # the engine and the UI
    ctx = LaunchContext()
    ctx.launch_configurations["stage_floor_cap_m"] = "0.02"
    for v in found:
        assert isinstance(v, ParameterValue)
        got = v.evaluate(ctx)
        assert isinstance(got, float) and got == 0.02


def test_a_file_approved_under_the_cap_is_refused_by_an_engine_without_it(env, urdf_text, stage_q, path_points):
    stage_store.save(env.stage_file, path_points, urdf_text, stage_q, person_floor_cap_m=0.02)
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_stage_path"), (res.error_code, res.payload_json)
    assert not env.fake.goals["execute_joint_trajectory"]


def test_a_file_approved_without_the_cap_is_refused_by_an_engine_with_it(env_cap, urdf_text, stage_q, path_points):
    approve(env_cap, urdf_text, stage_q, path_points)
    env_cap.at(HOME)
    res = env_cap.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_stage_path"), (res.error_code, res.payload_json)
    assert not env_cap.fake.goals["execute_joint_trajectory"]


def test_cap_mismatch_is_refused_at_load_both_ways(tmp_path, urdf_text, stage_q, path_points):
    chain = Chain.from_urdf(urdf_text)
    capped, plain = tmp_path / "capped.json", tmp_path / "plain.json"
    stage_store.save(capped, path_points, urdf_text, stage_q, person_floor_cap_m=0.02)
    stage_store.save(plain, path_points, urdf_text, stage_q)
    assert json.loads(plain.read_text())["person_floor_cap_m"] == 0.0
    stage_store.load(capped, urdf_text, stage_q, chain, floor_cap_m=0.02)
    stage_store.load(plain, urdf_text, stage_q, chain)
    for path, engine_cap in ((capped, 0.0), (plain, 0.02)):
        with pytest.raises(stage_store.StageStoreError) as e:
            stage_store.load(path, urdf_text, stage_q, chain, floor_cap_m=engine_cap)
        assert e.value.code == "bad_stage_path"


def test_an_old_file_without_the_key_reads_as_off(tmp_path, urdf_text, stage_q, path_points):
    chain = Chain.from_urdf(urdf_text)
    f = tmp_path / "old.json"
    stage_store.save(f, path_points, urdf_text, stage_q)
    doc = json.loads(f.read_text())
    del doc["person_floor_cap_m"]
    f.write_text(json.dumps(doc))
    assert stage_store.load(f, urdf_text, stage_q, chain).person_floor_cap_m == 0.0
    with pytest.raises(stage_store.StageStoreError):
        stage_store.load(f, urdf_text, stage_q, chain, floor_cap_m=0.02)


@pytest.mark.parametrize("bad", [True, "0.02", None, float("nan"), float("inf"), -0.02, 0.05, [0.02], 10**400])
def test_the_cap_in_a_file_is_type_and_value_checked(tmp_path, urdf_text, stage_q, path_points, bad):
    chain = Chain.from_urdf(urdf_text)
    f = tmp_path / "x.json"
    stage_store.save(f, path_points, urdf_text, stage_q)
    doc = json.loads(f.read_text())
    doc["person_floor_cap_m"] = bad
    f.write_text(json.dumps(doc))  # json.dumps writes NaN / Infinity as bare tokens, which json.loads reads back
    for engine_cap in (0.0, 0.02):
        with pytest.raises(stage_store.StageStoreError) as e:
            stage_store.load(f, urdf_text, stage_q, chain, floor_cap_m=engine_cap)
        assert e.value.code == "bad_stage_path"


def test_an_integer_zero_cap_in_a_file_is_off(tmp_path, urdf_text, stage_q, path_points):
    chain = Chain.from_urdf(urdf_text)
    f = tmp_path / "x.json"
    stage_store.save(f, path_points, urdf_text, stage_q)
    doc = json.loads(f.read_text())
    doc["person_floor_cap_m"] = 0
    f.write_text(json.dumps(doc))
    assert stage_store.load(f, urdf_text, stage_q, chain).person_floor_cap_m == 0.0


@pytest.mark.parametrize("env_bad_cap", [0.01, 0.03, -0.02, float("nan")], indirect=True)
@pytest.mark.parametrize("action", ["plan_stage", "stage", "rest", "recover", "express", "combo", "auto"])
def test_any_other_cap_refuses_every_action_with_bad_config_before_touching_the_arm(env_bad_cap, action):
    env = env_bad_cap
    env.at(HOME)
    res = env.run(action)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_config"), (res.error_code, res.payload_json)
    assert "stage_floor_cap_m" in json.loads(res.payload_json)["message"]
    for goals in env.fake.goals.values():
        assert not goals
    assert not env.cand_file.exists()
    assert "stage_floor_cap_m" in (env.dir / "playground.jsonl").read_text()  # logged, with the reason
