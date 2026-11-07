"""The start gate is recorded in the stage file and re-checked when it is loaded: a file that fails the gate, or
records it as skipped, is refused unless the engine's read-only accept_skipped_start_gate is true."""

import json

import numpy as np
import pytest
from rammp_orchestrator_interfaces.msg import Result
from test_stage_actions import HOME, _env, path_points, stage_q, urdf_text  # noqa: F401 -- fixtures

from puppy_motion import stage
from puppy_motion.joints import shortest_delta
from puppy_motion.urdf import Chain
from puppy_playground import stage_store

# a pose whose upper arm and forearm are inside the occupant box: it fails the start gate
INSIDE_OCCUPANT = np.deg2rad([22.9, 114.2, -43.4, -71.2, -15.7, 36.2, -143.6])


@pytest.fixture
def env(tmp_path):
    yield from _env(tmp_path)


@pytest.fixture
def env_accept(tmp_path):
    yield from _env(tmp_path, accept_skipped=True)


@pytest.fixture(scope="module")
def chain(urdf_text):
    return Chain.from_urdf(urdf_text)


@pytest.fixture(scope="module")
def deep_points(chain, stage_q):
    """A path that starts where the start gate fails."""
    target = INSIDE_OCCUPANT + shortest_delta(chain, INSIDE_OCCUPANT, stage_q)
    return stage.retime_path(stage.min_jerk_path(INSIDE_OCCUPANT, target, 6.0), chain)


def _hand_written(path, points, urdf_text, stage_q, drop=()):
    """Save, then drop keys: the way an older file (or tonight's) lacks start_gate."""
    stage_store.save(path, points, urdf_text, stage_q)
    with open(path, encoding="utf-8") as f:
        doc = json.loads(f.read())
    for k in drop:
        doc.pop(k)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc))


def test_save_records_passed_by_default_and_rejects_other_values(tmp_path, urdf_text, stage_q, path_points):
    f = tmp_path / "s.json"
    stage_store.save(f, path_points, urdf_text, stage_q)
    assert json.loads(f.read_text())["start_gate"] == "passed"
    with pytest.raises(ValueError):
        stage_store.save(f, path_points, urdf_text, stage_q, start_gate="ignored")


def test_an_ordinary_planned_file_still_loads(tmp_path, urdf_text, stage_q, chain, path_points):
    f = tmp_path / "s.json"
    stage_store.save(f, path_points, urdf_text, stage_q)
    assert stage_store.load(str(f), urdf_text, stage_q, chain).start_gate == "passed"


def test_an_older_file_without_the_key_is_rechecked_and_loads_when_the_start_passes(tmp_path, urdf_text, stage_q, chain, path_points):
    f = tmp_path / "s.json"
    _hand_written(f, path_points, urdf_text, stage_q, drop=("start_gate",))
    assert stage_store.load(str(f), urdf_text, stage_q, chain).start_gate == ""


def test_a_skipped_gate_file_is_refused_by_default_and_accepted_with_the_param(tmp_path, urdf_text, stage_q, chain, path_points):
    f = tmp_path / "s.json"
    stage_store.save(f, path_points, urdf_text, stage_q, start_gate="skipped")
    with pytest.raises(stage_store.StageStoreError) as e:
        stage_store.load(str(f), urdf_text, stage_q, chain)
    assert e.value.code == "bad_stage_path" and "start gate" in str(e.value)
    assert stage_store.load(str(f), urdf_text, stage_q, chain, accept_skipped_start_gate=True).start_gate == "skipped"


def test_a_first_sample_that_fails_the_gate_without_the_key_is_refused(tmp_path, urdf_text, stage_q, chain, deep_points):
    f = tmp_path / "s.json"
    _hand_written(f, deep_points, urdf_text, stage_q, drop=("start_gate",))  # like tonight's file
    with pytest.raises(stage_store.StageStoreError) as e:
        stage_store.load(str(f), urdf_text, stage_q, chain)
    assert e.value.code == "bad_stage_path" and "start gate" in str(e.value) and "occupant" in str(e.value)


def test_a_file_that_says_passed_but_fails_the_gate_is_refused(tmp_path, urdf_text, stage_q, chain, deep_points):
    f = tmp_path / "s.json"
    stage_store.save(f, deep_points, urdf_text, stage_q, start_gate="passed")
    with pytest.raises(stage_store.StageStoreError) as e:
        stage_store.load(str(f), urdf_text, stage_q, chain)
    assert e.value.code == "bad_stage_path"


def test_an_unknown_start_gate_value_is_refused(tmp_path, urdf_text, stage_q, chain, path_points):
    f = tmp_path / "s.json"
    stage_store.save(f, path_points, urdf_text, stage_q)
    doc = json.loads(f.read_text())
    doc["start_gate"] = "ignored"
    f.write_text(json.dumps(doc))
    with pytest.raises(stage_store.StageStoreError) as e:
        stage_store.load(str(f), urdf_text, stage_q, chain, accept_skipped_start_gate=True)
    assert e.value.code == "bad_stage_path"


def test_the_engine_refuses_a_skipped_file_and_sends_nothing(env, urdf_text, stage_q, path_points):
    stage_store.save(env.stage_file, path_points, urdf_text, stage_q, start_gate="skipped")
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_stage_path"), (res.error_code, res.payload_json)
    assert "start gate" in res.payload_json
    assert not env.fake.goals["execute_joint_trajectory"]


def test_the_engine_refuses_a_deep_start_file_without_the_key(env, urdf_text, stage_q, deep_points):
    _hand_written(env.stage_file, deep_points, urdf_text, stage_q, drop=("start_gate",))
    env.at(INSIDE_OCCUPANT)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_stage_path"), (res.error_code, res.payload_json)
    assert not env.fake.goals["execute_joint_trajectory"]


def test_the_engine_with_the_param_gets_past_the_load_of_a_skipped_file(env_accept, urdf_text, stage_q, path_points):
    assert env_accept.engine.accept_skipped_start_gate is True
    stage_store.save(env_accept.stage_file, path_points, urdf_text, stage_q, start_gate="skipped")
    env_accept.at(HOME)
    res = env_accept.run("stage")
    assert res.error_code != "bad_stage_path", res.payload_json


def test_the_param_is_off_by_default_and_read_only(env):
    from rclpy.parameter import Parameter

    assert env.engine.accept_skipped_start_gate is False
    assert not env.engine.set_parameters([Parameter("playground.accept_skipped_start_gate", value=True)])[0].successful


def test_the_launch_argument_defaults_false_and_reaches_the_engine_as_a_bool():
    import importlib.util
    from pathlib import Path

    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.parameter_descriptions import ParameterValue

    path = Path(__file__).resolve().parents[1] / "launch" / "playground.launch.py"
    spec = importlib.util.spec_from_file_location("playground_launch3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entities = module.generate_launch_description().entities
    decl = [e for e in entities if isinstance(e, DeclareLaunchArgument) and e.name == "accept_skipped_start_gate"]
    assert len(decl) == 1 and decl[0].default_value[0].text == "false"
    found = []

    def walk(x, depth=0):
        if depth > 6:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                name = k if isinstance(k, str) else "".join(getattr(t, "text", "") for t in k)
                if name == "playground.accept_skipped_start_gate":
                    found.append(v)
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)
        elif hasattr(x, "__dict__") and not isinstance(x, (str, bytes)):
            walk(vars(x), depth + 1)

    for e in entities:
        if type(e).__name__ == "Node":
            walk(vars(e))
    assert len(found) == 1  # the engine only
    ctx = LaunchContext()
    ctx.launch_configurations["accept_skipped_start_gate"] = "true"
    got = found[0].evaluate(ctx)
    assert isinstance(found[0], ParameterValue) and got is True
