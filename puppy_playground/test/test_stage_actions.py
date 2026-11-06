"""plan_stage, stage, rest and recover against a FakeDriver. Nothing here talks to a real arm."""

import json
import threading
import time

import numpy as np
import pytest
import rclpy
from fake_driver import FLATBOT_URDF, FakeDriver
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from puppy_motion.escape import escape_path
from puppy_motion.flatbot import chair_scene
from puppy_motion.urdf import Chain
from puppy_playground import stage_store
from puppy_playground.engine import PuppyPlayground

WAIT_S = 10.0
ACTION_WAIT_S = 240.0  # a path check costs 5-20 s on a CI runner
HOME = np.array([0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725])
IMPEDANCE = 1
SETTLE_S = 0.3  # the fake publishes state at 50 Hz


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture(scope="module")
def urdf_text():
    return FLATBOT_URDF.read_text()


@pytest.fixture(scope="module")
def stage_q(urdf_text):
    from importlib import resources

    text = resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text(encoding="utf-8")
    return np.array(json.loads(text)["q"])


@pytest.fixture(scope="module")
def path_points(urdf_text, stage_q):
    return escape_path(Chain.from_urdf(urdf_text), chair_scene(), HOME, stage_q)


@pytest.fixture
def env(tmp_path):
    rclpy.init()
    fake = FakeDriver()
    client_node = Node("playground_actions_client")
    stage_file, cand_file = tmp_path / "stage_path.json", tmp_path / "stage_path.candidate.json"
    overrides = [
        Parameter("use_orchestrator", value=False),
        Parameter("playground.stage_path", value=str(stage_file)),
        Parameter("playground.candidate_path", value=str(cand_file)),
        Parameter("playground.log", value=str(tmp_path / "playground.jsonl")),
    ]
    engine = PuppyPlayground(parameter_overrides=overrides)
    executor = MultiThreadedExecutor()
    for node in (fake, client_node, engine):
        executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    client = ActionClient(client_node, ExecuteAction, "/puppy_playground/execute_action")
    assert client.wait_for_server(timeout_sec=WAIT_S)

    def run(action):
        send = client.send_goal_async(ExecuteAction.Goal(action_id=action))
        wait_until(send.done)
        assert send.result().accepted
        result = send.result().get_result_async()
        wait_until(result.done, ACTION_WAIT_S)
        return result.result().result.result

    def at(q):
        fake.set_joints(q)
        time.sleep(SETTLE_S)

    class Env:
        pass

    e = Env()
    e.fake, e.run, e.at = fake, run, at
    e.stage_file, e.cand_file, e.dir = stage_file, cand_file, tmp_path
    yield e
    client.destroy()
    engine.stop()
    executor.shutdown()
    spinner.join()
    engine.destroy_node()
    client_node.destroy_node()
    fake.destroy_node()
    rclpy.shutdown()


def approve(env, urdf_text, stage_q, points):
    stage_store.save(env.stage_file, points, urdf_text, stage_q)


def sent(env):
    goals = env.fake.goals["execute_joint_trajectory"]
    assert len(goals) == 1
    g = goals[0]
    return g, [(np.array(p.positions), p.time_from_start.sec + p.time_from_start.nanosec * 1e-9) for p in g.trajectory.points]


def test_plan_stage_sends_no_motion_goal_and_writes_a_candidate(env, urdf_text, stage_q):
    env.at(HOME)
    straight = [HOME + (stage_q - HOME) * s for s in np.linspace(0.0, 1.0, 20)]
    env.fake.plan_result(straight)
    res = env.run("plan_stage")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    assert not env.fake.goals["execute_joint_trajectory"] and not env.fake.goals["go_to_joint_config"]
    assert len(env.fake.goals["plan_to_joints"]) == 1
    out = json.loads(res.payload_json)
    assert out["violations"] == [] and out["candidate"] == str(env.cand_file)
    sp = stage_store.load(env.cand_file, urdf_text, stage_q)  # endpoints check included
    assert sp.margin_m == stage_store.MARGIN_M
    assert not env.stage_file.exists()  # never approves itself


def test_stage_refuses_without_an_approved_file(env):
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "no_stage_path")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_stage_refuses_a_file_planned_for_another_urdf(env, urdf_text, stage_q, path_points):
    stage_store.save(env.stage_file, path_points, urdf_text + "<!-- other -->", stage_q)
    env.at(HOME)
    assert env.run("stage").error_code == "stale_stage_path"
    assert not env.fake.goals["execute_joint_trajectory"]


def test_stage_refuses_a_corrupt_file(env):
    env.stage_file.write_text("{not json")
    env.at(HOME)
    assert env.run("stage").error_code == "bad_stage_path"


@pytest.mark.parametrize("edit", ["start_time", "q_rest", "stage_q"])
def test_stage_refuses_a_path_with_bad_endpoints(env, urdf_text, stage_q, path_points, edit):
    approve(env, urdf_text, stage_q, path_points)
    doc = json.loads(env.stage_file.read_text())
    if edit == "start_time":
        for p in doc["points"]:
            p[0] += 0.02
    elif edit == "q_rest":
        doc["q_rest"][0] += 1e-4
    else:
        doc["stage_q"][0] += 1e-4
    env.stage_file.write_text(json.dumps(doc))
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_endpoints")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_stage_refuses_a_path_whose_margin_does_not_hold(env, urdf_text, stage_q, path_points):
    stage_store.save(env.stage_file, path_points, urdf_text, stage_q, margin_m=0.5)
    env.at(HOME)
    assert env.run("stage").error_code == "unsafe_stage_path"
    assert not env.fake.goals["execute_joint_trajectory"]


def test_stage_refuses_when_not_at_the_paths_start(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME + 0.3)
    assert env.run("stage").error_code == "not_at_start"
    assert not env.fake.goals["execute_joint_trajectory"]


def test_stage_and_rest_send_the_path_forward_and_reversed(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME)
    assert env.run("stage").status == Result.STATUS_SUCCESS
    g, pts = sent(env)
    assert g.control_mode == IMPEDANCE and list(g.trajectory.joint_names) == list(Chain.from_urdf(urdf_text).joint_names)
    assert len(pts) == len(path_points)
    for (q, _), (_, want, _) in zip(pts, path_points):
        assert np.allclose(q, want, atol=1e-9)
    assert np.allclose(g.trajectory.points[0].velocities, 0) and np.allclose(g.trajectory.points[-1].velocities, 0)
    env.fake.goals["execute_joint_trajectory"].clear()

    env.at(stage_q)
    assert env.run("rest").status == Result.STATUS_SUCCESS
    g, pts = sent(env)
    assert g.control_mode == IMPEDANCE
    for (q, _), (_, want, _) in zip(pts, path_points[::-1]):
        assert np.allclose(q, want, atol=1e-9)
    assert np.allclose(g.trajectory.points[0].velocities, 0) and np.allclose(g.trajectory.points[-1].velocities, 0)


def test_rest_refuses_when_not_at_the_paths_end(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME)
    assert env.run("rest").error_code == "not_at_end"


def test_recover_near_stage_sends_a_checked_min_jerk_move(env, stage_q):
    env.at(stage_q + 0.05)
    assert env.run("recover").status == Result.STATUS_SUCCESS
    g, pts = sent(env)
    assert g.control_mode == IMPEDANCE
    assert np.allclose(pts[-1][0], stage_q, atol=1e-9) and np.allclose(pts[0][0], stage_q + 0.05, atol=1e-9)


def test_recover_on_the_path_follows_it_back_to_rest(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    i = len(path_points) // 3
    env.at(path_points[i][1])
    assert env.run("recover").status == Result.STATUS_SUCCESS
    g, pts = sent(env)
    assert np.allclose(pts[-1][0], HOME, atol=1e-9) and np.allclose(pts[0][0], path_points[i][1], atol=1e-9)


def test_recover_elsewhere_refuses(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME + np.array([0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0]))
    res = env.run("recover")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "recover_by_hand")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_each_request_leaves_one_log_line(env):
    env.at(HOME)
    env.run("stage")
    lines = (env.dir / "playground.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["action"] == "stage"
