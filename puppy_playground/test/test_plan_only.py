"""A plan-only instance (the plan_stage run on the live ROS domain) refuses everything else."""

import threading
import time

import pytest
import rclpy
from fake_driver import FakeDriver
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from puppy_playground.engine import PuppyPlayground

WAIT_S = 10.0
HOME = [0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725]


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def env(tmp_path):
    rclpy.init()
    fake = FakeDriver()
    client_node = Node("plan_only_client")
    engine = PuppyPlayground(parameter_overrides=[
        Parameter("use_orchestrator", value=False),
        Parameter("plan_only", value=True),
        Parameter("playground.stage_path", value=str(tmp_path / "stage_path.json")),
        Parameter("playground.candidate_path", value=str(tmp_path / "candidate.json")),
        Parameter("playground.log", value=str(tmp_path / "playground.jsonl")),
    ])
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
        wait_until(result.done, 240.0)
        return result.result().result.result

    yield fake, run, tmp_path
    client.destroy()
    engine.stop()
    executor.shutdown()
    spinner.join()
    engine.destroy_node()
    client_node.destroy_node()
    fake.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize("action", ["stage", "rest", "recover", "express", "combo", "auto"])
def test_plan_only_refuses_every_action_but_plan_stage(env, action):
    fake, run, _ = env
    res = run(action)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "plan_only")
    assert res.recoverable
    assert sum(len(g) for g in fake.goals.values()) == 0


def test_plan_only_still_runs_plan_stage(env):
    fake, run, tmp = env
    fake.set_joints(HOME)
    time.sleep(0.3)
    res = run("plan_stage")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    assert not fake.goals["execute_joint_trajectory"]
    assert (tmp / "candidate.json").exists()
