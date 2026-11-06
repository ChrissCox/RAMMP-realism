"""The engine starts beside a FakeDriver and every action fails with "not_yet", moving nothing."""

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


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_express_fails_not_yet_and_the_driver_gets_no_goals(ros):
    fake = FakeDriver()
    client_node = Node("playground_test_client")
    engine = PuppyPlayground(parameter_overrides=[Parameter("use_orchestrator", value=False)])
    executor = MultiThreadedExecutor()
    for node in (fake, client_node, engine):
        executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    client = ActionClient(client_node, ExecuteAction, "/puppy_playground/execute_action")
    try:
        assert client.wait_for_server(timeout_sec=WAIT_S)
        send = client.send_goal_async(ExecuteAction.Goal(action_id="express"))
        wait_until(send.done)
        assert send.result().accepted
        result = send.result().get_result_async()
        wait_until(result.done)
        res = result.result().result.result
        assert res.status == Result.STATUS_FAILED
        assert res.error_code == "not_yet"
        assert sum(len(g) for g in fake.goals.values()) == 0
    finally:
        client.destroy()
        engine.stop()
        executor.shutdown()
        spinner.join()
        engine.destroy_node()
        client_node.destroy_node()
        fake.destroy_node()
