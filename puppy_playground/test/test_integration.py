"""The UI node's real ROS wiring (ui_node.main(): ActionClient, Control client, discovery, QoS) against
the real engine, a FakeDriver and a second fake engine, with the HTTP server driven by urllib. Nothing
here talks to a real arm."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import rclpy
from fake_driver import FakeDriver
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from puppy_playground import ui_node
from puppy_playground.engine import PuppyPlayground

HOME = np.array([0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725])
OTHER = "other_engine"


def wait_until(condition, timeout_s=10.0, what="condition"):
    deadline = time.monotonic() + timeout_s
    while True:
        value = condition()
        if value:
            return value
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


class OtherEngine(Node):
    """Another capability engine's status topic, as the UI discovers it."""

    def __init__(self):
        super().__init__(OTHER)
        self.state = "idle"
        self._pub = self.create_publisher(String, f"/{OTHER}/status", 10)
        self.create_timer(0.1, self.publish)

    def publish(self):
        self._pub.publish(String(data=json.dumps({"state": self.state})))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stack(tmp_path):
    port = free_port()
    errors = []

    def run_ui():
        try:
            ui_node.main(["--ros-args", "-p", f"playground.ui_port:={port}"])
        except BaseException as e:  # noqa: BLE001 -- shutdown order below can raise on the second shutdown
            errors.append(e)

    ui = threading.Thread(target=run_ui, daemon=True)
    ui.start()
    wait_until(rclpy.ok, what="rclpy.init in ui_node.main")
    t_start = time.monotonic()
    fake = FakeDriver()
    fake.set_joints(HOME)
    other = OtherEngine()
    overrides = [
        Parameter("use_orchestrator", value=False),
        Parameter("playground.stage_path", value=str(tmp_path / "stage_path.json")),
        Parameter("playground.candidate_path", value=str(tmp_path / "stage_path.candidate.json")),
        Parameter("playground.log", value=str(tmp_path / "playground.jsonl")),
    ]
    engine = PuppyPlayground(parameter_overrides=overrides)
    executor = MultiThreadedExecutor()
    for node in (fake, other, engine):
        executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    class Stack:
        pass

    s = Stack()
    s.fake, s.other, s.t_start = fake, other, t_start
    s.base = f"http://127.0.0.1:{port}"

    def call(method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(s.base + path, data=data, method=method)
        if method == "POST":
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    s.call = call
    wait_until(lambda: _up(s), 20.0, "the UI's HTTP server")
    yield s
    s.call("POST", "/stop", {})
    engine.stop()
    executor.shutdown()
    spinner.join(timeout=10)
    for node in (engine, other, fake):
        node.destroy_node()
    rclpy.shutdown()
    ui.join(timeout=10)


def _up(s):
    try:
        return s.call("GET", "/status")[0] == 200
    except OSError:
        return False


def test_status_express_refusal_stop_and_auto(stack):
    s = stack
    code, st = s.call("GET", "/status")
    assert code == 200 and {"state", "auto", "busy", "estopped", "last_refusal"} <= set(st)

    # the engine is discovered and the control status has arrived
    wait_until(lambda: s.call("GET", "/status")[1]["state"] == "idle", 20.0, "the engine's status")
    wait_until(lambda: not s.call("GET", "/status")[1]["estopped"], 10.0, "/control_status")

    # an expression away from the stage pose is refused, and the refusal shows in /status
    code, _ = s.call("POST", "/express", {"name": "head_tilt", "size": 0.25})
    assert code == 200
    refusal = wait_until(lambda: s.call("GET", "/status")[1]["last_refusal"], 60.0, "the refusal")
    assert refusal["action"] == "express" and refusal["error_code"] == "not_at_stage", refusal
    wait_until(lambda: not s.call("GET", "/status")[1]["busy"], 10.0, "the refused goal to end")
    assert not s.fake.goals["execute_joint_trajectory"]

    assert s.call("POST", "/stop", {})[0] == 200

    # auto: waits for discovery + grace, 3 s of quiet and a still arm
    code, out = s.call("POST", "/auto", {"on": True})
    assert code == 200
    started = wait_until(
        lambda: (lambda st: st["auto"] == "playing" or st["busy"])(s.call("GET", "/status")[1]),
        40.0,
        "auto to start",
    )
    assert started
    assert time.monotonic() - s.t_start >= 3.0, "auto started before the 3 s quiet rule"

    # another engine starts a task: auto yields within ~0.5 s
    t0 = time.monotonic()
    s.other.state = "executing"
    s.other.publish()
    wait_until(lambda: s.call("GET", "/status")[1]["auto"] == "waiting_for_task", 0.5, "auto to yield")
    assert time.monotonic() - t0 < 1.0
    assert s.call("POST", "/auto", {"on": False})[0] == 200
