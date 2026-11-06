"""The UI node's real ROS wiring (ui_node.main(): ActionClient, Control client, discovery, QoS) against
the real engine, a FakeDriver and a second fake engine, with the HTTP server driven by urllib. Nothing
here talks to a real arm."""

import importlib.util
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import rclpy
from fake_driver import FLATBOT_URDF, FakeDriver
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from puppy_motion.escape import escape_path
from puppy_motion.flatbot import chair_scene
from puppy_motion.urdf import Chain
from puppy_playground import stage_store, ui_node
from puppy_playground.engine import PuppyPlayground

HOME = np.array([0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725])
OTHER = "other_engine"
IMPEDANCE = 1
LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "playground.launch.py"


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


@pytest.fixture(scope="module")
def stage_q():
    from importlib import resources

    text = resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text(encoding="utf-8")
    return np.array(json.loads(text)["q"])


@pytest.fixture
def stack(tmp_path, stage_q):
    urdf = FLATBOT_URDF.read_text()
    points = escape_path(Chain.from_urdf(urdf), chair_scene(), HOME, stage_q)
    stage_store.save(str(tmp_path / "stage_path.json"), points, urdf, stage_q)  # the approved file
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


def same_pose(a, b):
    """Equal modulo whole turns (the continuous joints are sent turned to the reading)."""
    return bool(np.all(np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))) < 1e-3))


def goals(s):
    return s.fake.goals["execute_joint_trajectory"]


def points_of(goal):
    return [np.array(p.positions) for p in goal.trajectory.points]


def test_launch_file_declares_the_arguments():
    spec = importlib.util.spec_from_file_location("playground_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    names = {e.name for e in description.entities if type(e).__name__ == "DeclareLaunchArgument"}
    assert {"use_orchestrator", "mock", "plan_only", "mock_urdf", "mock_joints", "ui_port"} <= names


def test_status_express_refusal_stage_express_stop_and_auto(stack, stage_q):
    s = stack
    code, st = s.call("GET", "/status")
    assert code == 200 and {"state", "auto", "busy", "estopped", "last_refusal"} <= set(st)

    # the engine is discovered and the control status has arrived
    wait_until(lambda: s.call("GET", "/status")[1]["state"] == "idle", 20.0, "the engine's status")
    wait_until(lambda: not s.call("GET", "/status")[1]["estopped"], 10.0, "/control_status")

    # at rest an expression is refused, and the refusal shows in /status
    assert s.call("POST", "/express", {"name": "head_tilt", "size": 0.25})[0] == 200
    refusal = wait_until(lambda: s.call("GET", "/status")[1]["last_refusal"], 90.0, "the refusal")
    assert refusal["action"] == "express" and refusal["error_code"] == "not_at_stage", refusal
    wait_until(lambda: not s.call("GET", "/status")[1]["busy"], 10.0, "the refused goal to end")
    assert not goals(s)

    # Stage: the approved path reaches the driver, rest pose first and stage pose last
    s.fake.finish_at("execute_joint_trajectory", joints=stage_q)
    assert s.call("POST", "/stage", {})[0] == 200
    wait_until(lambda: len(goals(s)) == 1, 90.0, "the stage goal at the driver")
    pts = points_of(goals(s)[0])
    assert same_pose(pts[0], HOME) and same_pose(pts[-1], stage_q)
    assert goals(s)[0].control_mode == IMPEDANCE
    wait_until(lambda: not s.call("GET", "/status")[1]["busy"], 30.0, "stage to end")
    assert s.call("GET", "/status")[1]["last_refusal"] is None

    # at stage an expression reaches the driver too
    assert s.call("POST", "/express", {"name": "head_tilt", "size": 0.25})[0] == 200
    wait_until(lambda: len(goals(s)) == 2, 90.0, "the express goal at the driver")
    assert goals(s)[1].control_mode == IMPEDANCE
    wait_until(lambda: not s.call("GET", "/status")[1]["busy"], 30.0, "express to end")
    assert s.call("GET", "/status")[1]["last_refusal"] is None

    assert s.call("POST", "/stop", {})[0] == 200

    # auto: waits for discovery, grace, 3 s of quiet and a still arm, then a beat reaches the driver
    before = len(goals(s))
    assert s.call("POST", "/auto", {"on": True, "size": 0.25})[0] == 200
    wait_until(lambda: len(goals(s)) > before, 90.0, "an auto beat at the driver")
    assert s.call("GET", "/status")[1]["auto"] in ("playing", "waiting_idle")

    # another engine starts a task: auto yields
    s.other.state = "executing"
    s.other.publish()
    wait_until(lambda: s.call("GET", "/status")[1]["auto"] == "waiting_for_task", 2.0, "auto to yield")
    assert s.call("POST", "/auto", {"on": False})[0] == 200
