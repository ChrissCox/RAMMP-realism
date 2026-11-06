"""The hook with puppy.enabled false changes nothing: the arm goals a capability sends are exactly
the ones it asked for, and hook.run is never reached. With it true, hook.run is called once per
goal and (until Task 13) its result is the original execute's. Each engine runs beside a FakeDriver."""

import itertools
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from capability_engine.arm import IMPEDANCE, POSITION
from capability_engine.engine import Action, CapabilityEngine
from fake_driver import FakeDriver
from rammp_arm_interfaces.msg import ImpedanceGains
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.serialization import serialize_message

from puppy_mode_ros import hook

WAIT_S = 10.0
ACTION_WAIT_S = 60.0
Q1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
Q2 = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
KNOTS = [[0.3] * 7]
ARM_GOALS = ("go_to_joint_config", "execute_joint_trajectory")


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def make_capability():
    """A fresh capability class per test (install() mutates the class)."""

    class Capability(CapabilityEngine):
        actions = [Action("run", "Run")]

        def execute(self, action_id, params, ctx):
            self.arm.go_to_joint_config(np.array(Q1))
            self.arm.go_to_joint_config(np.array(Q2), control_mode=IMPEDANCE, gains="soft")
            self.arm.move_joints(KNOTS)
            return {"done": action_id}

    return Capability


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def env(ros):
    """make(cls, **params) -> a spun engine; run(engine) -> the "run" action's ExecuteAction result."""
    fake = FakeDriver()
    client_node = Node("hook_off_client")
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(client_node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    engines = []
    ids = itertools.count()

    def make(cls, **params):
        overrides = [Parameter("use_orchestrator", value=False)]
        overrides += [Parameter(k, value=v) for k, v in params.items()]
        engine = cls(f"hook_off_engine_{id(fake)}_{next(ids)}", parameter_overrides=overrides)
        engines.append(engine)
        executor.add_node(engine)
        return engine

    def run(engine):
        client = ActionClient(client_node, ExecuteAction, f"/{engine.get_name()}/execute_action")
        try:
            assert client.wait_for_server(timeout_sec=WAIT_S)
            send = client.send_goal_async(ExecuteAction.Goal(action_id="run"))
            wait_until(send.done)
            assert send.result().accepted
            result = send.result().get_result_async()
            wait_until(result.done, ACTION_WAIT_S)
            return result.result().result
        finally:
            client.destroy()

    yield SimpleNamespace(fake=fake, make=make, run=run)
    for engine in engines:
        engine.stop()
    executor.shutdown()
    spinner.join()
    for engine in engines:
        engine.destroy_node()
    client_node.destroy_node()
    fake.destroy_node()


def sent(fake):
    """Every goal the fake driver received, serialized, by action name."""
    return {n: [serialize_message(g) for g in gs] for n, gs in fake.goals.items() if gs}


def test_disabled_hook_leaves_goals_untouched(env, monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "run", lambda *a, **k: calls.append(a))
    engine = env.make(hook.install(make_capability()))
    assert engine.get_parameter("puppy.enabled").value is False
    result = env.run(engine)
    assert result.result.status == Result.STATUS_SUCCESS
    assert json.loads(result.result.payload_json) == {"done": "run"}
    assert calls == []

    assert set(sent(env.fake)) == set(ARM_GOALS)
    go_to = env.fake.goals["go_to_joint_config"]
    assert [list(g.target_joints) for g in go_to] == [Q1, Q2]
    assert [g.control_mode for g in go_to] == [POSITION, IMPEDANCE]
    assert go_to[0].gains.profile == ImpedanceGains().profile
    assert go_to[1].gains.profile == ImpedanceGains.PROFILE_SOFT

    (traj,) = env.fake.goals["execute_joint_trajectory"]
    assert traj.control_mode == POSITION
    points = [list(p.positions) for p in traj.trajectory.points]
    assert len(points) >= 2
    assert points[-1] == pytest.approx(KNOTS[-1])


def test_disabled_goals_equal_an_uninstalled_engine(env):
    """The same capability with and without install() sends byte-identical goals."""
    plain = env.make(make_capability())
    assert env.run(plain).result.status == Result.STATUS_SUCCESS
    baseline = sent(env.fake)
    assert baseline
    plain.stop()
    env.fake.goals.clear()

    hooked = env.make(hook.install(make_capability()))
    assert env.run(hooked).result.status == Result.STATUS_SUCCESS
    assert sent(env.fake) == baseline


def test_enabled_hook_run_called_once_per_goal(env, monkeypatch):
    calls = []
    real_run = hook.run

    def spy(engine, action_id, params, ctx, original):
        calls.append((engine, action_id))
        return real_run(engine, action_id, params, ctx, original)

    monkeypatch.setattr(hook, "run", spy)
    engine = env.make(hook.install(make_capability()), **{"puppy.enabled": True})
    result = env.run(engine)
    assert result.result.status == Result.STATUS_SUCCESS
    assert json.loads(result.result.payload_json) == {"done": "run"}  # result unchanged
    assert calls == [(engine, "run")]
    assert len(env.fake.goals["go_to_joint_config"]) == 2
    assert len(env.fake.goals["execute_joint_trajectory"]) == 1
    env.run(engine)
    assert len(calls) == 2


def test_params_declared_with_defaults(env):
    engine = env.make(hook.install(make_capability()))
    assert engine.get_parameter("puppy.enabled").value is False
    assert engine.get_parameter("puppy.preset").value == "subtle"
    assert engine.get_parameter("puppy.person_zone").value == ""
    assert engine.get_parameter("puppy.lock_on").value == "view"


def test_params_declared_lazily_for_a_node_built_before_install(env):
    cls = make_capability()
    engine = env.make(cls)
    assert not engine.has_parameter("puppy.enabled")
    hook.install(cls)
    assert env.run(engine).result.status == Result.STATUS_SUCCESS
    assert engine.get_parameter("puppy.preset").value == "subtle"


def test_install_twice_wraps_once(env, monkeypatch):
    calls = []
    monkeypatch.setattr(hook, "run", lambda e, a, p, c, original: calls.append(a) or original(e, a, p, c))
    cls = hook.install(hook.install(make_capability()))
    engine = env.make(cls, **{"puppy.enabled": True})
    env.run(engine)
    assert calls == ["run"]
