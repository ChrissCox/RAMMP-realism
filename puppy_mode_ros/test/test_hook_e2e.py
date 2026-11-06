"""hook.run end to end: a scripted capability under the installed hook, beside a FakeDriver.

The capability stands in for box_opening: it reports locate, look, press, grip, carry, release and
home through ctx.progress and never moves self.arm (the real one runs its motion in a subprocess).
Nothing here talks to a real arm."""

import json
import math
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from capability_engine.engine import Action, CapabilityEngine, EngineError
from fake_driver import FakeDriver
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from puppy_mode_ros import hook
from puppy_motion.geometry import Scene

WAIT_S = 10.0
ACTION_WAIT_S = 60.0
START = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
SETTLE_S = 0.3
EXECUTE = "execute_joint_trajectory"
PHASES = ("locate", "look", "press", "grip", "carry", "release", "home")
# a person and an obstacle far from the arm: every beat the director composes is fit to send
SCENE = json.dumps({"person": {"spheres": [[5, 5, 5, 0.1]]}, "obstacles": {"boxes": [[-5, 5, 5, 0.1, 0.1, 0.1]]}})
SUBTLE_BUDGET_S = 2.5


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def make_capability(fake, seen, fail_in=None, error=None):
    """A fresh box_opening-like class per test. `seen` collects what the task saw; the task raises
    `error` when it reaches phase `fail_in`."""

    class Capability(CapabilityEngine):
        actions = [Action("run", "Run")]

        def execute(self, action_id, params, ctx):
            seen["arm_type"] = type(self.arm).__name__
            seen["ctx_type"] = type(ctx).__name__
            seen["at_start"] = len(fake.goals.get(EXECUTE, []))
            for i, phase in enumerate(PHASES):
                ctx.progress(phase, i / len(PHASES), "")
                seen.setdefault("at_first_progress", len(fake.goals.get(EXECUTE, [])))
                if phase == fail_in:
                    raise error
            seen["at_end"] = len(fake.goals.get(EXECUTE, []))
            return {"done": action_id}

    return Capability


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def env(ros):
    fake = FakeDriver()
    client_node = Node("hook_e2e_client")
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(client_node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    engines = []
    logs = []

    class Logger:
        """The engine's logger, with its info lines also collected."""

        def __init__(self, real):
            self._real = real

        def info(self, msg, *a, **k):
            logs.append(str(msg))
            return self._real.info(msg, *a, **k)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def make(cls, name="box_opening", **params):
        overrides = [Parameter("use_orchestrator", value=False)]
        overrides += [Parameter(k, value=v) for k, v in params.items()]
        engine = hook.install(cls)(name, parameter_overrides=overrides)
        real = engine.get_logger()
        engine.get_logger = lambda: Logger(real)
        engines.append(engine)
        executor.add_node(engine)
        fake.set_joints(START)
        time.sleep(SETTLE_S)  # the fake publishes state at 50 Hz
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

    yield SimpleNamespace(fake=fake, make=make, run=run, logs=logs)
    for engine in engines:
        engine.stop()
    executor.shutdown()
    spinner.join()
    for engine in engines:
        engine.destroy_node()
    client_node.destroy_node()
    fake.destroy_node()


def on(**extra):
    return {"puppy.enabled": True, "puppy.preset": "subtle", "puppy.person_zone": SCENE, **extra}


def goals(env):
    return env.fake.goals.get(EXECUTE, [])


def duration_s(goal):
    last = goal.trajectory.points[-1].time_from_start
    return last.sec + last.nanosec * 1e-9


def test_box_opening_success_plays_boundary_beats_only(env):
    seen = {}
    engine = env.make(make_capability(env.fake, seen), **on())
    arm_before = engine.arm
    result = env.run(engine).result
    assert result.status == Result.STATUS_SUCCESS
    assert json.loads(result.payload_json) == {"done": "run"}
    assert seen["arm_type"] == "PuppyArm" and seen["ctx_type"] == "PuppyCtx"
    assert engine.arm is arm_before  # restored

    before = seen["at_start"]
    assert before <= 1  # eager, or nothing when the subtle budget keeps the room for the proud
    # no beat goal between the first progress call and execute's return
    assert seen["at_first_progress"] == before == seen["at_end"]
    assert len(goals(env)) == before + 1  # exactly one proud after
    assert sum(duration_s(g) for g in goals(env)) <= SUBTLE_BUDGET_S + 1e-6
    assert all(math.isfinite(p) for g in goals(env) for p in g.trajectory.points[-1].positions)


def test_recoverable_failure_in_a_precision_phase_gets_no_beat_and_the_same_error(env):
    seen = {}
    error = EngineError("press_failed", "the press did not take", recoverable=True)
    engine = env.make(make_capability(env.fake, seen, fail_in="press", error=error), **on())
    result = env.run(engine).result
    assert result.status == Result.STATUS_FAILED
    assert result.error_code == "press_failed" and result.recoverable is True
    assert len(goals(env)) == seen["at_start"] <= 1  # the "before" beat at most, nothing after
    assert type(engine.arm).__name__ != "PuppyArm"


def test_non_recoverable_failure_plays_nothing_after(env):
    seen = {}
    error = EngineError("fatal", "no", recoverable=False)
    engine = env.make(make_capability(env.fake, seen, fail_in="home", error=error), **on())
    result = env.run(engine).result
    assert result.status == Result.STATUS_FAILED
    assert result.error_code == "fatal" and result.recoverable is False
    assert len(goals(env)) == seen["at_start"]


def test_another_exception_propagates_with_no_beats(env):
    seen = {}
    engine = env.make(make_capability(env.fake, seen, fail_in="look", error=KeyError("bug")), **on())
    result = env.run(engine).result
    assert result.status == Result.STATUS_FAILED and result.error_code == "exception"
    assert len(goals(env)) == seen["at_start"]
    assert type(engine.arm).__name__ != "PuppyArm"


def test_empty_person_zone_runs_without_puppy_mode(env):
    seen = {}
    engine = env.make(make_capability(env.fake, seen), **on(**{"puppy.person_zone": ""}))
    assert env.run(engine).result.status == Result.STATUS_SUCCESS
    assert not goals(env)
    assert seen["arm_type"] != "PuppyArm" and seen["ctx_type"] != "PuppyCtx"
    assert any("puppy mode off: no person zone" in line for line in env.logs)


def test_a_bad_person_zone_runs_without_puppy_mode(env):
    seen = {}
    engine = env.make(make_capability(env.fake, seen), **on(**{"puppy.person_zone": "not json"}))
    assert env.run(engine).result.status == Result.STATUS_SUCCESS
    assert not goals(env) and seen["arm_type"] != "PuppyArm"
    assert any(line.startswith("puppy mode off:") for line in env.logs)


def test_a_capability_without_rules_runs_without_puppy_mode(env):
    seen = {}
    engine = env.make(make_capability(env.fake, seen), name="no_such_capability", **on())
    assert env.run(engine).result.status == Result.STATUS_SUCCESS
    assert not goals(env) and seen["arm_type"] != "PuppyArm"
    assert any("has no rules" in line for line in env.logs)


def test_an_unknown_preset_runs_without_puppy_mode(env):
    seen = {}
    engine = env.make(make_capability(env.fake, seen), **on(**{"puppy.preset": "loud"}))
    assert env.run(engine).result.status == Result.STATUS_SUCCESS
    assert not goals(env) and seen["arm_type"] != "PuppyArm"


def test_flatbot_person_zone_is_the_chair_scene(env, monkeypatch):
    asked = []
    real = hook._scene

    def spy(text):
        asked.append(text)
        return real(text)

    monkeypatch.setattr(hook, "_scene", spy)
    seen = {}
    engine = env.make(make_capability(env.fake, seen), **on(**{"puppy.person_zone": "flatbot"}))
    assert env.run(engine).result.status == Result.STATUS_SUCCESS
    assert asked == ["flatbot"]
    assert seen["arm_type"] == "PuppyArm"  # the chair scene parsed and Puppy Mode went on
    scene = real("flatbot")
    assert scene.person and scene.obstacles
    assert isinstance(real(SCENE), Scene)


def test_held_is_fail_safe():
    def engine_with(gripper_state):
        return SimpleNamespace(arm=SimpleNamespace(gripper_state=gripper_state))

    def reading(position, present=True):
        return lambda: SimpleNamespace(position=position, present=present)

    def missing():
        raise RuntimeError("no /gripper_state")

    assert hook.held_source(engine_with(reading(0.0)))() is False
    assert hook.held_source(engine_with(reading(0.6)))() is True
    assert hook.held_source(engine_with(reading(float("nan"))))() is True
    assert hook.held_source(engine_with(reading(0.0, present=False)))() is True
    assert hook.held_source(engine_with(missing))() is True
    assert hook.held_source(SimpleNamespace(arm=SimpleNamespace()))() is True


def test_flatbot_scene_has_a_head_target():
    """The chair layers are boxes only: the head is the person's highest box (the layer's "head")."""
    import numpy as np

    targets = hook._targets(hook._scene("flatbot"))
    assert np.allclose(targets.person_head, [-0.01895, -0.36, 0.87885])
