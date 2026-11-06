"""The proxies and the beat player, against a FakeDriver. Nothing here talks to a real arm.

Each test gets an ArmInterface inside a begun action (on a fresh node, spun with the fake) and a
Player wired to it through PuppyArm, with a stub director."""

import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from capability_engine.arm import HALTED, IMPEDANCE, POSITION, PREEMPTED, QUEUE, ArmInterface
from capability_engine.driver_state import ControlTaken, EStopped
from capability_engine.engine import Cancelled
from fake_driver import FLATBOT_URDF, FakeCtx, FakeDriver
from rammp_arm import IMPEDANCE as IMPEDANCE_PARAMS
from rammp_arm import JOINTS
from rammp_arm_interfaces.msg import ImpedanceGains
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict

from puppy_mode_ros.player import Player
from puppy_mode_ros.proxies import PuppyArm, PuppyCtx
from puppy_motion.beat import Beat, to_points
from puppy_motion.director import AfterMove, BeforeMove, Phase
from puppy_motion.geometry import Box, Scene
from puppy_motion.track import DT, Zone
from puppy_motion.urdf import Chain

START = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])
EXECUTE = "execute_joint_trajectory"
SETTLE_S = 0.3  # the fake publishes state at 50 Hz


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def make_beat(zone=Zone.FREE, name="nod", start=START, amp=0.05, length_s=1.6):
    """A small raised-cosine wrist turn that starts and ends at `start`, at rest."""
    n = int(round(length_s / DT)) + 1
    k = np.arange(n)
    q = np.tile(np.asarray(start, dtype=float), (n, 1))
    q[:, 6] += amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (n - 1)))
    return Beat(name, "happy", to_points(SimpleNamespace(q=q)), zone)


class StubDirector:
    """Records the events; answers each with `beats` (a list, or a function of the event)."""

    def __init__(self, beats=()):
        self.events, self.beats, self.calls = [], beats, []

    def on_event(self, event, q_now, held):
        self.events.append(event)
        self.calls.append((np.array(q_now), held))
        return self.beats(event) if callable(self.beats) else list(self.beats)


class Arm:
    """The real arm with joint_states() changed by `change(js)`."""

    def __init__(self, arm, change):
        self._arm, self._change = arm, change

    def joint_states(self):
        return self._change(self._arm.joint_states())

    def __getattr__(self, name):
        return getattr(self._arm, name)


def moving(js):
    js.velocity = np.full(7, 0.05)
    return js


@pytest.fixture
def env(ros):
    fake = FakeDriver()
    node = Node("puppy_player_engine")
    arm = ArmInterface(node, ReentrantCallbackGroup())
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    fake.set_joints(START)
    time.sleep(SETTLE_S)
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    arm.joint_states()  # warm: the state subscription has its first message
    chain = Chain.from_urdf(FLATBOT_URDF.read_text())
    scene = Scene(
        person=[Box(np.array([5.0, 5.0, 5.0]), np.array([0.1, 0.1, 0.1]))],
        obstacles=[Box(np.array([-5.0, 5.0, 5.0]), np.array([0.1, 0.1, 0.1]))],
    )
    logs = []

    def build(director=None, real_arm=None, **kw):
        player = Player(real_arm or arm, ctx, director or StubDirector(), chain, scene, log=logs.append, **kw)
        return player, PuppyArm(arm, player)

    yield SimpleNamespace(fake=fake, arm=arm, ctx=ctx, chain=chain, scene=scene, logs=logs, build=build)
    arm.end_action()
    executor.shutdown()
    spinner.join()
    node.destroy_node()
    fake.destroy_node()


def goals(env):
    return env.fake.goals[EXECUTE]


def serialized(fake):
    out = {}
    for name, items in fake.goals.items():
        for goal in items:
            goal = copy.copy(goal)
            goal.sender_id = ""
            out.setdefault(name, []).append(dict(message_to_ordereddict(goal)))
    return out


def test_beat_copies_the_tasks_control_mode(env):
    player, puppy = env.build()
    assert puppy.go_to_joint_config(START, control_mode=IMPEDANCE, gains="soft")
    player.play([make_beat()])
    (g,) = goals(env)
    assert g.control_mode == IMPEDANCE
    assert g.gains.profile == ImpedanceGains.PROFILE_SOFT
    assert g.preemption == QUEUE
    assert list(g.trajectory.joint_names) == list(JOINTS)


def test_a_position_task_gets_position_beats(env):
    player, puppy = env.build()
    assert puppy.go_to_joint_config(START)
    player.play([make_beat()])
    (g,) = goals(env)
    assert g.control_mode == POSITION


def test_near_user_beat_forces_impedance(env):
    player, puppy = env.build()
    assert puppy.go_to_joint_config(START)  # POSITION
    player.play([make_beat(Zone.NEAR_USER)])
    (g,) = goals(env)
    assert g.control_mode == IMPEDANCE
    assert g.gains.profile == ImpedanceGains().profile  # "default"


def test_near_user_beat_keeps_the_tasks_gains(env):
    player, puppy = env.build()
    assert puppy.go_to_joint_config(START, gains="soft")  # POSITION, soft gains
    player.play([make_beat(Zone.NEAR_USER)])
    (g,) = goals(env)
    assert g.control_mode == IMPEDANCE and g.gains.profile == ImpedanceGains.PROFILE_SOFT


def test_no_move_recorded_yet_is_impedance_default(env):
    player, _ = env.build()
    player.play([make_beat()])
    (g,) = goals(env)
    assert g.control_mode == IMPEDANCE
    assert g.gains.profile == ImpedanceGains().profile


def test_move_calls_record_impedance_by_keyword_and_position(env):
    player, puppy = env.build()
    knots = [START + np.array([0, 0, 0, 0, 0, 0, 0.2])]
    assert puppy.move_joints(knots, impedance=IMPEDANCE_PARAMS)
    assert player.control_mode == IMPEDANCE
    assert player.gains == {k: IMPEDANCE_PARAMS[k] for k in ("kq", "zeta", "torque_limit")}
    assert puppy.move_joints([START + np.array([0, 0, 0, 0, 0, 0, 0.1])])
    assert player.control_mode == POSITION and player.gains is None
    assert puppy.move_joints(knots, None, 1.0, IMPEDANCE_PARAMS)
    assert player.control_mode == IMPEDANCE


def test_moving_arm_skips_the_beat(env):
    player, _ = env.build(real_arm=Arm(env.arm, moving))
    player.play([make_beat()])
    assert not goals(env) and not player.disabled
    assert any("moving" in line for line in env.logs)


def test_nan_velocity_counts_as_moving(env):
    def nan(js):
        js.velocity = np.full(7, np.nan)
        return js

    player, _ = env.build(real_arm=Arm(env.arm, nan))
    player.play([make_beat()])
    assert not goals(env)


def test_non_finite_q_skips_the_beat(env):
    def bad(js):
        js.position = js.position.copy()
        js.position[3] = np.nan
        return js

    player, _ = env.build(real_arm=Arm(env.arm, bad))
    player.play([make_beat()])
    assert not goals(env) and not player.disabled
    assert any("finite" in line for line in env.logs)


def test_a_beat_that_starts_elsewhere_is_skipped(env):
    player, _ = env.build()
    elsewhere = START.copy()
    elsewhere[0] += 0.01
    player.play([make_beat(start=elsewhere)])
    assert not goals(env) and not player.disabled
    near = START.copy()
    near[0] += 0.001  # inside the 0.002 rad tolerance: sent, and starts exactly at the measured q
    player.play([make_beat(start=near)])
    (g,) = goals(env)
    assert np.array(g.trajectory.points[0].positions) == pytest.approx(START, abs=1e-9)
    assert np.array(g.trajectory.points[-1].positions) == pytest.approx(START, abs=1e-9)


def test_the_driver_reports_a_continuous_joint_wrapped(env):
    wrapped = START.copy()
    wrapped[2] -= 2.0 * np.pi  # the same posture, one turn away
    env.fake.set_joints(wrapped)
    time.sleep(SETTLE_S)
    player, _ = env.build()
    player.play([make_beat()])  # the beat was made on the other turn
    (g,) = goals(env)
    assert g.trajectory.points[0].positions[2] == pytest.approx(wrapped[2], abs=1e-9)
    assert g.trajectory.points[-1].positions[2] == pytest.approx(wrapped[2], abs=1e-9)


def test_a_beat_that_fails_check_beat_is_not_sent(env):
    player, _ = env.build()
    player.play([make_beat(amp=1.0, length_s=0.4)])  # far too fast
    assert not goals(env) and not player.disabled
    assert any("skip nod" in line for line in env.logs)


def test_cancel_mid_beat_propagates_and_stops(env):
    player, _ = env.build()
    env.fake.hold_goals(10.0)
    threading.Timer(0.5, env.ctx.cancel).start()
    with pytest.raises(Cancelled):
        player.play([make_beat(), make_beat(name="second")])
    assert len(goals(env)) == 1
    assert env.fake.cancelled(EXECUTE)


def test_refused_beat_disables_and_task_continues(env):
    player, puppy = env.build()
    env.fake.accept(EXECUTE, False)
    player.play([make_beat(), make_beat(name="second")])
    assert player.disabled
    assert len(goals(env)) == 1  # the second beat was dropped
    env.fake.accept(EXECUTE, True)
    player.play([make_beat()])
    assert len(goals(env)) == 1  # disabled: nothing more is sent
    assert puppy.go_to_joint_config(START)  # the task's own goal goes out
    assert len(env.fake.goals["go_to_joint_config"]) == 1


def test_a_false_return_is_a_refusal(env):
    class FalseArm(Arm):
        def execute_joint_trajectory(self, *a, **k):
            return False

    player, _ = env.build(real_arm=FalseArm(env.arm, lambda js: js))
    player.play([make_beat(), make_beat(name="second")])
    assert player.disabled


def test_stale_state_skips(env):
    player, _ = env.build()
    env.fake.pause_publishing()
    time.sleep(0.8)  # older than STATE_STALE_S
    player.play([make_beat()])
    env.fake.resume_publishing()
    assert player.disabled and not goals(env)


def test_no_beats_while_streaming(env):
    player, puppy = env.build()
    with puppy.stream("joint_position"):
        assert player.streaming
        player.play([make_beat()])
        player.emit(AfterMove())
    assert not player.streaming
    assert not goals(env)
    time.sleep(SETTLE_S)
    player.play([make_beat()])
    assert len(goals(env)) == 1


@pytest.mark.parametrize("code, outcome", [(-4, "disabled"), (-5, "disabled"), (-7, "disabled")])
def test_a_driver_error_ends_beats_but_not_the_task(env, code, outcome):
    player, _ = env.build()
    env.fake.set_result(EXECUTE, code)
    player.play([make_beat(), make_beat(name="second")])
    assert player.disabled and len(goals(env)) == 1


def test_preempted_beat_means_control_was_taken(env):
    player, _ = env.build()
    env.fake.set_result(EXECUTE, PREEMPTED)
    with pytest.raises(ControlTaken):
        player.play([make_beat()])


def test_halted_beat_without_estop_means_control_was_taken(env):
    player, _ = env.build()
    env.fake.set_result(EXECUTE, HALTED)
    with pytest.raises(ControlTaken):
        player.play([make_beat()])


def test_estop_propagates(env):
    player, _ = env.build()
    env.fake.engage_estop()
    time.sleep(SETTLE_S)
    with pytest.raises(EStopped):
        player.play([make_beat()])


def test_events_reach_the_director_with_q_and_the_precise_end(env):
    director = StubDirector()
    player, puppy = env.build(director, capability="grab_container")
    assert puppy.go_to_joint_config(START)
    assert director.events == [BeforeMove(False), AfterMove()]
    assert director.calls[0][0] == pytest.approx(START) and director.calls[0][1] is False
    player.emit(Phase("approach", "view -> pre-grasp -> grasp, one motion"))
    assert puppy.go_to_joint_config(START)
    assert director.events[-2:] == [BeforeMove(True), AfterMove()]


def test_the_director_beats_play_at_the_event(env):
    director = StubDirector(beats=lambda e: [make_beat()] if isinstance(e, BeforeMove) else [])
    _, puppy = env.build(director)
    assert puppy.go_to_joint_config(START)
    assert len(goals(env)) == 1  # the BeforeMove beat, before the task's own goal
    assert len(env.fake.goals["go_to_joint_config"]) == 1


def test_a_failing_director_disables_the_player_and_not_the_task(env):
    def boom(event):
        raise RuntimeError("boom")

    player, puppy = env.build(StubDirector(beats=boom))
    assert puppy.go_to_joint_config(START)
    assert player.disabled and not goals(env)


def test_held_reaches_the_director(env):
    director = StubDirector()
    player, _ = env.build(director, held=lambda: True)
    player.emit(Phase("a", "b"))
    assert director.calls[-1][1] is True


def test_proxies_forward_every_call_unchanged(env):
    knots = [START + np.array([0, 0, 0, 0, 0, 0, 0.2])]
    points = [(0.0, START, np.zeros(7)), (1.0, START + 0.01, np.zeros(7))]

    def calls(arm):
        return [
            arm.go_to_joint_config(START, control_mode=IMPEDANCE, gains="soft"),
            arm.move_joints(knots, impedance=IMPEDANCE_PARAMS),
            arm.move_joints(knots),
            arm.execute_joint_trajectory(list(JOINTS), points, control_mode=IMPEDANCE, gains="soft"),
        ]

    plain = calls(env.arm)
    baseline = serialized(env.fake)
    env.fake.goals.clear()
    _, puppy = env.build()
    proxied = calls(puppy)
    assert serialized(env.fake) == baseline
    assert [type(r) for r in proxied] == [type(r) for r in plain]
    assert proxied[0] is True and plain[0] is True
    assert proxied[3].error_code == plain[3].error_code == 0


def test_other_attributes_pass_through_the_arm_proxy(env):
    _, puppy = env.build()
    assert puppy.joint_states().position == pytest.approx(START)
    assert puppy.control_status().estopped is False


def test_ctx_proxy_forwards_progress_first_then_emits_phase():
    order = []

    class Ctx:
        marker = 7

        def progress(self, phase, pct, message="", pausable=True):
            order.append(("progress", phase, pct, message, pausable))
            return "r"

        def checkpoint(self):
            return "ok"

    ctx = PuppyCtx(Ctx(), lambda event: order.append(("event", event)))
    assert ctx.progress("approach", 0.5, "go") == "r"
    assert order == [("progress", "approach", 0.5, "go", True), ("event", Phase("approach", "go"))]
    assert ctx.marker == 7 and ctx.checkpoint() == "ok"
