"""express, combo and auto against a FakeDriver. Nothing here talks to a real arm."""

import json
import threading
import time
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from capability_engine.arm import QUEUE, _impedance_gains
from capability_engine.engine import EngineError
from fake_driver import FLATBOT_URDF, FakeDriver
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from puppy_motion.beat import Beat
from puppy_motion.behaviors import Targets
from puppy_motion.flatbot import chair_scene
from puppy_motion.idle import LOOK_TARGET, IdleDirector
from puppy_motion.track import Zone
from puppy_motion.urdf import Chain
from puppy_motion.validate import Violation
from puppy_playground import motion, pipeline
from puppy_playground.engine import PuppyPlayground

WAIT_S = 10.0
ACTION_WAIT_S = 240.0
IMPEDANCE = 1
SETTLE_S = 0.3  # the fake publishes state at 50 Hz


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture(scope="module")
def chain():
    return Chain.from_urdf(FLATBOT_URDF.read_text())


@pytest.fixture(scope="module")
def stage_q():
    text = resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text(encoding="utf-8")
    return np.array(json.loads(text)["q"])


@pytest.fixture
def env(tmp_path):
    rclpy.init()
    fake = FakeDriver()
    client_node = Node("playground_motion_client")
    overrides = [
        Parameter("use_orchestrator", value=False),
        Parameter("playground.stage_path", value=str(tmp_path / "stage_path.json")),
        Parameter("playground.candidate_path", value=str(tmp_path / "stage_path.candidate.json")),
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
    request_pub = client_node.create_publisher(String, "/puppy_playground/request", 10)

    def start(action, params=None):
        send = client.send_goal_async(ExecuteAction.Goal(action_id=action, params_json=json.dumps(params or {})))
        wait_until(send.done)
        assert send.result().accepted
        return send.result()

    def finish(handle, timeout_s=ACTION_WAIT_S):
        result = handle.get_result_async()
        wait_until(result.done, timeout_s)
        return result.result().result.result

    def at(q):
        fake.set_joints(q)
        time.sleep(SETTLE_S)

    class Env:
        pass

    e = Env()
    e.fake, e.at, e.start, e.finish = fake, at, start, finish
    e.run = lambda action, params=None: finish(start(action, params))
    e.request = lambda name: request_pub.publish(String(data=name))
    e.log = tmp_path / "playground.jsonl"
    yield e
    client.destroy()
    engine.stop()
    executor.shutdown()
    spinner.join()
    engine.destroy_node()
    client_node.destroy_node()
    fake.destroy_node()
    rclpy.shutdown()


def lines(env):
    return [json.loads(x) for x in env.log.read_text().splitlines()]


def goals(env):
    return env.fake.goals["execute_joint_trajectory"]


def only_trajectory_goals(env):
    return all(not v for k, v in env.fake.goals.items() if k != "execute_joint_trajectory")


def secs(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


class StubCtx:
    def checkpoint(self):
        pass


class StubArm:
    def __init__(self, q, speed=0.0):
        self.q, self.speed, self.sent = np.array(q, dtype=float), speed, []

    def joint_states(self):
        return SimpleNamespace(position=self.q, velocity=np.full(7, self.speed))

    sag = 0.0  # how far the reading falls (joint 7) after each beat: impedance creep

    def execute_joint_trajectory(self, *a, **k):
        self.sent.append((a, k))
        self.q = self.q.copy()
        self.q[6] -= self.sag
        return SimpleNamespace(error_code=0, error_string="")


def test_express_sends_one_impedance_default_goal_with_central_velocities(env, stage_q):
    env.at(stage_q)
    res = env.run("express", {"name": "head_tilt", "size": 0.25})
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    assert len(goals(env)) == 1
    g = goals(env)[0]
    assert g.control_mode == IMPEDANCE and g.gains == _impedance_gains("default") and g.preemption == QUEUE
    assert not env.fake.setpoints["/setpoint/gripper"]
    pts = g.trajectory.points
    q = np.array([p.positions for p in pts])
    dq = np.array([p.velocities for p in pts])
    dt = secs(pts[1]) - secs(pts[0])
    assert np.allclose(q[0], stage_q, atol=1e-9)
    assert np.all(dq[0] == 0) and np.all(dq[-1] == 0)
    assert np.allclose(dq[1:-1], (q[2:] - q[:-2]) / (2 * dt), atol=1e-6)
    assert only_trajectory_goals(env)
    rec = lines(env)[-1]
    assert rec["ok"] and rec["motions"][0]["driver"]["error_code"] == 0 and rec["motions"][0]["checks"] == []


def test_express_refuses_away_from_stage_or_while_moving(env, stage_q, chain):
    env.at(stage_q + 0.05)
    res = env.run("express", {"name": "nod", "size": 0.25})
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "not_at_stage")
    assert not goals(env)
    arm = StubArm(stage_q, speed=0.05)  # moving: the pipeline refuses on the measured speed
    with pytest.raises(EngineError) as e:
        pipeline.play(arm, StubCtx(), chain, chair_scene(), lambda q, h: pytest.fail("built"), False)
    assert e.value.error_code == "not_still" and not arm.sent


def test_a_failed_check_sends_nothing_and_logs_the_rules(env, stage_q, monkeypatch):
    monkeypatch.setattr(pipeline, "check_beat", lambda *a, **k: [Violation("person", 3, "too close")])
    env.at(stage_q)
    res = env.run("express", {"name": "nod", "size": 0.25})
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "check_failed", True)
    assert not goals(env)
    assert lines(env)[-1]["motions"][0]["checks"] == ["person: too close"]


def test_combo_plays_in_order_with_checkpoints(env, stage_q):
    env.at(stage_q)
    res = env.run("combo", {"names": ["nod", "head_tilt", "perk"], "size": 0.25})
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    assert len(goals(env)) == 3
    assert [m["motion"] for m in lines(env)[-1]["motions"]] == ["nod", "head_tilt", "perk"]
    # a cancel during the first beat ends the combo at the next checkpoint
    goals(env).clear()
    env.fake.hold_goals(1.0)
    h = env.start("combo", {"names": ["nod", "head_tilt", "perk"], "size": 0.25})
    wait_until(lambda: env.fake.running() == 1)
    cancel = h.cancel_goal_async()
    wait_until(cancel.done)
    assert env.finish(h).status == Result.STATUS_CANCELLED
    assert len(goals(env)) == 1


def test_auto_plays_seeded_beats_until_cancelled_and_takes_requests(env, stage_q, chain):
    env.at(stage_q)
    director = IdleDirector(chain, chair_scene(), Targets(person_head=LOOK_TARGET), 0.25, 3)
    want = director.next(stage_q, False, 0.0)[0].name
    h = env.start("auto", {"size": 0.25, "seed": 3})
    wait_until(lambda: len(goals(env)) >= 1, 60.0)
    wait_until(lambda: env.log.exists() and lines(env), 60.0)
    assert lines(env)[0]["motion"] == want and lines(env)[0]["action"] == "auto"
    env.request("not_a_thing")
    env.request("nod")
    wait_until(lambda: any(m["motion"] == "nod" and m.get("ok") for m in lines(env)), 60.0)
    cancel = h.cancel_goal_async()
    wait_until(cancel.done)
    assert env.finish(h).status == Result.STATUS_CANCELLED
    assert only_trajectory_goals(env)


def test_gripper_stale_or_mid_range_counts_as_held():
    now = [100.0]
    w = motion.GripperWatch(clock=lambda: now[0])
    assert w.held()  # no message yet
    for pos, held in [(0.0, False), (0.05, False), (0.06, True), (0.5, True), (0.92, True), (0.93, False), (1.0, False)]:
        w.update(pos)
        assert w.held() is held, pos
    w.update(0.0)
    now[0] += 0.9
    assert not w.held()
    now[0] += 0.2  # older than 1 s
    assert w.held()


def test_a_nan_gripper_position_counts_as_held():
    eng = SimpleNamespace(arm=SimpleNamespace(gripper_state=lambda: SimpleNamespace(position=float("nan"))))
    assert PuppyPlayground._held(eng) is True


def test_held_gripper_mutes_the_beat(env, stage_q):
    env.fake.set_gripper(0.5)
    env.at(stage_q)
    res = env.run("express", {"name": "wag", "size": 0.25})  # not while holding something
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "no_beat")
    assert not goals(env)


def test_estop_mid_beat_ends_auto_without_retry(env, stage_q):
    env.at(stage_q)
    env.fake.hold_goals(5.0)
    h = env.start("auto", {"size": 0.25, "seed": 1})
    wait_until(lambda: env.fake.running() == 1, 60.0)
    env.fake.engage_estop()
    res = env.finish(h, 30.0)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "estopped")
    time.sleep(1.0)
    assert len(goals(env)) == 1


def test_no_gripper_goal_is_ever_sent(env, stage_q, chain):
    env.at(stage_q)
    assert env.run("combo", {"names": ["nod", "perk"], "size": 0.5}).status == Result.STATUS_SUCCESS
    assert only_trajectory_goals(env)
    assert all(list(g.trajectory.joint_names) == list(chain.joint_names) for g in goals(env))


def test_a_beat_that_starts_far_from_the_measured_posture_is_skipped(stage_q, chain):
    arm = StubArm(stage_q)
    off = stage_q + 0.05
    pts = [(0.0, off, np.zeros(7)), (0.02, off, np.zeros(7))]
    log = []
    with pytest.raises(EngineError) as e:
        pipeline.play(
            arm, StubCtx(), chain, chair_scene(), lambda q, h: Beat("x", "calm", pts, Zone.FREE), False, log=log.append
        )
    assert e.value.error_code == "start_mismatch" and e.value.recoverable
    assert not arm.sent and log and log[0]["ok"] is False


def test_a_cached_beat_slightly_off_is_shifted_onto_the_measured_pose_and_checked(stage_q, chain, monkeypatch):
    scene = chair_scene()
    source = motion.BeatSource(chain, scene, stage_q)
    measured = stage_q.copy()
    measured[1] += 0.01
    measured[4] -= 0.01
    arm = StubArm(measured)
    seen = []
    real = pipeline.check_beat

    def spy(points, q, *a, **k):
        seen.append((np.array(points[0][1]), np.array(q)))
        return real(points, q, *a, **k)

    monkeypatch.setattr(pipeline, "check_beat", spy)
    rec = pipeline.play(arm, StubCtx(), chain, scene, lambda q, h: source.beat("head_tilt", 0.25, q, h), False)
    assert rec["ok"] and len(arm.sent) == 1
    sent_points = arm.sent[0][0][1]
    assert np.array_equal(sent_points[0][1], measured) and np.array_equal(seen[0][0], measured)
    assert np.allclose(sent_points[-1][1], measured)
    assert np.all(sent_points[0][2] == 0) and np.all(sent_points[-1][2] == 0)
    assert ("stage", "head_tilt", 0.25, False, 0) in source._cache  # composed once for the stage pose


def test_a_beat_far_from_the_stage_pose_is_composed_live_at_the_measured_pose(stage_q, chain):
    scene = chair_scene()
    source = motion.BeatSource(chain, scene, stage_q)
    measured = stage_q.copy()
    measured[1] += 0.05
    measured[3] -= 0.05
    arm = StubArm(measured)
    rec = pipeline.play(arm, StubCtx(), chain, scene, lambda q, h: source.beat("nod", 0.25, q, h), False)
    assert rec["ok"] and len(arm.sent) == 1
    assert np.array_equal(arm.sent[0][0][1][0][1], measured)
    assert not source._cache  # not cached: composed at the measured pose


def test_a_shifted_beat_that_fails_its_check_sends_nothing(stage_q, chain, monkeypatch):
    scene = chair_scene()
    source = motion.BeatSource(chain, scene, stage_q)
    measured = stage_q.copy()
    measured[1] += 0.01
    arm = StubArm(measured)
    seen = []

    def refuse(points, q, *a, **k):
        seen.append(np.array(points[0][1]))
        return [Violation("obstacle", 5, "shifted into the seat")]

    monkeypatch.setattr(pipeline, "check_beat", refuse)
    with pytest.raises(EngineError) as e:
        pipeline.play(arm, StubCtx(), chain, scene, lambda q, h: source.beat("nod", 0.25, q, h), False)
    assert e.value.error_code == "check_failed" and not arm.sent
    assert np.array_equal(seen[0], measured)  # it was the shifted beat that was checked


def test_express_plays_when_the_driver_reports_continuous_joints_wrapped(env, stage_q, chain):
    wrapped = stage_q.copy()
    for j in (0, 2, 4, 6):
        wrapped[j] = (stage_q[j] + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(wrapped - stage_q).max() > 1.0  # stage_pose.json is on another turn
    env.at(wrapped)
    res = env.run("express", {"name": "head_tilt", "size": 0.25})
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    assert len(goals(env)) == 1
    first = np.array(goals(env)[0].trajectory.points[0].positions)
    assert np.allclose(first, wrapped, atol=1e-9)  # the beat was shifted to the reading's turn


def test_nan_readings_fail_closed_and_send_nothing(stage_q, chain):
    for field, code in (("velocity", "not_still"), ("position", "bad_state")):
        arm = StubArm(stage_q)
        orig = arm.joint_states

        def js(orig=orig, field=field):
            r = orig()
            bad = np.array(getattr(r, field), dtype=float)
            bad[3] = np.nan
            return SimpleNamespace(**{**vars(r), field: bad})

        arm.joint_states = js
        with pytest.raises(EngineError) as e:
            pipeline.play(arm, StubCtx(), chain, chair_scene(), lambda q, h: pytest.fail("built"), False)
        assert e.value.error_code == code and e.value.recoverable and not arm.sent


def test_a_sagging_reading_ends_play_as_drifted_before_total_drift_passes_the_anchor(stage_q, chain):
    scene = chair_scene()
    source = motion.BeatSource(chain, scene, stage_q)
    arm = StubArm(stage_q)
    arm.sag = 0.005
    beats = 0
    with pytest.raises(EngineError) as e:
        for _ in range(20):
            pipeline.play(
                arm, StubCtx(), chain, scene, lambda q, h: source.beat("head_tilt", 0.25, q, h), False,
                anchors=[stage_q],
            )
            beats += 1
    assert e.value.error_code == "drifted" and not e.value.recoverable
    assert len(arm.sent) == beats
    assert abs(arm.q[6] - stage_q[6]) <= pipeline.ANCHOR_TOL_RAD + 2 * arm.sag
    sent_before = len(arm.sent)
    with pytest.raises(EngineError):
        pipeline.play(arm, StubCtx(), chain, scene, lambda q, h: source.beat("head_tilt", 0.25, q, h), False, anchors=[stage_q])
    assert len(arm.sent) == sent_before  # nothing sent after it


def test_auto_ends_as_drifted_when_a_beat_leaves_the_arm_off_the_stage_pose(env, stage_q):
    env.at(stage_q)
    after = stage_q.copy()
    after[6] += 0.03
    env.fake.finish_at("execute_joint_trajectory", joints=after)
    res = env.finish(env.start("auto", {"size": 0.25, "seed": 1}), 90.0)
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "drifted", False)
    assert len(goals(env)) == 1


@pytest.mark.parametrize(
    "code,expected,recoverable",
    [(-6, "control_taken", False), (-4, "contact", False), (-5, "contact", False), (-1, "driver_error", True)],
)
def test_driver_aborts_end_combo_and_auto(env, stage_q, code, expected, recoverable):
    env.at(stage_q)
    env.fake.set_result("execute_joint_trajectory", code)
    res = env.run("combo", {"names": ["nod", "perk"], "size": 0.25})
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, expected, recoverable)
    assert len(goals(env)) == 1
    if recoverable:
        return  # auto carries on after a recoverable driver error
    goals(env).clear()
    res = env.finish(env.start("auto", {"size": 0.25, "seed": 1}), 90.0)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, expected)
    assert len(goals(env)) == 1


def test_the_beat_cache_survives_actions(env, stage_q, monkeypatch):
    composed = []
    real = motion.BeatSource._compose

    def spy(self, name, size, q, held, seed):
        composed.append((name, size))
        return real(self, name, size, q, held, seed)

    monkeypatch.setattr(motion.BeatSource, "_compose", spy)
    env.at(stage_q)
    for _ in range(2):
        assert env.run("express", {"name": "nod", "size": 0.25}).status == Result.STATUS_SUCCESS
    assert composed == [("nod", 0.25)]
    assert len(goals(env)) == 2


def test_bad_size_and_seed_are_refused_not_clamped(env, stage_q):
    for bad in (0.1, 1.5, True, "big", None, float("nan")):
        with pytest.raises(EngineError) as e:
            motion.parse_size({"size": bad})
        assert e.value.error_code == "bad_params" and e.value.recoverable
    assert motion.parse_size({}) == 0.25 and motion.parse_size({"size": 1}) == 1.0
    for bad in (-1, True, None, 1.5, "3"):
        with pytest.raises(EngineError) as e:
            motion.parse_seed({"seed": bad})
        assert e.value.error_code == "bad_params"
    assert motion.parse_seed({}) == 1 and motion.parse_seed({"seed": 0}) == 0
    env.at(stage_q)
    res = env.run("express", {"name": "nod", "size": 3.0})
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_params")
    assert not goals(env)


def test_a_goal_the_driver_refuses_ends_express_as_refused(env, stage_q):
    env.at(stage_q)
    env.fake.accept("execute_joint_trajectory", False)
    res = env.run("express", {"name": "nod", "size": 0.25})
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "refused", True)
    payload = json.loads(res.payload_json)
    assert payload["action"] == "express" and payload["error_code"] == "refused"


def test_an_estop_on_at_the_press_ends_express_and_auto_as_estopped(env, stage_q):
    env.at(stage_q)
    env.fake.engage_estop()
    time.sleep(SETTLE_S)
    res = env.run("express", {"name": "nod", "size": 0.25})
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "estopped", False)
    res = env.finish(env.start("auto", {"size": 0.25, "seed": 1}), 90.0)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "estopped")


def test_a_false_result_from_the_arm_is_refused_not_an_attribute_error(stage_q, chain):
    class FalseArm(StubArm):
        last_error = "rejected: not ready"

        def execute_joint_trajectory(self, *a, **k):
            return False

    arm = FalseArm(stage_q)
    source = motion.BeatSource(chain, chair_scene(), stage_q)
    with pytest.raises(EngineError) as e:
        pipeline.play(arm, StubCtx(), chain, chair_scene(), lambda q, h: source.beat("head_tilt", 0.25, q, h), False)
    assert (e.value.error_code, e.value.recoverable) == ("refused", True) and "rejected: not ready" in str(e.value)
