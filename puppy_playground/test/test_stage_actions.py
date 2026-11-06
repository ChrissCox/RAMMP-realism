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

from capability_engine.arm_messages import _impedance_gains

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
    e.fake, e.run, e.at, e.engine = fake, run, at, engine
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
    sp = stage_store.load(env.cand_file, urdf_text, stage_q, Chain.from_urdf(urdf_text))  # endpoints included
    assert sp.margin_m == stage_store.MARGIN_M
    assert sp.method in ("curobo", "min_jerk", "escape") and sp.method == out["method"]
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
    assert g.preemption == 0 and g.gains == _impedance_gains("default")  # QUEUE, the default profile
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


# ---- wrapped readings, load-time checks, ordering ------------------------------------------------


def wrapped(q):
    """What the real driver reads: continuous joints (1, 3, 5, 7) wrapped to (-pi, pi]."""
    q = np.array(q, dtype=float)
    for j in (0, 2, 4, 6):
        q[j] = np.pi - np.mod(np.pi - q[j], 2 * np.pi)
    return q


def turn_shift_is_whole_and_uniform(sent, stored):
    d = np.array(sent) - np.array(stored)
    k = d / (2 * np.pi)
    assert np.allclose(k[:, [1, 3, 5]], 0.0)  # revolute joints untouched
    assert np.allclose(k, np.round(k), atol=1e-9) and np.allclose(k, k[0], atol=1e-9)
    return k[0]


def test_stage_works_from_a_wrapped_rest_reading_and_sends_the_path_on_that_turn(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    reading = HOME.copy()
    reading[2] = 3.141  # the driver's wrapped reading of j3 = -3.137 (a turn away from the stored value)
    env.at(reading)
    res = env.run("stage")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    g, pts = sent(env)
    assert np.abs(pts[0][0] - reading)[[0, 2, 4, 6]].max() <= np.pi
    assert np.abs(pts[0][0] - reading).max() <= 0.02
    turn_shift_is_whole_and_uniform([q for q, _ in pts][60:], [q for _, q, _ in path_points][60:])  # after the blend
    chain = Chain.from_urdf(urdf_text)
    from puppy_motion.geometry import clearances

    for (q, _), (_, want, _) in list(zip(pts, path_points))[300::300]:
        assert np.allclose(clearances(chain, chair_scene(), q, False), clearances(chain, chair_scene(), want, False), atol=1e-9)


def test_rest_works_right_after_stage_when_the_driver_reports_wrapped_joints(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(wrapped(HOME))
    assert env.run("stage").status == Result.STATUS_SUCCESS
    env.fake.goals["execute_joint_trajectory"].clear()
    env.at(wrapped(stage_q))  # j3 reads 0.73, not -5.55
    res = env.run("rest")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    g, pts = sent(env)
    assert np.abs(pts[0][0] - wrapped(stage_q)).max() < 1e-6


def test_recover_near_stage_and_on_the_path_work_with_wrapped_readings(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(wrapped(stage_q + 0.05))
    assert env.run("recover").status == Result.STATUS_SUCCESS
    env.fake.goals["execute_joint_trajectory"].clear()
    i = len(path_points) // 3
    env.at(wrapped(path_points[i][1]))
    res = env.run("recover")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    g, pts = sent(env)
    assert np.abs(shortest(urdf_text, pts[-1][0], HOME)).max() < 1e-6


def shortest(urdf_text, a, b):
    from puppy_motion.joints import shortest_delta

    return shortest_delta(Chain.from_urdf(urdf_text), a, b)


@pytest.mark.parametrize("edit", ["nan_q_rest", "nan_stage_q", "nan_margin", "negative_margin", "inf_margin"])
def test_stage_refuses_non_finite_or_negative_header_fields(env, urdf_text, stage_q, path_points, edit):
    approve(env, urdf_text, stage_q, path_points)
    doc = json.loads(env.stage_file.read_text())
    if edit == "nan_q_rest":
        doc["q_rest"][0] = float("nan")
    elif edit == "nan_stage_q":
        doc["stage_q"][3] = float("nan")
    elif edit == "nan_margin":
        doc["margin_m"] = float("nan")
    elif edit == "inf_margin":
        doc["margin_m"] = float("inf")
    else:
        doc["margin_m"] = -0.05
    env.stage_file.write_text(json.dumps(doc))  # json.dumps writes NaN/Infinity, which json.loads reads back
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_stage_path")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_a_changed_approved_file_is_rechecked_after_a_cached_ok(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME)
    assert env.run("stage").status == Result.STATUS_SUCCESS
    env.fake.goals["execute_joint_trajectory"].clear()
    bad = [(t, q.copy(), dq.copy()) for t, q, dq in path_points]
    mid = len(bad) // 2
    bad[mid] = (bad[mid][0], bad[mid][1] + 0.5, bad[mid][2])  # a jump: breaks the caps
    approve(env, urdf_text, stage_q, bad)
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "unsafe_stage_path")
    assert not env.fake.goals["execute_joint_trajectory"]


KINOVA_HOME = np.deg2rad([0.0, 15.0, 180.0, -130.0, 0.0, 55.0, 90.0])  # Kinova Home, in the engine's joint convention


def test_plan_stage_plans_from_a_kinova_home_like_start_and_stage_replays_it(env, urdf_text, stage_q):
    from puppy_motion.escape import grown_scene
    from puppy_motion.stage import path_violations

    env.at(KINOVA_HOME)
    res = env.run("plan_stage")
    assert res.status == Result.STATUS_SUCCESS, (res.error_code, res.payload_json)
    chain = Chain.from_urdf(urdf_text)
    cand = stage_store.load(env.cand_file, urdf_text, stage_q, chain)
    assert np.allclose(cand.points[0][1], KINOVA_HOME, atol=1e-6) and np.allclose(cand.q_rest, KINOVA_HOME, atol=1e-6)
    assert not path_violations(cand.points, chain, grown_scene(chair_scene(), stage_store.MARGIN_M))
    env.cand_file.replace(env.stage_file)  # approve
    env.at(KINOVA_HOME)
    res = env.run("stage")
    assert res.status == Result.STATUS_SUCCESS, (res.error_code, res.payload_json)
    assert env.fake.goals["execute_joint_trajectory"]


def test_stage_refuses_from_elsewhere_than_the_approved_start(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME + 0.3)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "not_at_start")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_plan_stage_from_a_start_in_collision_refuses_bad_start_and_writes_nothing(env, monkeypatch):
    monkeypatch.setattr(
        env.engine, "_obstacle_clearances", lambda chain, q, margin: np.array([0.0, 0.2, 0.2, 0.2])
    )
    env.at(HOME)
    res = env.run("plan_stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_start")
    assert "obstacle" in json.loads(res.payload_json)["message"]
    assert not env.cand_file.exists() and not env.fake.goals["plan_to_joints"]


def test_plan_stage_from_a_start_past_a_joint_limit_refuses_bad_start(env, urdf_text):
    chain = Chain.from_urdf(urdf_text)
    q = HOME.copy()
    q[1] = chain.pos_limits[1, 1] + 0.05  # a limited joint
    env.at(q)
    res = env.run("plan_stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_start")
    assert not env.cand_file.exists()


def test_the_obstacle_floor_is_anchored_like_the_person_floor(env, urdf_text, stage_q, path_points, monkeypatch):
    approve(env, urdf_text, stage_q, path_points)
    chain = Chain.from_urdf(urdf_text)
    refs = []
    real = env.engine._path_violations

    def spy(pts, chain_, margin, person_floor_ref=None, obstacle_floor_ref=None):
        if obstacle_floor_ref is not None:
            refs.append(np.array(obstacle_floor_ref))
        return real(pts, chain_, margin, person_floor_ref, obstacle_floor_ref)

    monkeypatch.setattr(env.engine, "_path_violations", spy)
    c_stored = env.engine._obstacle_clearances(chain, path_points[0][1], stage_store.MARGIN_M)
    for off in (0.005, 0.01, 0.015):  # the reviewer's case: a j1 offset from the stored start
        refs.clear()
        reading = HOME.copy()
        reading[0] += off
        env.at(reading)
        res = env.run("stage")
        assert res.status == Result.STATUS_SUCCESS, (off, res.error_code, res.payload_json)
        c_reading = env.engine._obstacle_clearances(chain, reading, stage_store.MARGIN_M)
        assert np.allclose(refs[0], np.minimum(c_reading, c_stored), atol=1e-9)
        env.fake.goals["execute_joint_trajectory"].clear()
        env.at(stage_q)
        assert env.run("rest").status == Result.STATUS_SUCCESS
        env.fake.goals["execute_joint_trajectory"].clear()


def test_plan_stage_with_no_passing_plan_deletes_a_stale_candidate_and_fails_saying_what_was_tried(env, monkeypatch):
    env.cand_file.write_text("stale")
    monkeypatch.setattr(env.engine, "_candidate_points", lambda *a, **k: None)  # every planner fails
    env.at(HOME)
    res = env.run("plan_stage")
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "no_candidate", True)
    out = json.loads(res.payload_json)
    assert out["action"] == "plan_stage" and out["error_code"] == "no_candidate"
    assert "curobo: no plan" in out["message"] and "min_jerk: no plan" in out["message"] and "escape: no plan" in out["message"]
    assert not env.cand_file.exists()


def test_plan_stage_from_a_wrapped_rest_reading_writes_a_candidate(env, urdf_text, stage_q):
    reading = HOME.copy()
    reading[2] = 3.141
    env.at(reading)
    env.fake.plan_result([HOME, stage_q])
    res = env.run("plan_stage")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    out = json.loads(res.payload_json)
    assert out["violations"] == [] and env.cand_file.exists()
    assert not env.fake.goals["execute_joint_trajectory"]


def test_plan_stage_refuses_a_candidate_path_equal_to_the_stage_path(tmp_path):
    rclpy.init()
    try:
        engine = PuppyPlayground(parameter_overrides=[
            Parameter("use_orchestrator", value=False),
            Parameter("playground.stage_path", value=str(tmp_path / "same.json")),
            Parameter("playground.candidate_path", value=str(tmp_path / "same.json")),
        ])
        with pytest.raises(Exception) as e:
            engine._plan_stage(None, {})
        assert getattr(e.value, "error_code", "") == "bad_config"
        engine.destroy_node()
    finally:
        rclpy.shutdown()


# ---- margin is fixed, stale measurement, odd readings ---------------------------------------------


@pytest.mark.parametrize("margin", [5.0, 1.0, True, "0.002", -0.0, None, 0.0021])
def test_stage_refuses_any_margin_but_the_fixed_one(env, urdf_text, stage_q, path_points, margin):
    approve(env, urdf_text, stage_q, path_points)
    doc = json.loads(env.stage_file.read_text())
    doc["margin_m"] = margin
    env.stage_file.write_text(json.dumps(doc))
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "bad_stage_path")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_load_requires_a_chain_when_given_a_stage_pose(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    with pytest.raises(TypeError):
        stage_store.load(env.stage_file, urdf_text, stage_q)


def test_a_back_and_forth_path_is_refused_at_the_fixed_margin(env, urdf_text, stage_q, path_points):
    """Out to the all-clear sample, back toward rest, out again: legal at a huge margin (nothing
    ever reaches 0.35 m on grown boxes) but refused at exactly MARGIN_M."""
    from puppy_motion.escape import grown_scene
    from puppy_motion.geometry import clearances_batch
    from puppy_motion.stage import path_violations, points_from_q, retime_path

    chain = Chain.from_urdf(urdf_text)
    qs = np.array([q for _, q, _ in path_points])
    c = clearances_batch(chain, chair_scene(), qs, False)[:, 0, :]
    k = int(np.argmax((c[:, [1, 2, 3]] >= 0.35).all(axis=1)))
    j = max(0, k - 200)
    detour = retime_path(points_from_q(np.vstack([qs, qs[j:][::-1][1:], qs[j:][1:]])), chain)
    rules = {v.rule for v in path_violations(detour, chain, grown_scene(chair_scene(), stage_store.MARGIN_M))}
    assert "path_person_monotone" in rules
    stage_store.save(env.stage_file, detour, urdf_text, stage_q)
    env.at(HOME)
    res = env.run("stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "unsafe_stage_path")
    assert not env.fake.goals["execute_joint_trajectory"]


def test_recover_refuses_when_the_reading_changes_between_the_check_and_the_send(
    env, urdf_text, stage_q, path_points, monkeypatch
):
    approve(env, urdf_text, stage_q, path_points)
    q = stage_q + 0.05
    env.at(q)
    reads = iter([q, q + 0.01])
    monkeypatch.setattr(env.engine, "_still_q", lambda: next(reads))
    res = env.run("recover")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "moved")
    assert not env.fake.goals["execute_joint_trajectory"]


@pytest.mark.parametrize("action,at_q", [("stage", "home"), ("rest", "stage")])
def test_the_checkpoint_comes_after_the_slow_checks_and_before_the_measurement(
    env, urdf_text, stage_q, path_points, monkeypatch, action, at_q
):
    from capability_engine.engine import Context

    approve(env, urdf_text, stage_q, path_points)
    events = []
    orig_cp, orig_load, orig_still = Context.checkpoint, env.engine._load, env.engine._still_q

    def cp(self):
        events.append("checkpoint")
        return orig_cp(self)

    def load(*a, **k):
        r = orig_load(*a, **k)
        events.append("load")
        return r

    def still():
        events.append("still_q")
        return orig_still()

    monkeypatch.setattr(Context, "checkpoint", cp)
    monkeypatch.setattr(env.engine, "_load", load)
    monkeypatch.setattr(env.engine, "_still_q", still)
    env.at(HOME if at_q == "home" else stage_q)
    assert env.run(action).status == Result.STATUS_SUCCESS
    i_load, i_still = events.index("load"), events.index("still_q")
    assert i_load < i_still and "checkpoint" in events[i_load:i_still]


def test_a_nan_reading_is_refused_and_nothing_is_sent(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(np.full(7, np.nan))
    for action in ("stage", "rest", "recover", "plan_stage"):
        res = env.run(action)
        assert res.status == Result.STATUS_FAILED and res.error_code
    assert not env.fake.goals["execute_joint_trajectory"]


def test_a_driver_without_a_model_refuses_with_no_model(env, monkeypatch):
    from capability_engine.sim import SimModelError

    def no_model():
        raise SimModelError("no urdf")

    monkeypatch.setattr(env.engine.arm, "robot_description", no_model)
    env.at(HOME)
    res = env.run("plan_stage")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "no_model")


# ---- a refused goal, the failure payload, and the seat onto the reading --------------------------


def test_a_refused_stage_carries_the_action_the_code_and_the_rule_in_its_payload(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME + 0.3)
    res = env.run("stage")
    assert res.error_code == "not_at_start"
    assert json.loads(res.payload_json) == {
        "action": "stage",
        "error_code": "not_at_start",
        "message": "the arm is not at the path's start (rest)",
    }


@pytest.mark.parametrize("action,at_q", [("stage", "home"), ("rest", "stage")])
def test_a_goal_the_driver_refuses_is_refused_not_an_exception(env, urdf_text, stage_q, path_points, action, at_q):
    approve(env, urdf_text, stage_q, path_points)
    env.fake.accept("execute_joint_trajectory", False)
    env.at(HOME if at_q == "home" else stage_q)
    res = env.run(action)
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "refused", True)
    assert json.loads(res.payload_json)["error_code"] == "refused"


@pytest.mark.parametrize("action,at_q", [("stage", "home"), ("rest", "stage")])
def test_an_estop_on_at_the_press_is_estopped_and_not_recoverable(env, urdf_text, stage_q, path_points, action, at_q):
    approve(env, urdf_text, stage_q, path_points)
    env.at(HOME if at_q == "home" else stage_q)
    env.fake.engage_estop()
    time.sleep(SETTLE_S)
    res = env.run(action)
    assert (res.status, res.error_code, res.recoverable) == (Result.STATUS_FAILED, "estopped", False)


def test_recover_reuses_the_driver_error_mapping(env, urdf_text, stage_q, path_points):
    approve(env, urdf_text, stage_q, path_points)
    env.fake.set_result("execute_joint_trajectory", -4)  # a path tolerance violation
    env.at(stage_q + 0.05)
    res = env.run("recover")
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "contact")


def sent_points(env):
    g, pts = sent(env)
    return np.array([q for q, _ in pts])


def rule_failures(urdf_text, qs, full, ref=None):
    from puppy_motion.escape import grown_scene
    from puppy_motion.stage import path_violations, points_from_q

    chain = Chain.from_urdf(urdf_text)
    scene = grown_scene(chair_scene(), stage_store.MARGIN_M) if full else chair_scene()
    bad = path_violations(points_from_q(qs), chain, scene, person_floor_ref=ref)
    return bad if full else [v for v in bad if v.rule in ("path_caps", "path_ends")]


def test_stage_from_a_reading_off_rest_starts_at_the_reading_and_ends_at_the_stored_pose(
    env, urdf_text, stage_q, path_points
):
    approve(env, urdf_text, stage_q, path_points)
    reading = HOME.copy()
    reading[1] += 0.015  # j2 sags 0.015 rad off the stored rest
    env.at(reading)
    res = env.run("stage")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    qs = sent_points(env)
    stored = np.array([p[1] for p in path_points])
    assert np.allclose(qs[0], reading, atol=1e-9)
    assert np.allclose(qs[-1], stored[-1], atol=1e-9)  # the end is exact
    assert len(qs) == len(stored)
    after = int(round(1.0 / 0.02)) + 1
    assert np.allclose(qs[after:], stored[after:], atol=1e-9)  # w == 0 after BLEND_S
    chain = Chain.from_urdf(urdf_text)
    clear = lambda x: env.engine._person_clearances(chain, x, stage_store.MARGIN_M)  # noqa: E731
    ref = np.minimum(clear(reading), clear(stored[0]))
    assert rule_failures(urdf_text, qs, True, ref) == []  # the stage rules at MARGIN_M, caps and ends


def test_rest_from_a_reading_off_the_stage_pose_starts_at_the_reading_and_ends_at_rest(
    env, urdf_text, stage_q, path_points
):
    approve(env, urdf_text, stage_q, path_points)
    reading = stage_q.copy()
    reading[1] += 0.015
    env.at(reading)
    res = env.run("rest")
    assert res.status == Result.STATUS_SUCCESS, res.error_code
    qs = sent_points(env)
    stored = np.array([p[1] for p in path_points])[::-1]
    assert np.allclose(qs[0], reading, atol=1e-9)
    assert np.allclose(qs[-1], stored[-1], atol=1e-9)
    assert rule_failures(urdf_text, qs, False) == []


@pytest.mark.parametrize("action", ["stage", "rest"])
def test_a_blended_path_that_breaks_a_rule_is_refused_and_nothing_is_sent(
    env, urdf_text, stage_q, path_points, monkeypatch, action
):
    from types import SimpleNamespace

    approve(env, urdf_text, stage_q, path_points)
    start = path_points[0][1] if action == "stage" else path_points[-1][1]
    reading = start.copy()
    reading[1] += 0.015
    real_full, real_caps = env.engine._path_violations, env.engine._caps_ends
    bad = [SimpleNamespace(rule="path_caps", detail="constructed")]

    def off_start(points):
        return float(np.abs(np.asarray(points[0][1]) - start).max()) > 1e-6

    if action == "stage":  # (rest is checked by _caps_ends only; patching _path_violations would trip _load)
        monkeypatch.setattr(
            env.engine,
            "_path_violations",
            lambda pts, chain, margin, person_floor_ref=None, obstacle_floor_ref=None: bad
            if off_start(pts)
            else real_full(pts, chain, margin, person_floor_ref, obstacle_floor_ref),
        )
    else:
        monkeypatch.setattr(env.engine, "_caps_ends", lambda pts, chain: bad if off_start(pts) else real_caps(pts, chain))
    env.at(reading)
    res = env.run(action)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "unsafe_move")
    assert "constructed" in json.loads(res.payload_json)["message"]
    assert not env.fake.goals["execute_joint_trajectory"]


@pytest.mark.parametrize("action", ["stage", "rest"])
def test_stage_and_rest_refuse_moved_when_the_arm_moves_during_the_checks(
    env, urdf_text, stage_q, path_points, monkeypatch, action
):
    approve(env, urdf_text, stage_q, path_points)
    start = HOME if action == "stage" else stage_q
    env.at(start)
    reads = iter([start, start + 0.01])
    monkeypatch.setattr(env.engine, "_still_q", lambda: next(reads))
    res = env.run(action)
    assert (res.status, res.error_code) == (Result.STATUS_FAILED, "moved")
    assert not env.fake.goals["execute_joint_trajectory"]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_person_floor_is_the_worse_of_the_reading_and_the_stored_start(
    env, urdf_text, stage_q, path_points, monkeypatch, sign
):
    approve(env, urdf_text, stage_q, path_points)
    chain = Chain.from_urdf(urdf_text)
    reading = HOME.copy()
    reading[1] += sign * 0.015  # farther from / closer to the person than the stored start
    refs = []
    real = env.engine._path_violations

    def spy(pts, chain_, margin, person_floor_ref=None, obstacle_floor_ref=None):
        if person_floor_ref is not None:
            refs.append(np.array(person_floor_ref))
        return real(pts, chain_, margin, person_floor_ref, obstacle_floor_ref)

    monkeypatch.setattr(env.engine, "_path_violations", spy)
    env.at(reading)
    res = env.run("stage")
    assert res.status == Result.STATUS_SUCCESS, (res.error_code, res.payload_json)
    assert len(refs) == 1
    c_reading = env.engine._person_clearances(chain, reading, stage_store.MARGIN_M)
    c_stored = env.engine._person_clearances(chain, path_points[0][1], stage_store.MARGIN_M)
    assert np.allclose(refs[0], np.minimum(c_reading, c_stored), atol=1e-9)
    if sign < 0:  # a reading closer to the person than the stored start still uses the reading as the floor
        assert np.any(c_reading < c_stored - 1e-6) and np.allclose(refs[0][c_reading < c_stored], c_reading[c_reading < c_stored])
