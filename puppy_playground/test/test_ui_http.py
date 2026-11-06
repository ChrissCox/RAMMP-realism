"""The HTTP layer and controller against a fake backend (no ROS)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from puppy_playground.supervisor import Supervisor
from puppy_playground.ui_node import EXPRESSIONS, UiController, arm_moving, make_server


class FakeBackend:
    def __init__(self):
        self.goals = []  # (token, action_id, params)
        self.cancels = 0
        self.goal_cancels = []
        self.requests = []
        self.control_ready = True

    def send_goal(self, token, action_id, params):
        self.goals.append((token, action_id, params))

    def control_cancel(self):
        self.cancels += 1
        return self.control_ready

    def cancel_goal(self, token):
        self.goal_cancels.append(token)

    def publish_request(self, name):
        self.requests.append(name)


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def ui():
    backend = FakeBackend()
    clock = Clock()
    sup = Supervisor(clock)
    ctl = UiController(backend, sup, clock)
    ctl.estopped(False)
    ctl.discovered()
    server = make_server(ctl, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    class Client:
        def __init__(self):
            self.backend, self.ctl, self.sup, self.clock = backend, ctl, sup, clock

        def post(self, path, body=None, ctype="application/json", raw=None):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=raw if raw is not None else json.dumps(body or {}).encode(),
                method="POST",
                headers={"Content-Type": ctype} if ctype else {},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def get(self, path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                return r.status, r.read()

        def advance(self, seconds):
            for _ in range(round(seconds * 10)):
                clock.t += 0.1
                ctl.arm_still_for(99)
                ctl.tick()

        def finish(self, ok=True, code="", message="", cancelled=False):
            ctl.goal_finished(backend.goals[-1][0], ok, code, message, cancelled=cancelled)

    yield Client()
    server.shutdown()
    server.server_close()


def start_auto(ui):
    assert ui.post("/auto", {"on": True, "size": 0.5})[0] == 200
    ui.advance(8)
    assert ui.backend.goals and ui.backend.goals[-1][1] == "auto"
    return ui.backend.goals[-1][0]


def test_double_post_returns_409_busy(ui):
    assert ui.post("/stage")[0] == 200
    code, body = ui.post("/stage")
    assert (code, body["error"]) == (409, "busy")
    assert ui.post("/express", {"name": "nod", "size": 0.25})[0] == 409
    assert ui.post("/rest")[0] == 409
    assert len(ui.backend.goals) == 1


def test_goal_done_frees_the_ui_and_records_the_refusal(ui):
    ui.post("/stage")
    ui.finish(False, "stage_missing", "no stored path")
    status = json.loads(ui.get("/status")[1])
    assert status["busy"] is False
    assert status["last_refusal"] == {"action": "stage", "error_code": "stage_missing", "message": "no stored path"}
    assert ui.post("/rest")[0] == 200


def test_a_cancelled_goal_is_not_a_refusal(ui):
    ui.post("/stage")
    ui.finish(False, "", "", cancelled=True)
    assert json.loads(ui.get("/status")[1])["last_refusal"] is None


def test_a_stale_goal_done_is_ignored(ui):
    ui.post("/stage")
    ui.ctl.goal_finished(999, True)
    assert ui.post("/rest")[0] == 409


def test_goal_params(ui):
    ui.post("/express", {"name": "nod", "size": 0.5})
    assert ui.backend.goals[0][1:] == ("express", {"name": "nod", "size": 0.5})
    ui.finish()
    ui.post("/combo", {"names": ["nod", "wag"], "size": 0.3})
    assert ui.backend.goals[1][1:] == ("combo", {"names": ["nod", "wag"], "size": 0.3})
    ui.finish()
    for path in ("/stage", "/rest", "/recover"):
        ui.post(path)
        assert ui.backend.goals[-1][1:] == (path[1:], {})
        ui.finish()


def test_bad_input_is_400_and_sends_nothing(ui):
    assert ui.post("/express", {"name": "nope", "size": 0.25})[0] == 400
    assert ui.post("/express", {"size": 0.25})[0] == 400
    assert ui.post("/combo", {"names": [], "size": 0.25})[0] == 400
    assert ui.post("/combo", {"names": ["nod", "x"], "size": 0.25})[0] == 400
    assert ui.post("/nothing")[0] == 404
    assert ui.post("/stage", raw=b"[1]")[0] == 400
    assert ui.backend.goals == []


def test_post_requires_json_content_type(ui):
    for ctype in ("text/plain", "application/x-www-form-urlencoded", None):
        assert ui.post("/stage", ctype=ctype)[0] == 415
        assert ui.post("/stop", ctype=ctype)[0] == 415
    assert ui.backend.goals == [] and ui.backend.cancels == 0
    assert ui.post("/stage", ctype="application/json; charset=utf-8")[0] == 200


def test_size_is_clamped(ui):
    ui.post("/express", {"name": "nod", "size": 9})
    assert ui.backend.goals[0][2]["size"] == 1.0
    ui.finish()
    ui.post("/express", {"name": "nod", "size": -1})
    assert ui.backend.goals[1][2]["size"] == 0.25


def test_auto_goal_carries_size_and_seed(ui):
    start_auto(ui)
    action, params = ui.backend.goals[0][1:]
    assert action == "auto" and params["size"] == 0.5 and "seed" in params


def test_auto_waits_for_discovery_grace_and_quiet(ui):
    ui.post("/auto", {"on": True})
    ui.advance(1.5)
    assert ui.backend.goals == []
    ui.advance(2.0)  # past grace and 3 s of quiet
    assert len(ui.backend.goals) == 1


def test_auto_on_must_be_the_boolean_true(ui):
    for on in ("true", "false", 1, "yes", None):
        ui.post("/auto", {"on": on})
        assert ui.sup.state.value == "off", on


def test_auto_on_during_a_manual_goal_is_409_and_cancels_nothing(ui):
    ui.post("/plan_stage")
    code, body = ui.post("/auto", {"on": True})
    assert (code, body["error"]) == (409, "busy")
    assert ui.backend.cancels == 0 and ui.backend.goal_cancels == []
    assert ui.sup.state.value == "off"


def test_express_during_auto_publishes_the_plain_name_not_a_goal(ui):
    start_auto(ui)
    n = len(ui.backend.goals)
    code, _ = ui.post("/express", {"name": "wag", "size": 0.4})
    assert code == 200
    assert ui.backend.requests == ["wag"]
    assert len(ui.backend.goals) == n


def test_other_posts_during_auto_are_409(ui):
    start_auto(ui)
    for path, body in (("/stage", {}), ("/rest", {}), ("/recover", {}), ("/combo", {"names": ["nod"], "size": 0.3})):
        assert ui.post(path, body)[0] == 409
    assert len(ui.backend.goals) == 1


def test_express_while_auto_is_waiting_is_409(ui):
    ui.ctl.engine_status("task", "executing")
    ui.post("/auto", {"on": True})
    assert ui.post("/express", {"name": "nod", "size": 0.25})[0] == 409
    assert ui.backend.requests == []


def test_a_task_starting_cancels_control_and_goal_handle(ui):
    token = start_auto(ui)
    before = ui.backend.cancels
    ui.ctl.engine_status("rammp_task", "executing")
    assert ui.backend.cancels == before + 1
    assert ui.backend.goal_cancels[-1] == token
    assert ui.sup.state.value == "waiting_for_task"


def test_cancel_is_resent_every_300ms_while_the_auto_goal_still_runs(ui):
    token = start_auto(ui)
    ui.ctl.engine_status("puppy_playground", "executing", {"state": "executing"})
    ui.ctl.engine_status("rammp_task", "executing")
    base = ui.backend.cancels
    for _ in range(6):  # task keeps running; the first cancel was lost
        ui.ctl.engine_status("rammp_task", "executing")
        ui.advance(0.5)
        ui.ctl.engine_status("puppy_playground", "executing", {"state": "executing"})
    assert ui.backend.cancels - base >= 6
    assert ui.backend.goal_cancels.count(token) >= 6


def test_cancel_resend_stops_once_the_playground_reports_idle(ui):
    start_auto(ui)
    ui.ctl.engine_status("rammp_task", "executing")
    ui.advance(0.1)
    ui.ctl.engine_status("puppy_playground", "idle", {"state": "idle"})
    base = ui.backend.cancels
    ui.ctl.engine_status("rammp_task", "executing")
    ui.advance(2.0)
    assert ui.backend.cancels == base


def test_auto_off_cancels_an_active_auto_goal(ui):
    token = start_auto(ui)
    base = ui.backend.cancels
    ui.post("/auto", {"on": False})
    assert ui.backend.cancels > base and token in ui.backend.goal_cancels


def test_auto_is_not_resent_while_the_cancelled_goal_is_still_active(ui):
    token = start_auto(ui)
    ui.ctl.engine_status("t", "executing")
    ui.ctl.engine_status("t", "idle")
    ui.advance(10)
    assert len(ui.backend.goals) == 1  # old goal not finished yet
    ui.ctl.goal_finished(token, False, cancelled=True)
    ui.ctl.engine_status("t", "idle")
    for _ in range(10):
        ui.advance(1)
        ui.ctl.engine_status("t", "idle")
    assert len(ui.backend.goals) == 2


def test_the_ending_of_the_auto_goal_does_not_send_a_cancel(ui):
    token = start_auto(ui)
    base = ui.backend.cancels
    ui.ctl.goal_finished(token, True)
    assert ui.backend.cancels == base


def test_stop_cancels_and_turns_auto_off(ui):
    start_auto(ui)
    code, _ = ui.post("/stop")
    assert code == 200
    assert ui.backend.cancels >= 1
    assert ui.sup.state.value == "off"


def test_stop_is_allowed_while_busy(ui):
    ui.post("/stage")
    assert ui.post("/stop")[0] == 200
    assert ui.backend.cancels == 1 and ui.backend.goal_cancels == [1]


def test_stop_reports_failure_when_control_is_unavailable(ui):
    ui.post("/stage")
    ui.backend.control_ready = False
    code, body = ui.post("/stop")
    assert code == 503 and "error" in body
    assert ui.backend.goal_cancels == [1]  # the goal handle was still cancelled


def test_stop_does_not_forget_a_goal_the_engine_still_runs(ui):
    ui.post("/stage")
    ui.post("/stop")
    ui.ctl.engine_status("puppy_playground", "executing", {"state": "executing"})
    ui.advance(30.0)
    assert json.loads(ui.get("/status")[1])["busy"] is True
    assert ui.post("/rest")[0] == 409
    assert len(ui.backend.goal_cancels) > 1  # the cancel is re-sent while it runs


def test_stop_releases_the_goal_once_the_engine_reports_idle(ui):
    ui.post("/stage")
    ui.post("/stop")
    for _ in range(7):  # the engine publishes its status at 1 Hz
        ui.advance(1.0)
        ui.ctl.engine_status("puppy_playground", "idle", {"state": "idle"})
    assert ui.post("/rest")[0] == 200


def test_stop_before_the_handle_exists_is_remembered(ui):
    ui.post("/stage")
    token = ui.backend.goals[-1][0]
    assert ui.ctl.stop_requested(token) is False
    ui.post("/stop")
    assert ui.ctl.stop_requested(token) is True  # ui_node._accepted cancels the handle on arrival
    ui.finish(False, "", "", cancelled=True)
    assert ui.ctl.stop_requested(token) is False


def test_payload_json_is_parsed_and_tolerates_junk():
    from puppy_playground.ui_node import parse_payload

    p = json.dumps({"action": "stage", "error_code": "check_failed", "message": "floor: 3 mm"})
    assert parse_payload(p) == ("check_failed", "floor: 3 mm")
    assert parse_payload("") == ("", "")
    assert parse_payload("not json") == ("", "not json")
    assert parse_payload("[1, 2]") == ("", "[1, 2]")
    assert parse_payload('{"message": 5}') == ("", "5")


def test_status_carries_the_action_and_a_plan_stage_result(ui):
    ui.ctl.engine_status("puppy_playground", "executing", {"state": "executing", "action": "express", "mood": "playful"})
    status = json.loads(ui.get("/status")[1])
    assert (status["action"], status["mood"]) == ("express", "playful")
    ui.post("/plan_stage")
    ui.finish()
    assert json.loads(ui.get("/status")[1])["last_done"] == {"action": "plan_stage", "result": "ok"}
    ui.post("/plan_stage")
    ui.finish(False, "no_candidate", "tried escape, min_jerk")
    status = json.loads(ui.get("/status")[1])
    assert status["last_done"]["result"] == "refused"
    assert status["last_refusal"]["error_code"] == "no_candidate"


def test_page_has_a_separate_plan_stage_button_and_the_hints(ui):
    html = ui.get("/")[1].decode()
    assert 'id="plan_stage"' in html and "Plan stage (no motion)" in html
    assert html.index("Plan stage (no motion)") > html.index('id="auto"')  # apart from the motion buttons
    for hint in ("press Recover", "check the arm, then press Recover", "another controller has the arm",
                 "release the e-stop, then turn auto on again"):
        assert hint in html
    assert "press Stage (or Rest)" not in html


def test_status_json(ui):
    ui.ctl.engine_status("puppy_playground", "executing", {"mood": "playful", "state": "executing"})
    ui.ctl.estopped(True)
    status = json.loads(ui.get("/status")[1])
    assert status["state"] == "executing"
    assert status["mood"] == "playful"
    assert status["auto"] == "off"
    assert status["estopped"] is True
    assert "last_refusal" in status


def test_page_contains_the_e_stop_sentence_and_every_expression_button(ui):
    code, body = ui.get("/")
    html = body.decode()
    assert code == 200
    assert "The hardware e-stop is the safety; Stop only cancels the current motion." in html
    for name in EXPRESSIONS:
        assert f'data-expr="{name}"' in html, name
    for word in ("Stage", "Rest", "Recover", "Stop", "Auto"):
        assert word in html
    assert "application/json" in html  # the page sends the header the server requires


def test_expression_list_matches_puppy_motion():
    pytest.importorskip("numpy")
    from puppy_motion.idle import EXPRESSIONS as REAL

    assert tuple(EXPRESSIONS) == tuple(REAL)


# -- the arm-still detector ------------------------------------------------------

NAN = float("nan")
FAKE_NAMES = ["finger_joint"] + [f"arm_joint_{i}" for i in range(1, 8)]  # gripper first, as the drivers do


def test_fakedriver_layout_with_nan_gripper_and_still_arm_is_still():
    assert arm_moving(FAKE_NAMES, [NAN] + [0.0] * 7) is False


def test_fakedriver_layout_moving_arm_is_moving_at_any_speed():
    for v in (0.02, 0.1, -0.5, 2.0):
        for joint in range(7):
            vel = [NAN] + [0.0] * 7
            vel[1 + joint] = v
            assert arm_moving(FAKE_NAMES, vel) is True, (v, joint)


def test_nonfinite_arm_velocity_or_missing_joints_count_as_moving():
    assert arm_moving(FAKE_NAMES, [0.0] + [NAN] + [0.0] * 6) is True
    assert arm_moving(FAKE_NAMES, [0.0] + [float("inf")] + [0.0] * 6) is True
    assert arm_moving(FAKE_NAMES[:-1], [0.0] * 7) is True
    assert arm_moving([], []) is True
    assert arm_moving(FAKE_NAMES, [0.0] * 3) is True  # short velocity array


def test_joint_order_does_not_matter():
    names = list(reversed(FAKE_NAMES))
    vel = [0.0] * 7 + [NAN]
    assert arm_moving(names, vel) is False


def test_a_stop_code_from_the_auto_goal_turns_auto_off_and_shows_the_reason(ui):
    start_auto(ui)
    ui.finish(False, "drifted", "the arm has drifted from the stage pose")
    status = json.loads(ui.get("/status")[1])
    assert status["auto"] == "off" and status["off_reason"] == "drifted"
    ui.advance(30)
    assert len(ui.backend.goals) == 1  # nothing more is sent
    assert ui.post("/stage")[0] == 200  # the user can act again
