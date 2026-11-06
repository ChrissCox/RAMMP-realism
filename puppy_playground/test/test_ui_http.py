"""The HTTP layer against a fake backend (no ROS)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from puppy_playground.supervisor import Supervisor
from puppy_playground.ui_node import EXPRESSIONS, UiController, make_server


class FakeBackend:
    def __init__(self):
        self.goals = []  # (token, action_id, params)
        self.cancels = 0
        self.requests = []
        self.accept = True

    def send_goal(self, token, action_id, params):
        self.goals.append((token, action_id, params))

    def control_cancel(self):
        self.cancels += 1

    def publish_request(self, name, size):
        self.requests.append((name, size))


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def ui():
    backend = FakeBackend()
    clock = Clock()
    sup = Supervisor(clock)
    ctl = UiController(backend, sup, clock)
    server = make_server(ctl, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class Client:
        def __init__(self):
            self.backend, self.ctl, self.sup, self.clock = backend, ctl, sup, clock

        def post(self, path, body=None):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(body or {}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def get(self, path):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                return r.status, r.read()

    yield Client()
    server.shutdown()
    server.server_close()


def test_double_post_returns_409_busy(ui):
    assert ui.post("/stage")[0] == 200
    code, body = ui.post("/stage")
    assert (code, body["error"]) == (409, "busy")
    assert ui.post("/express", {"name": "nod", "size": 0.25})[0] == 409
    assert ui.post("/rest")[0] == 409
    assert len(ui.backend.goals) == 1


def test_goal_done_frees_the_ui_and_records_a_refusal(ui):
    ui.post("/stage")
    token = ui.backend.goals[0][0]
    ui.ctl.goal_finished(token, False, "stage_missing", "no stored path")
    code, body = ui.get("/status")
    status = json.loads(body)
    assert status["busy"] is False
    assert status["last_refusal"]["error_code"] == "stage_missing"
    assert ui.post("/rest")[0] == 200


def test_a_stale_goal_done_is_ignored(ui):
    ui.post("/stage")
    ui.ctl.goal_finished(999, True, "", "")
    assert ui.post("/rest")[0] == 409


def test_goal_params(ui):
    ui.post("/express", {"name": "nod", "size": 0.5})
    assert ui.backend.goals[0][1:] == ("express", {"name": "nod", "size": 0.5})
    ui.ctl.goal_finished(ui.backend.goals[0][0], True, "", "")
    ui.post("/combo", {"names": ["nod", "wag"], "size": 0.3})
    assert ui.backend.goals[1][1:] == ("combo", {"names": ["nod", "wag"], "size": 0.3})
    ui.ctl.goal_finished(ui.backend.goals[1][0], True, "", "")
    for path in ("/stage", "/rest", "/recover"):
        ui.post(path)
        assert ui.backend.goals[-1][1:] == (path[1:], {})
        ui.ctl.goal_finished(ui.backend.goals[-1][0], True, "", "")


def test_bad_input_is_400_and_sends_nothing(ui):
    assert ui.post("/express", {"name": "nope", "size": 0.25})[0] == 400
    assert ui.post("/express", {"size": 0.25})[0] == 400
    assert ui.post("/combo", {"names": [], "size": 0.25})[0] == 400
    assert ui.post("/combo", {"names": ["nod", "x"], "size": 0.25})[0] == 400
    assert ui.post("/nothing")[0] == 404
    assert ui.backend.goals == []


def test_size_is_clamped(ui):
    ui.post("/express", {"name": "nod", "size": 9})
    assert ui.backend.goals[0][2]["size"] == 1.0
    ui.ctl.goal_finished(ui.backend.goals[0][0], True, "", "")
    ui.post("/express", {"name": "nod", "size": -1})
    assert ui.backend.goals[1][2]["size"] == 0.25


def _start_auto(ui):
    ui.sup.arm_still_for(10)
    assert ui.post("/auto", {"on": True, "size": 0.5})[0] == 200
    ui.ctl.tick()
    assert ui.backend.goals[-1][1] == "auto"
    return ui.backend.goals[-1][0]


def test_auto_goal_carries_size_and_seed(ui):
    _start_auto(ui)
    action, params = ui.backend.goals[0][1:]
    assert action == "auto" and params["size"] == 0.5 and "seed" in params


def test_express_during_auto_publishes_a_request_not_a_goal(ui):
    _start_auto(ui)
    n = len(ui.backend.goals)
    code, _ = ui.post("/express", {"name": "wag", "size": 0.4})
    assert code == 200
    assert ui.backend.requests == [("wag", 0.4)]
    assert len(ui.backend.goals) == n


def test_other_posts_during_auto_are_409(ui):
    _start_auto(ui)
    for path, body in (("/stage", {}), ("/rest", {}), ("/recover", {}), ("/combo", {"names": ["nod"], "size": 0.3})):
        assert ui.post(path, body)[0] == 409
    assert len(ui.backend.goals) == 1


def test_express_while_auto_is_waiting_is_409(ui):
    ui.sup.engine_status("task", "executing")
    ui.post("/auto", {"on": True})
    assert ui.post("/express", {"name": "nod", "size": 0.25})[0] == 409
    assert ui.backend.requests == []


def test_a_task_starting_cancels_through_the_control_service(ui):
    _start_auto(ui)
    ui.ctl.engine_status("rammp_task", "executing")
    assert ui.backend.cancels == 1
    ui.ctl.tick()
    assert ui.sup.state.value == "waiting_for_task"


def test_auto_is_not_resent_while_the_cancelled_goal_is_still_active(ui):
    token = _start_auto(ui)
    ui.ctl.engine_status("t", "executing")
    ui.ctl.engine_status("t", "idle")
    ui.clock.t += 10
    ui.ctl.tick()
    assert len(ui.backend.goals) == 1  # old goal not finished yet
    ui.ctl.goal_finished(token, True, "", "")
    ui.clock.t += 10
    ui.ctl.tick()
    assert len(ui.backend.goals) == 2


def test_stop_cancels_and_turns_auto_off(ui):
    _start_auto(ui)
    code, _ = ui.post("/stop")
    assert code == 200
    assert ui.backend.cancels >= 1
    assert ui.sup.state.value == "off"


def test_stop_is_allowed_while_busy(ui):
    ui.post("/stage")
    assert ui.post("/stop")[0] == 200
    assert ui.backend.cancels == 1


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


def test_expression_list_matches_puppy_motion():
    pytest.importorskip("numpy")
    from puppy_motion.idle import EXPRESSIONS as REAL

    assert tuple(EXPRESSIONS) == tuple(REAL)
