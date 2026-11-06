"""Temporary testing UI: HTTP page + ROS wiring + the auto-mode supervisor.

UiController and make_server are plain Python (tested with a fake backend); rclpy is imported only
inside main().
"""

import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from puppy_playground.supervisor import OWN_ID, Cancel, SendAuto, State, Supervisor

# Keep in step with puppy_motion.idle.EXPRESSIONS (a test compares them).
EXPRESSIONS = (
    "breathe", "glance", "look_around", "head_tilt", "perk", "droop", "puppy_eyes", "nod", "bounce",
    "wag", "sniff", "double_take", "search", "startle", "shake_off", "proud", "doze", "lock_on",
    "look_at_you",
)
SIZE_MIN, SIZE_MAX, SIZE_DEFAULT = 0.25, 1.0, 0.25
SIMPLE = ("stage", "rest", "recover", "plan_stage")
WEB_DIR = Path(__file__).resolve().parent / "web"


def _size(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = SIZE_DEFAULT
    return min(SIZE_MAX, max(SIZE_MIN, value))


class UiController:
    """Serializes requests: one goal at a time, and the supervisor decides when auto sends."""

    def __init__(self, backend, supervisor, clock=time.monotonic):
        self.backend = backend
        self.sup = supervisor
        self.clock = clock
        self.lock = threading.RLock()
        self._active = None  # (token, kind)
        self._token = 0
        self._engine = {}  # the playground engine's last status JSON
        self._estopped = False
        self._refusal = None

    # -- inputs from ROS ------------------------------------------------------

    def engine_status(self, engine_id, state, full=None):
        with self.lock:
            if engine_id == OWN_ID:
                self._engine = dict(full or {"state": state})
            self._run(self.sup.engine_status(engine_id, state))

    def estopped(self, value):
        with self.lock:
            self._estopped = bool(value)
            self._run(self.sup.estopped(value))

    def arm_still_for(self, seconds):
        with self.lock:
            self.sup.arm_still_for(seconds)

    def tick(self):
        with self.lock:
            cmd = self.sup.tick(self.clock())
            self._run([cmd] if cmd else [])

    def goal_finished(self, token, ok, error_code="", message=""):
        with self.lock:
            if self._active is None or self._active[0] != token:
                return  # stale
            kind = self._active[1]
            self._active = None
            if not ok:
                self._refusal = {"action": kind, "error_code": error_code, "message": message}
            if kind == "auto":
                self._run(self.sup.goal_done())

    # -- HTTP -----------------------------------------------------------------

    def status(self):
        with self.lock:
            return {
                "state": self._engine.get("state", "unknown"),
                "mood": self._engine.get("mood", ""),
                "last_refusal": self._refusal,
                "auto": self.sup.state.value,
                "busy": self._active is not None,
                "estopped": self._estopped,
            }

    def post(self, path, body):
        """-> (http_code, json_dict)"""
        with self.lock:
            name = path.strip("/")
            if name == "stop":
                self.sup.toggle(False)  # Stop also turns auto off; the cancel below covers its goal
                self.backend.control_cancel()
                return 200, {"ok": True}
            if name == "auto":
                if body.get("on"):
                    self._run(self.sup.toggle(True, size=_size(body.get("size"))))
                else:
                    self._run(self.sup.toggle(False))
                return 200, {"ok": True, "auto": self.sup.state.value}
            if name == "express":
                expr = body.get("name")
                if expr not in EXPRESSIONS:
                    return 400, {"error": "unknown expression"}
                size = _size(body.get("size"))
                if self.sup.state is State.PLAYING:
                    self.backend.publish_request(expr, size)
                    return 200, {"ok": True, "queued": True}
                return self._goal("express", {"name": expr, "size": size})
            if name == "combo":
                names = body.get("names")
                if not isinstance(names, list) or not names or any(n not in EXPRESSIONS for n in names):
                    return 400, {"error": "combo needs a non-empty list of known expressions"}
                return self._goal("combo", {"names": names, "size": _size(body.get("size"))})
            if name in SIMPLE:
                return self._goal(name, {})
            return 404, {"error": "not found"}

    # -- internals ------------------------------------------------------------

    def _goal(self, kind, params):
        if self._active is not None or self.sup.state is not State.OFF:
            return 409, {"error": "busy"}
        self._send(kind, params)
        return 200, {"ok": True}

    def _send(self, kind, params):
        self._token += 1
        self._active = (self._token, kind)
        self._refusal = None
        self.backend.send_goal(self._token, kind, params)

    def _run(self, commands):
        for cmd in commands:
            if isinstance(cmd, Cancel):
                self.backend.control_cancel()
            elif isinstance(cmd, SendAuto):
                if self._active is not None:  # a cancelled goal has not ended yet: try again later
                    self._run(self.sup.goal_done())
                else:
                    self._send("auto", {"size": cmd.size, "seed": cmd.seed})


def make_server(controller, port, web_dir=WEB_DIR, preview_dir=None, host="0.0.0.0"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, data, ctype="application/json"):
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._reply(200, (Path(web_dir) / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/status":
                self._reply(200, controller.status())
            elif path.startswith("/previews/") and preview_dir:
                f = Path(preview_dir) / os.path.basename(path)
                if f.is_file():
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    self._reply(200, f.read_bytes(), ctype)
                else:
                    self._reply(404, {"error": "not found"})
            elif path == "/previews" and preview_dir:
                names = sorted(p.name for p in Path(preview_dir).glob("*") if p.is_file())
                items = "".join(f'<li><a href="/previews/{n}">{n}</a></li>' for n in names)
                self._reply(200, f"<ul>{items}</ul>".encode(), "text/html")
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                self._reply(400, {"error": "bad json"})
                return
            code, out = controller.post(self.path.split("?")[0], body)
            self._reply(code, out)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


# -- ROS wiring ----------------------------------------------------------------

STILL_RATE = 0.02  # rad/s: every joint slower than this counts as still
JOINT_STALE_S = 0.5  # no joint_states for this long: the arm is not known to be still


def main(args=None):
    import rclpy
    from rammp_common_interfaces.msg import ControlStatus
    from rammp_orchestrator_interfaces.action import ExecuteAction
    from rammp_orchestrator_interfaces.msg import Result
    from rammp_orchestrator_interfaces.srv import Control
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    class UiNode(Node):
        def __init__(self):
            super().__init__("puppy_playground_ui")
            self.declare_parameter("playground.ui_port", 8080)
            self.declare_parameter("playground.preview_dir", "")
            self._group = ReentrantCallbackGroup()
            self._client = ActionClient(
                self, ExecuteAction, "/puppy_playground/execute_action", callback_group=self._group
            )
            self._control = self.create_client(
                Control, "/puppy_playground/control", callback_group=self._group
            )
            self._request_pub = self.create_publisher(String, "/puppy_playground/request", 10)
            self._status_subs = {}
            self._moving_at = time.monotonic()
            self._last_joint = None
            self.controller = UiController(self, Supervisor(time.monotonic))
            self.create_subscription(
                JointState, "/joint_states", self._on_joints, 10, callback_group=self._group
            )
            self.create_subscription(
                ControlStatus, "/control_status", self._on_control_status, 10, callback_group=self._group
            )
            self.create_timer(0.1, self._tick, callback_group=self._group)
            self.create_timer(2.0, self._discover, callback_group=self._group)
            preview_dir = self.get_parameter("playground.preview_dir").value or None
            self._server = make_server(
                self.controller, self.get_parameter("playground.ui_port").value, preview_dir=preview_dir
            )
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            self.get_logger().info(f"playground UI on port {self._server.server_address[1]}")

        # backend interface used by UiController (called with its lock held: must not block)
        def send_goal(self, token, action_id, params):
            if not self._client.server_is_ready():
                self.controller.goal_finished(token, False, "engine_unavailable", "playground engine is not up")
                return
            goal = ExecuteAction.Goal(
                engine_id="puppy_playground", action_id=action_id, params_json=json.dumps(params)
            )
            self._client.send_goal_async(goal).add_done_callback(lambda f, t=token: self._accepted(t, f))

        def _accepted(self, token, future):
            handle = future.result()
            if not handle.accepted:
                self.controller.goal_finished(token, False, "rejected", "goal rejected by the engine")
                return
            handle.get_result_async().add_done_callback(lambda f, t=token: self._result(t, f))

        def _result(self, token, future):
            r = future.result().result.result
            ok = r.status == Result.STATUS_SUCCESS
            self.controller.goal_finished(token, ok, r.error_code, r.payload_json)

        def control_cancel(self):
            if self._control.service_is_ready():
                self._control.call_async(Control.Request(command=Control.Request.CANCEL))

        def publish_request(self, name, size):
            self._request_pub.publish(String(data=json.dumps({"name": name, "size": size})))

        # subscriptions
        def _discover(self):
            for topic, types in self.get_topic_names_and_types():
                parts = topic.split("/")
                if len(parts) == 3 and parts[2] == "status" and "std_msgs/msg/String" in types:
                    if topic not in self._status_subs:
                        self._status_subs[topic] = self.create_subscription(
                            String,
                            topic,
                            lambda m, e=parts[1]: self._on_status(e, m),
                            10,
                            callback_group=self._group,
                        )

        def _on_status(self, engine_id, msg):
            try:
                data = json.loads(msg.data)
            except ValueError:
                return
            if isinstance(data, dict):
                self.controller.engine_status(engine_id, str(data.get("state", "")), data)

        def _on_control_status(self, msg):
            self.controller.estopped(bool(msg.estopped))

        def _on_joints(self, msg):
            now = time.monotonic()
            velocity = list(msg.velocity)
            if not velocity or max(abs(v) for v in velocity) >= STILL_RATE:
                self._moving_at = now  # no velocity data: never call the arm still
            self._last_joint = now

        def _tick(self):
            now = time.monotonic()
            if self._last_joint is None or now - self._last_joint > JOINT_STALE_S:
                self._moving_at = now
            self.controller.arm_still_for(now - self._moving_at)
            self.controller.tick()

    rclpy.init(args=args)
    node = UiNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
