"""Puppy Playground engine: plan_stage, stage, rest and recover move the arm only along paths the
puppy_motion rules approve; express, combo and auto (motion.py) play beats through the one checked pipeline (pipeline.py).

Safety shape: plan_stage plans and writes a candidate and never moves the arm. stage and rest replay
the APPROVED file only (the operator copies the candidate over it), after the file passes the path
rules again (cached per engine start by file hash, chair-scene commit and URDF hash). rest replays
the stored forward path backwards: the same samples in reverse order with velocities re-derived, so
it visits exactly the postures the forward path was checked on. The forward-direction floor rules
do not apply to the reversed order (the arm starts wherever it was parked), so the forward check
gates it, plus the cap and ends rules on the reversed points themselves."""

import json
import os
import time
from importlib import resources

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from capability_engine.arm import IMPEDANCE, QUEUE
from capability_engine.driver_state import ArmError, ControlTaken, EStopped
from capability_engine.engine import Action, CapabilityEngine, EngineError, run
from capability_engine.sim import SimModelError

from puppy_motion import stage as stage_rules
from puppy_motion.escape import escape_path, grown_scene
from puppy_motion.flatbot import CHAIR_LAYERS_COMMIT, chair_scene
from puppy_motion.geometry import PARTS, clearances_batch
from puppy_motion.joints import align_turns, shortest_delta
from puppy_motion.track import DT, Zone
from puppy_motion.urdf import Chain
from puppy_motion.validate import check_beat

from . import motion, pipeline, stage_store

STILL_RAD_S = 0.02  # every |dq| must be below this to start a move
AT_TOL_RAD = 0.02  # "at" a path sample
MOVED_RAD = 1e-3  # recover: the reading may change this little between the check and the send
# plan_stage start gate, on the MARGIN_M-grown scene (every stage-path floor is relative to the first
# sample, so a start that is already deep in a box would make them meaningless)
START_PERSON_MIN_M = {"upper_arm": -0.010, "forearm": 0.010, "wrist": 0.010, "head": 0.010}
START_OBSTACLE_MIN_M = 0.010
RECOVER_STAGE_RAD = 0.10  # recover: this close to the stage pose, min-jerk to it
STAGE_PATH = "/data/stage_path.json"
CANDIDATE_PATH = "/data/stage_path.candidate.json"
LOG_PATH = "/data/playground.jsonl"
BLEND_S = 1.0  # stage / rest: the stored path is blended onto the reading over this long
# The only non-zero value `stage_floor_cap_m` may take (attended tests with nobody in the chair): every
# part's person-clearance floor on a stage path is min(its first sample, this) instead of its first sample.
ATTENDED_FLOOR_CAP_M = stage_store.ATTENDED_FLOOR_CAP_M  # 0.020

_SIZE = {"type": "number", "minimum": 0.25, "maximum": 1.0, "default": 0.25}


def _schema(**properties):
    return {"type": "object", "properties": properties}


class PuppyPlayground(CapabilityEngine):
    display_name = "Puppy Playground"
    required_resources = ["arm"]
    actions = [
        Action("plan_stage", "Plan the stage path (no motion)", params_schema=_schema()),
        Action("stage", "Move from rest to the stage pose", params_schema=_schema()),
        Action("rest", "Move from the stage pose back to rest", params_schema=_schema()),
        Action("recover", "Return to rest from near the stage path", params_schema=_schema()),
        Action(
            "express",
            "Play one expressive behavior",
            params_schema=_schema(name={"type": "string"}, size=_SIZE),
        ),
        Action(
            "combo",
            "Play a sequence of behaviors",
            params_schema=_schema(names={"type": "array", "items": {"type": "string"}}, size=_SIZE),
        ),
        Action(
            "auto",
            "Play behaviors on its own whenever the arm is idle",
            params_schema=_schema(size=_SIZE, seed={"type": "integer", "default": 1}),
        ),
    ]

    def __init__(self, engine_id="puppy_playground", **node_kwargs):
        super().__init__(engine_id, **node_kwargs)
        self.stage_path = self.declare_parameter("playground.stage_path", STAGE_PATH).value
        self.candidate_path = self.declare_parameter(
            "playground.candidate_path", CANDIDATE_PATH
        ).value
        self.log_path = self.declare_parameter("playground.log", LOG_PATH).value
        # a plan-only instance (the plan_stage run on the live ROS domain) refuses every other action
        # read-only: both are read once here, so a runtime `ros2 param set` must not appear to change them
        read_only = ParameterDescriptor(read_only=True)
        self.plan_only = bool(self.declare_parameter("plan_only", False, read_only).value)
        # 0.0 (off, the default) or ATTENDED_FLOOR_CAP_M; any other value refuses every action (execute)
        self.stage_floor_cap_m = float(self.declare_parameter("playground.stage_floor_cap_m", 0.0, read_only).value)
        if self.stage_floor_cap_m == ATTENDED_FLOOR_CAP_M:
            self.get_logger().warning(
                f"ATTENDED FLOOR CAP {ATTENDED_FLOOR_CAP_M * 1000:.0f} mm is ON: stage-path person floors are "
                "min(first sample, cap). For attended tests with NOBODY in the chair only."
            )
        elif self.stage_floor_cap_m == 0.0:
            self.get_logger().info("floor cap off")
        else:
            self.get_logger().error(f"stage_floor_cap_m {self.stage_floor_cap_m!r} is invalid: every action will refuse (bad_config)")

        self._checked = {}  # (path sha256, chair-scene commit, urdf sha256) -> violations
        self._chains = {}  # urdf sha256 -> Chain
        self._failure = None  # (action, error code, message) of the action that last failed

    # -- helpers ---------------------------------------------------------------------------------

    def _urdf(self):
        try:
            text = self.arm.robot_description()
        except SimModelError as e:
            raise EngineError("no_model", f"the arm has no model: {e}")
        sha = stage_store.urdf_hash(text)
        if sha not in self._chains:
            try:
                self._chains[sha] = Chain.from_urdf(text)
            except Exception as e:  # noqa: BLE001 -- any parse failure refuses the action
                raise EngineError("bad_urdf", f"cannot read the driver's URDF: {e}")
        return text, self._chains[sha]

    @staticmethod
    def _stage_q(chain):
        text = resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text(encoding="utf-8")
        q = np.array(json.loads(text)["q"], dtype=float)
        bad = stage_rules.stage_violations(chain, chair_scene(), q)
        if bad:
            raise EngineError("bad_stage_pose", f"stage_pose.json breaks the stage rules for this URDF: {bad[0].detail}")
        return q

    def _still_q(self):
        """The measured joints, refusing unless every one is finite and every |dq| < STILL_RAD_S."""
        try:
            js = self.arm.joint_states()
        except (EStopped, ControlTaken) as e:
            raise pipeline._ended(e)
        except ArmError as e:
            raise EngineError("no_state", f"cannot read the arm: {e}", recoverable=True)
        pos, vel = np.asarray(js.position, dtype=float), np.asarray(js.velocity, dtype=float)
        if pos.shape != (7,) or vel.shape != (7,) or not np.isfinite(pos).all():
            raise EngineError("bad_state", "the joint reading is malformed", recoverable=True)
        if not (np.abs(vel).max() < STILL_RAD_S):  # a NaN velocity refuses too
            raise EngineError("not_still", "the arm is moving", recoverable=True)
        return pos

    @staticmethod
    def _near(chain, q, target, tol):
        """Whether q is within tol rad of target on every joint, modulo whole turns of the
        continuous joints (the driver reports those wrapped). NaN is never near."""
        return bool(np.all(np.abs(shortest_delta(chain, q, target)) <= tol))

    def _unmoved(self, chain, q):
        """Re-read the arm just before sending: refuse unless it is within MOVED_RAD of `q`."""
        q2 = self._still_q()
        if not self._near(chain, q, q2, MOVED_RAD):
            raise EngineError("moved", "the arm moved while the move was being checked", recoverable=True)

    def _held(self):
        try:
            pos = self.arm.gripper_state().position
        except Exception:  # noqa: BLE001 -- no fresh gripper state counts as held
            return True
        return not np.isfinite(pos) or 0.05 < pos < 0.93

    def _path_violations(self, points, chain, margin, person_floor_ref=None, obstacle_floor_ref=None):
        """The stage-path rules on `points` against the chair scene grown by `margin` metres.
        `person_floor_ref`: see puppy_motion.stage.path_violations (default: the first sample).
        `obstacle_floor_ref`: likewise for the obstacle clearances.
        With `stage_floor_cap_m` on, a missing `person_floor_ref` is min(the first sample's person
        clearances, the cap); off, nothing changes."""
        cap = getattr(self, "stage_floor_cap_m", 0.0)
        if cap and person_floor_ref is None:
            try:
                first = PuppyPlayground._clearances(chain, points[0][1], margin)[0]
                person_floor_ref = np.minimum(first, cap)
            except Exception:  # noqa: BLE001 -- a malformed path: path_violations reports it
                person_floor_ref = None
        return stage_rules.path_violations(
            points, chain, grown_scene(chair_scene(), margin),
            person_floor_ref=person_floor_ref, obstacle_floor_ref=obstacle_floor_ref,
        )

    @staticmethod
    def _clearances(chain, q, margin):
        """(person, obstacle): the four per-part clearances at q (against the scene grown by `margin`)."""
        scene = grown_scene(chair_scene(), margin)
        c = clearances_batch(chain, scene, np.asarray(q, dtype=float)[None, :], False)[0]
        return c[0], c[1]

    @classmethod
    def _person_clearances(cls, chain, q, margin):
        return cls._clearances(chain, q, margin)[0]

    @classmethod
    def _obstacle_clearances(cls, chain, q, margin):
        return cls._clearances(chain, q, margin)[1]

    @classmethod
    def _bad_start(cls, chain, q):
        """Why q cannot be a plan_stage start (None when it can): outside the URDF limits or the joint
        guards, or too close to the occupant boxes or an obstacle (START_PERSON_MIN_M,
        START_OBSTACLE_MIN_M, on the grown scene)."""
        lo, hi = chain.pos_limits[:, 0], chain.pos_limits[:, 1]
        if np.any(q < lo) or np.any(q > hi):
            j = int(np.argmax(np.maximum(lo - q, q - hi)))
            return f"joint {j + 1} is outside its URDF range"
        over = np.abs(q) - stage_rules.guard_limits(chain)
        if np.any(over > 1e-9):
            return f"joint {int(np.argmax(over)) + 1} is past its guard"
        person, obstacle = cls._clearances(chain, q, stage_store.MARGIN_M)
        if not (np.all(np.isfinite(person)) and np.all(np.isfinite(obstacle))):
            return "the clearance is not finite"
        for i, part in enumerate(PARTS):
            if person[i] < START_PERSON_MIN_M[part]:
                return f"the {part} is {person[i] * 1000:.0f} mm from the chair occupant (needs {START_PERSON_MIN_M[part] * 1000:.0f} mm)"
            if obstacle[i] < START_OBSTACLE_MIN_M:
                return f"the {part} is {obstacle[i] * 1000:.0f} mm from an obstacle (needs {START_OBSTACLE_MIN_M * 1000:.0f} mm)"
        return None

    def _load(self, path, urdf_text, chain, stage_q):
        """The StagePath in `path`, re-checked once per engine start (cached by file hash)."""
        try:
            sp = stage_store.load(path, urdf_text, stage_q, chain, floor_cap_m=self.stage_floor_cap_m)
        except stage_store.StageStoreError as e:
            raise EngineError(e.code, str(e))
        key = (sp.file_sha256, CHAIR_LAYERS_COMMIT, stage_store.urdf_hash(urdf_text))
        if key not in self._checked:
            # exactly the margin candidates are written with (load refuses any other value)
            self._checked[key] = self._path_violations(sp.points, chain, stage_store.MARGIN_M)
        if self._checked[key]:
            v = self._checked[key][0]
            raise EngineError("unsafe_stage_path", f"{v.rule}: {v.detail}")
        return sp

    def _send(self, chain, points, q_measured, rec):
        """Send `points` turned (whole turns of the continuous joints, the same for every point) to
        the measured reading; clearances are 2*pi-periodic and those joints have no limits, so the
        checked path is unchanged. A refused goal, an e-stop or a driver abort raises the same
        EngineErrors a beat does (pipeline.send_goal / raise_for_result)."""
        points = align_turns(chain, q_measured, points)
        rec["goal"] = {"points": len(points), "seconds": points[-1][0], "control_mode": "IMPEDANCE", "gains": "default"}
        res = pipeline.send_goal(self.arm, chain, points, rec, "move")
        pipeline.raise_for_result(self.arm, res, "move")

    @staticmethod
    def _blend(chain, points, q):
        """The stored path with its start moved onto the reading `q`: sent(t) = stored(t) + w(t) * delta,
        delta = the wrap-aware shortest step stored[0] -> q, w a min-jerk weight from 1 at t = 0 to 0 at
        BLEND_S (or the path's end if shorter) and 0 after. The last point is exactly the stored one, so
        offsets never accumulate over Stage / Rest cycles. Velocities are the central differences of
        the blended samples (the stored ones are, too, plus the w'(t) * delta term)."""
        stored = np.array([p[1] for p in points], dtype=float)
        t = np.array([p[0] for p in points], dtype=float)
        delta = shortest_delta(chain, stored[0], q)
        blend_s = min(BLEND_S, float(t[-1]))
        x = np.clip(t / blend_s, 0.0, 1.0)
        w = 1.0 - (10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5)
        out = stored + w[:, None] * delta
        out[0] = stored[0] + delta
        out[-1] = stored[-1]
        return stage_rules.points_from_q(out)

    def _seat_and_check(self, chain, points, q, full_rules, rec):
        """Blend `points` onto the reading `q`, turn them to it, check exactly what will be sent
        (the stage-path rules at MARGIN_M, or only caps and ends for a reversed path), then refuse
        `moved` if the arm has since moved. Returns the points to send."""
        blended = align_turns(chain, q, self._blend(chain, points, q))
        if full_rules:
            # the person floor is anchored at the WORSE of where the arm is and where the approved
            # path starts: the blend may come no closer to the person than either
            # (and the obstacle floor likewise)
            p_now, o_now = self._clearances(chain, q, stage_store.MARGIN_M)
            p_start, o_start = self._clearances(chain, points[0][1], stage_store.MARGIN_M)
            ref, oref = np.minimum(p_now, p_start), np.minimum(o_now, o_start)
            if self.stage_floor_cap_m:
                ref = np.minimum(ref, self.stage_floor_cap_m)
            rec["person_floor_ref"] = [float(x) for x in ref]
            rec["obstacle_floor_ref"] = [float(x) for x in oref]
            bad = self._path_violations(
                blended, chain, stage_store.MARGIN_M, person_floor_ref=ref, obstacle_floor_ref=oref
            )
        else:
            bad = self._caps_ends(blended, chain)
        rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
        if bad:
            raise EngineError("unsafe_move", f"the move from the reading breaks {bad[0].rule}: {bad[0].detail}")
        self._unmoved(chain, q)
        return blended

    def _log(self, rec):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        except OSError as e:
            self.get_logger().warning(f"cannot write {self.log_path}: {e}")

    @staticmethod
    def _caps_ends(points, chain):
        """Cap and ends violations of `points` (the only rules that apply to a replay in reverse)."""
        bad = stage_rules.path_violations(points, chain, chair_scene())
        return [v for v in bad if v.rule in ("path_caps", "path_ends")]

    # -- actions ---------------------------------------------------------------------------------

    def execute(self, action_id, params, ctx):
        self._failure = None
        handler = {
            "plan_stage": self._plan_stage, "stage": self._stage, "rest": self._rest, "recover": self._recover,
            "express": lambda c, r: motion.express(self, c, r),
            "combo": lambda c, r: motion.combo(self, c, r),
            "auto": lambda c, r: motion.auto(self, c, r),
        }.get(action_id)
        if handler is None:
            self._failure = (action_id, "not_yet", f"{action_id} is not implemented yet")
            raise EngineError("not_yet", f"{action_id} is not implemented yet")
        rec = {"t": time.time(), "action": action_id, "params": params}
        try:
            if self.stage_floor_cap_m not in (0.0, ATTENDED_FLOOR_CAP_M):  # a NaN is in neither
                why = (f"playground.stage_floor_cap_m is {self.stage_floor_cap_m!r}; it must be exactly 0.0 (off) "
                       f"or {ATTENDED_FLOOR_CAP_M} (attended floor cap)")
                self.get_logger().error(why)
                raise EngineError("bad_config", why)
            if self.plan_only and action_id != "plan_stage":
                raise EngineError("plan_only", "this instance is plan-only", recoverable=True)
            out = handler(ctx, rec)
            rec["ok"] = True
            return out
        except BaseException as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            if isinstance(e, EngineError):
                self._failure = (action_id, e.error_code, str(e))
            else:
                self._failure = (action_id, "exception", rec["error"])
            raise
        finally:
            self._log(rec)

    def _on_execute(self, goal_handle):
        """capability_engine sets payload_json only on success. Every failed or refused action also
        carries {"action", "error_code", "message"} so the page can show the rule that broke."""
        self._failure = None
        out = super()._on_execute(goal_handle)
        result = out.result
        if result.status == result.STATUS_FAILED:
            action = goal_handle.request.action_id
            failure = self._failure
            message = failure[2] if failure and failure[1] == result.error_code else result.error_code
            result.payload_json = json.dumps(
                {"action": action, "error_code": result.error_code, "message": message}
            )
        return out

    def _plan_stage(self, ctx, rec):
        if os.path.abspath(self.candidate_path) == os.path.abspath(self.stage_path):
            raise EngineError("bad_config", "playground.candidate_path must differ from playground.stage_path")
        ctx.checkpoint()
        urdf_text, chain = self._urdf()
        stage_q = self._stage_q(chain)
        q = self._still_q()
        why = self._bad_start(chain, q)
        if why:
            raise EngineError("bad_start", f"plan_stage cannot plan from where the arm is: {why}", recoverable=True)
        target = q + shortest_delta(chain, q, stage_q)  # the stage pose on the turn nearest the reading
        tried, found = {}, None
        for method in ("curobo", "min_jerk", "escape", "via"):
            ctx.checkpoint()
            try:
                points = self._candidate_points(method, chain, q, target, rec, ctx)
            except ValueError as e:
                tried[method] = [f"{type(e).__name__}: {e}"]
                continue
            if points is None:
                tried[method] = ["no plan"]
                continue
            bad = [f"{v.rule}: {v.detail}" for v in self._path_violations(points, chain, stage_store.MARGIN_M)]
            if not bad:
                stub = stage_store.StagePath(points, points[0][1], target, "", 0.0, "")
                try:
                    stage_store.check_endpoints(stub, chain, stage_q)
                except stage_store.StageStoreError as e:
                    bad = [f"{e.code}: {e}"]
            tried[method] = bad
            if not bad:
                found = (method, points)
                break
        if found is None:
            try:
                os.remove(self.candidate_path)  # a stale candidate must not look like this plan's
            except FileNotFoundError:
                pass
            rec["tried"] = tried
            why = "; ".join(f"{m}: {', '.join(vs) or 'rejected'}" for m, vs in tried.items())
            raise EngineError("no_candidate", f"no planner produced a passing stage path ({why})", recoverable=True)
        else:
            method, points = found
            stage_store.save(
                self.candidate_path, points, urdf_text, stage_q, method=method,
                person_floor_cap_m=self.stage_floor_cap_m,
            )
            out = {"violations": [], "candidate": self.candidate_path, "method": method}
        rec["tried"] = tried
        rec["result"] = out
        return out

    def _candidate_points(self, method, chain, q, target, rec, ctx=None):
        """One planner's retimed points from q to target (None when it has no plan)."""
        if method == "curobo":
            plan = self.arm.plan_to_joints(target, q)  # nothing moves
            if not (plan.success and len(plan.t) >= 2):
                rec["curobo"] = plan.message
                return None
            grid = np.arange(0.0, plan.t[-1] + 1e-9, DT)
            qs = np.stack([np.interp(grid, plan.t, plan.q[:, j]) for j in range(7)], axis=1)
            qs[0], qs[-1] = q, target
            return stage_rules.retime_path(stage_rules.points_from_q(qs), chain)
        if method == "min_jerk":
            seconds = max(2.0, float(np.abs(target - q).max()) / 0.2)
            return stage_rules.retime_path(stage_rules.min_jerk_path(q, target, seconds), chain)
        if method == "via":
            return self._via_points(chain, q, target, ctx, self.stage_floor_cap_m)
        return escape_path(chain, chair_scene(), q, target)

    VIA_TRIES = 600

    @staticmethod
    def _via_points(chain, q, target, ctx=None, floor_cap_m=0.0):
        """Two straight joint-space legs q -> w -> target through a seeded random waypoint w, for starts
        (a parked arm) where neither the straight line nor escape_path passes. A waypoint is kept only
        if both legs, sampled finely, keep the person floor, the 0.35 m rule and the obstacle floor
        against the scene grown by MARGIN_M; the full path rules are checked by the caller. Checks
        `ctx` (Stop / Cancel) every few tries. Returns None when no waypoint is found."""
        scene = grown_scene(chair_scene(), stage_store.MARGIN_M)
        lim = stage_rules.guard_limits(chain)
        far = stage_rules._FAR_PARTS
        rng = np.random.default_rng(0)
        for i in range(PuppyPlayground.VIA_TRIES):
            if ctx is not None and i % 5 == 0:
                ctx.checkpoint()
            w = q + rng.uniform(0.2, 0.8) * (target - q) + rng.normal(0.0, 0.35, 7) * (rng.random(7) < 0.5)
            qs = np.vstack([q[None, :], np.linspace(q, w, 60)[1:], np.linspace(w, target, 60)[1:]])
            if np.any(np.abs(qs) > lim):
                continue
            c = clearances_batch(chain, scene, qs, False)
            person, obstacle = c[:, 0, :], c[:, 1, :]
            floor = person[0] if not floor_cap_m else np.minimum(person[0], floor_cap_m)
            if np.isnan(c).any() or np.any(person < floor - 1e-9):
                continue
            f = person[:, far]
            reached = np.maximum.accumulate(f >= stage_rules.PERSON_STAGE_M, axis=0)
            if np.any(np.where(reached, stage_rules.PERSON_STAGE_M - f, 0.0) > 1e-9):
                continue
            if np.any(obstacle < np.minimum(obstacle[0], stage_rules.OBSTACLE_KEEP_M) - stage_rules.OBSTACLE_TOL_M):
                continue
            dense = np.vstack([np.linspace(q, w, 200), np.linspace(w, target, 200)[1:]])
            return stage_rules.retime_path(stage_rules.points_from_q(dense), chain)
        return None

    def _stage(self, ctx, rec):
        urdf_text, chain = self._urdf()
        sp = self._load(self.stage_path, urdf_text, chain, self._stage_q(chain))
        # slow, reading-independent work is done: a pause blocks here, then the reading is taken
        ctx.checkpoint()
        q = self._still_q()
        if not self._near(chain, q, sp.points[0][1], AT_TOL_RAD):
            raise EngineError("not_at_start", "the arm is not at the path's start (rest)")
        points = self._seat_and_check(chain, sp.points, q, True, rec)
        self._send(chain, points, q, rec)
        return {"points": len(points)}

    def _rest(self, ctx, rec):
        urdf_text, chain = self._urdf()
        sp = self._load(self.stage_path, urdf_text, chain, self._stage_q(chain))
        pts = stage_rules.points_from_q(np.array([p[1] for p in sp.points])[::-1])
        bad = self._caps_ends(pts, chain)
        rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
        if bad:
            raise EngineError("unsafe_stage_path", f"reversed path: {bad[0].rule}: {bad[0].detail}")
        ctx.checkpoint()
        q = self._still_q()
        if not self._near(chain, q, sp.points[-1][1], AT_TOL_RAD):
            raise EngineError("not_at_end", "the arm is not at the path's end (stage)")
        pts = self._seat_and_check(chain, pts, q, False, rec)
        self._send(chain, pts, q, rec)
        return {"points": len(pts)}

    def _recover(self, ctx, rec):
        urdf_text, chain = self._urdf()
        stage_q = self._stage_q(chain)
        try:  # only needed off the stage pose; loaded (and re-checked) before measuring
            sp = self._load(self.stage_path, urdf_text, chain, stage_q)
        except EngineError:
            sp = None
        ctx.checkpoint()
        q = self._still_q()
        if self._near(chain, q, stage_q, RECOVER_STAGE_RAD):
            target = q + shortest_delta(chain, q, stage_q)
            seconds = max(1.0, float(np.abs(target - q).max()) / 0.2)
            pts = stage_rules.retime_path(stage_rules.min_jerk_path(q, target, seconds), chain)
            # check_beat expects a beat that returns to q_start; this move ends at the stage pose,
            # so only that one rule ("end") is waived, every other rule applies
            bad = [v for v in check_beat(pts, q, chain, chair_scene(), Zone.FREE, held=self._held()) if v.rule != "end"]
            rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
            if bad:
                raise EngineError("unsafe_move", f"{bad[0].rule}: {bad[0].detail}")
            self._unmoved(chain, q)
            self._send(chain, pts, q, rec)
            return {"recovered": "to_stage", "points": len(pts)}
        if sp is None:
            raise EngineError("recover_by_hand", "not near the stage pose and no usable stage path: recover by hand")
        qs = np.array([p[1] for p in sp.points])
        err = np.array([np.abs(shortest_delta(chain, q, x)).max() for x in qs])
        i = int(np.argmin(err))
        if not (err[i] <= AT_TOL_RAD):
            raise EngineError("recover_by_hand", "not near the stage pose or the stage path: recover by hand")
        if i == 0:
            return {"recovered": "already_at_rest"}
        back = align_turns(chain, q, stage_rules.points_from_q(qs[: i + 1][::-1]))
        arr = np.array([p[1] for p in back])
        arr[0] = q
        pts = stage_rules.retime_path(stage_rules.points_from_q(arr), chain)
        bad = self._caps_ends(pts, chain)
        rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
        if bad:
            raise EngineError("unsafe_move", f"{bad[0].rule}: {bad[0].detail}")
        self._unmoved(chain, q)
        self._send(chain, pts, q, rec)
        return {"recovered": "along_path", "points": len(pts)}


def main():
    run(PuppyPlayground)


if __name__ == "__main__":
    main()
