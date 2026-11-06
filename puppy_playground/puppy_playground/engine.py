"""Puppy Playground engine: plan_stage, stage, rest and recover move the arm only along paths the
puppy_motion rules approve; express, combo and auto (motion.py) play beats through the one checked pipeline (pipeline.py).

Safety shape: plan_stage plans and writes a candidate and never moves the arm. stage and rest replay
the APPROVED file only (the operator copies the candidate over it), after the file passes the path
rules again (cached per engine start by file hash, chair-scene commit and URDF hash). rest replays
the stored forward path backwards: the same samples in reverse order with velocities re-derived, so
it visits exactly the postures the forward path was checked on. The forward-direction floor rules
do not apply to the reversed order (it ends at rest, inside the torso box), so the forward check
gates it, plus the cap and ends rules on the reversed points themselves."""

import json
import os
import time
from importlib import resources

import numpy as np
from capability_engine.arm import IMPEDANCE, QUEUE
from capability_engine.engine import Action, CapabilityEngine, EngineError, run
from capability_engine.sim import SimModelError

from puppy_motion import stage as stage_rules
from puppy_motion.escape import escape_path, grown_scene
from puppy_motion.flatbot import CHAIR_LAYERS_COMMIT, chair_scene
from puppy_motion.joints import align_turns, shortest_delta
from puppy_motion.stage import REST_Q
from puppy_motion.track import DT, Zone
from puppy_motion.urdf import Chain
from puppy_motion.validate import check_beat

from . import motion, stage_store

STILL_RAD_S = 0.02  # every |dq| must be below this to start a move
AT_TOL_RAD = 0.02  # "at" a path sample
MOVED_RAD = 1e-3  # recover: the reading may change this little between the check and the send
RECOVER_STAGE_RAD = 0.10  # recover: this close to the stage pose, min-jerk to it
STAGE_PATH = "/data/stage_path.json"
CANDIDATE_PATH = "/data/stage_path.candidate.json"
LOG_PATH = "/data/playground.jsonl"

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
        self.plan_only = bool(self.declare_parameter("plan_only", False).value)


        self._checked = {}  # (path sha256, chair-scene commit, urdf sha256) -> violations
        self._chains = {}  # urdf sha256 -> Chain

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
        js = self.arm.joint_states()
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
        return 0.05 < pos < 0.93

    def _path_violations(self, points, chain, margin):
        """The stage-path rules on `points` against the chair scene grown by `margin` metres."""
        return stage_rules.path_violations(points, chain, grown_scene(chair_scene(), margin))

    def _load(self, path, urdf_text, chain, stage_q):
        """The StagePath in `path`, re-checked once per engine start (cached by file hash)."""
        try:
            sp = stage_store.load(path, urdf_text, stage_q, chain)
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
        checked path is unchanged."""
        points = align_turns(chain, q_measured, points)
        rec["goal"] = {"points": len(points), "seconds": points[-1][0], "control_mode": "IMPEDANCE", "gains": "default"}
        res = self.arm.execute_joint_trajectory(
            chain.joint_names, points, control_mode=IMPEDANCE, gains="default", preemption=QUEUE
        )
        rec["driver"] = {"error_code": int(res.error_code), "error_string": res.error_string}
        if res.error_code != 0:
            raise EngineError("driver_error", f"the driver ended the move: {res.error_string}", recoverable=True)

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
        handler = {
            "plan_stage": self._plan_stage, "stage": self._stage, "rest": self._rest, "recover": self._recover,
            "express": lambda c, r: motion.express(self, c, r),
            "combo": lambda c, r: motion.combo(self, c, r),
            "auto": lambda c, r: motion.auto(self, c, r),
        }.get(action_id)
        if handler is None:
            raise EngineError("not_yet", f"{action_id} is not implemented yet")
        rec = {"t": time.time(), "action": action_id, "params": params}
        try:
            if self.plan_only and action_id != "plan_stage":
                raise EngineError("plan_only", "this instance is plan-only", recoverable=True)
            out = handler(ctx, rec)
            rec["ok"] = True
            return out
        except BaseException as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._log(rec)

    def _plan_stage(self, ctx, rec):
        if os.path.abspath(self.candidate_path) == os.path.abspath(self.stage_path):
            raise EngineError("bad_config", "playground.candidate_path must differ from playground.stage_path")
        ctx.checkpoint()
        urdf_text, chain = self._urdf()
        stage_q = self._stage_q(chain)
        q = self._still_q()
        if not self._near(chain, q, REST_Q, AT_TOL_RAD):
            raise EngineError("not_at_rest", "plan_stage plans from the rest pose; the arm is not there", recoverable=True)
        target = q + shortest_delta(chain, q, stage_q)  # the stage pose on the turn nearest the reading
        tried, found = {}, None
        for method in ("curobo", "min_jerk", "escape"):
            ctx.checkpoint()
            try:
                points = self._candidate_points(method, chain, q, target, rec)
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
            out = {"violations": [f"{m}: {v}" for m, vs in tried.items() for v in vs], "candidate": None, "method": None}
        else:
            method, points = found
            stage_store.save(self.candidate_path, points, urdf_text, stage_q, method=method)
            out = {"violations": [], "candidate": self.candidate_path, "method": method}
        rec["tried"] = tried
        rec["result"] = out
        return out

    def _candidate_points(self, method, chain, q, target, rec):
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
        return escape_path(chain, chair_scene(), q, target)

    def _stage(self, ctx, rec):
        urdf_text, chain = self._urdf()
        sp = self._load(self.stage_path, urdf_text, chain, self._stage_q(chain))
        # slow, reading-independent work is done: a pause blocks here, then the reading is taken
        ctx.checkpoint()
        q = self._still_q()
        if not self._near(chain, q, sp.points[0][1], AT_TOL_RAD):
            raise EngineError("not_at_start", "the arm is not at the path's start (rest)")
        self._send(chain, sp.points, q, rec)
        return {"points": len(sp.points)}

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
