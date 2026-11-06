"""Puppy Playground engine: plan_stage, stage, rest and recover move the arm only along paths the
puppy_motion rules approve; express, combo and auto are not implemented yet (Task 6).

Safety shape: plan_stage plans and writes a candidate and never moves the arm. stage and rest replay
the APPROVED file only (the operator copies the candidate over it), after the file passes the path
rules again (cached per engine start by file hash, chair-scene commit and URDF hash). rest replays
the stored forward path backwards: the same samples in reverse order with velocities re-derived, so
it visits exactly the postures the forward path was checked on. The forward-direction floor rules
do not apply to the reversed order (it ends at rest, inside the torso box), so the forward check
gates it, plus the cap and ends rules on the reversed points themselves."""

import json
import time
from importlib import resources

import numpy as np
from capability_engine.arm import IMPEDANCE, QUEUE
from capability_engine.engine import Action, CapabilityEngine, EngineError, run

from puppy_motion import stage as stage_rules
from puppy_motion.escape import escape_path, grown_scene
from puppy_motion.flatbot import CHAIR_LAYERS_COMMIT, chair_scene
from puppy_motion.track import DT, Zone
from puppy_motion.urdf import Chain
from puppy_motion.validate import Violation, check_beat

from . import stage_store

STILL_RAD_S = 0.02  # every |dq| must be below this to start a move
AT_TOL_RAD = 0.02  # "at" a path sample
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
            params_schema=_schema(
                behavior={"type": "string"},
                size=_SIZE,
                intensity={"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0},
            ),
        ),
        Action(
            "combo",
            "Play a sequence of behaviors",
            params_schema=_schema(
                behaviors={"type": "array", "items": {"type": "string"}}, size=_SIZE
            ),
        ),
        Action(
            "auto",
            "Play behaviors on its own whenever the arm is idle",
            params_schema=_schema(size=_SIZE, duration_s={"type": "number", "minimum": 0}),
        ),
    ]

    def __init__(self, engine_id="puppy_playground", **node_kwargs):
        super().__init__(engine_id, **node_kwargs)
        self.stage_path = self.declare_parameter("playground.stage_path", STAGE_PATH).value
        self.candidate_path = self.declare_parameter(
            "playground.candidate_path", CANDIDATE_PATH
        ).value
        self.log_path = self.declare_parameter("playground.log", LOG_PATH).value


        self._checked = {}  # (path sha256, chair-scene commit, urdf sha256) -> violations
        self._chains = {}  # urdf sha256 -> Chain

    # -- helpers ---------------------------------------------------------------------------------

    def _urdf(self):
        text = self.arm.robot_description()
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
        js = self.arm.joint_states()
        if np.abs(js.velocity).max() >= STILL_RAD_S:
            raise EngineError("not_still", "the arm is moving", recoverable=True)
        return np.array(js.position, dtype=float)

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
        """The StagePath in `path`, re-checked once per engine start (cached)."""
        try:
            sp = stage_store.load(path, urdf_text, stage_q)
        except stage_store.StageStoreError as e:
            raise EngineError(e.code, str(e))
        key = (sp.file_sha256, CHAIR_LAYERS_COMMIT, stage_store.urdf_hash(urdf_text))
        if key not in self._checked:
            self._checked[key] = self._path_violations(sp.points, chain, sp.margin_m)
        if self._checked[key]:
            v = self._checked[key][0]
            raise EngineError("unsafe_stage_path", f"{v.rule}: {v.detail}")
        return sp

    def _send(self, chain, points, rec):
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
        }.get(action_id)
        if handler is None:
            raise EngineError("not_yet", f"{action_id} is not implemented yet")
        rec = {"t": time.time(), "action": action_id, "params": params}
        try:
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
        urdf_text, chain = self._urdf()
        stage_q = self._stage_q(chain)
        q = self._still_q()
        ctx.checkpoint()
        plan = self.arm.plan_to_joints(stage_q, q)  # nothing moves
        points, used = None, "curobo"
        if plan.success and len(plan.t) >= 2:
            grid = np.arange(0.0, plan.t[-1] + 1e-9, DT)
            qs = np.stack([np.interp(grid, plan.t, plan.q[:, j]) for j in range(7)], axis=1)
            qs[0], qs[-1] = q, stage_q
            points = stage_rules.retime_path(stage_rules.points_from_q(qs), chain)
        violations = [] if points is None else self._path_violations(points, chain, stage_store.MARGIN_M)
        if points is None or violations:
            rec["curobo"] = plan.message if points is None else [f"{v.rule}: {v.detail}" for v in violations[:3]]
            ctx.checkpoint()
            used = "escape_path"
            try:
                points = escape_path(chain, chair_scene(), q, stage_q)
                violations = self._path_violations(points, chain, stage_store.MARGIN_M)
            except ValueError as e:
                points, violations = None, [Violation("escape", 0, str(e))]
        if points is not None:
            stage_store.save(self.candidate_path, points, urdf_text, stage_q)
            try:
                stage_store.check_endpoints(stage_store.load(self.candidate_path, urdf_text))
            except stage_store.StageStoreError as e:
                violations = list(violations) + [Violation(e.code, 0, str(e))]
        out = {
            "violations": [f"{v.rule}: {v.detail}" for v in violations],
            "candidate": self.candidate_path if points is not None else None,
            "planner": used,
        }
        rec["result"] = out
        return out

    def _stage(self, ctx, rec):
        urdf_text, chain = self._urdf()
        sp = self._load(self.stage_path, urdf_text, chain, self._stage_q(chain))
        q = self._still_q()
        if np.abs(q - sp.points[0][1]).max() > AT_TOL_RAD:
            raise EngineError("not_at_start", "the arm is not at the path's start (rest)")
        ctx.checkpoint()
        self._send(chain, sp.points, rec)
        return {"points": len(sp.points)}

    def _rest(self, ctx, rec):
        urdf_text, chain = self._urdf()
        sp = self._load(self.stage_path, urdf_text, chain, self._stage_q(chain))
        q = self._still_q()
        if np.abs(q - sp.points[-1][1]).max() > AT_TOL_RAD:
            raise EngineError("not_at_end", "the arm is not at the path's end (stage)")
        pts = stage_rules.points_from_q(np.array([p[1] for p in sp.points])[::-1])
        bad = self._caps_ends(pts, chain)
        rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
        if bad:
            raise EngineError("unsafe_stage_path", f"reversed path: {bad[0].rule}: {bad[0].detail}")
        ctx.checkpoint()
        self._send(chain, pts, rec)
        return {"points": len(pts)}

    def _recover(self, ctx, rec):
        urdf_text, chain = self._urdf()
        stage_q = self._stage_q(chain)
        q = self._still_q()
        if np.abs(q - stage_q).max() <= RECOVER_STAGE_RAD:
            seconds = max(1.0, float(np.abs(stage_q - q).max()) / 0.2)
            pts = stage_rules.retime_path(stage_rules.min_jerk_path(q, stage_q, seconds), chain)
            # check_beat expects a beat that returns to q_start; this move ends at the stage pose,
            # so only that one rule ("end") is waived, every other rule applies
            bad = [v for v in check_beat(pts, q, chain, chair_scene(), Zone.FREE, held=self._held()) if v.rule != "end"]
            rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
            if bad:
                raise EngineError("unsafe_move", f"{bad[0].rule}: {bad[0].detail}")
            ctx.checkpoint()
            self._send(chain, pts, rec)
            return {"recovered": "to_stage", "points": len(pts)}
        try:
            sp = self._load(self.stage_path, urdf_text, chain, stage_q)
        except EngineError:
            raise EngineError("recover_by_hand", "not near the stage pose and no usable stage path: recover by hand")
        qs = np.array([p[1] for p in sp.points])
        err = np.abs(qs - q).max(axis=1)
        i = int(np.argmin(err))
        if err[i] > AT_TOL_RAD:
            raise EngineError("recover_by_hand", "not near the stage pose or the stage path: recover by hand")
        if i == 0:
            return {"recovered": "already_at_rest"}
        back = qs[: i + 1][::-1].copy()
        back[0] = q
        pts = stage_rules.retime_path(stage_rules.points_from_q(back), chain)
        bad = self._caps_ends(pts, chain)
        rec["checks"] = [f"{v.rule}: {v.detail}" for v in bad]
        if bad:
            raise EngineError("unsafe_move", f"{bad[0].rule}: {bad[0].detail}")
        ctx.checkpoint()
        self._send(chain, pts, rec)
        return {"recovered": "along_path", "points": len(pts)}


def main():
    run(PuppyPlayground)


if __name__ == "__main__":
    main()
