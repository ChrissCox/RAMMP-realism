"""One checked motion: the single path every playground beat takes to the arm.

measure q -> require stillness -> build the beat -> measure again -> require the beat to start where
the arm is -> check_beat(...) == [] -> one IMPEDANCE / "default" / QUEUE joint-trajectory goal.
Nothing else is ever sent (no gripper goal). EStopped and ControlTaken end the action; any other
refusal is a recoverable failure the caller may log and carry on from."""

import time

import numpy as np
from capability_engine.arm import HALTED, IMPEDANCE, QUEUE
from capability_engine.driver_state import ArmError, ControlTaken, EStopped
from capability_engine.engine import EngineError

from rammp_arm import ERRORS

from puppy_motion.joints import align_turns, shortest_delta
from puppy_motion.validate import check_beat

from . import playlog

STILL_RAD_S = 0.02  # every |dq| must be below this to start a beat
SHIFT_TOL_RAD = 0.02  # a beat starting this close to the measured posture is shifted onto it
ANCHOR_TOL_RAD = 0.02  # auto beats start within this of a named pose (stage, or the stored rest)
HALT_STATUS_WAIT_S = 0.5  # after a HALTED goal, how long /control_status may take to show an e-stop


def _measure(arm):
    """(q, max |dq|) of the arm now. EStopped / ControlTaken end the action, other errors skip."""
    try:
        js = arm.joint_states()
    except (EStopped, ControlTaken) as e:
        raise _ended(e)
    except ArmError as e:
        raise EngineError("no_state", f"cannot read the arm: {e}", recoverable=True)
    try:
        q = np.array(js.position, dtype=float)
        v = np.array(js.velocity, dtype=float)
    except (TypeError, ValueError) as e:
        raise EngineError("bad_state", f"unreadable joint state: {e}", recoverable=True)
    if q.shape != (7,) or v.shape != (7,) or not np.isfinite(q).all():
        raise EngineError("bad_state", "the joint state is not seven finite positions", recoverable=True)
    # a NaN velocity makes the speed NaN, which fails every "still" comparison below (closed)
    return q, float(np.abs(v).max())


def _ended(e):
    if isinstance(e, EStopped):
        return EngineError("estopped", str(e))
    return EngineError("control_taken", str(e))


def play(arm, ctx, chain, scene, beat_builder, held, log=None, name=None, anchors=None):
    """Play one beat; returns the log record. `beat_builder(q, held)` gives the Beat (or None when
    there is nothing to play) for the measured posture q. On a failed check nothing is sent and
    EngineError("check_failed", <rules>, recoverable=True) is raised. `log(rec)` is called with the
    record however this ends. `anchors`: poses (7-vectors); when given, the final measured posture must
    be within ANCHOR_TOL_RAD of one of them or EngineError("drifted") ends the action."""
    rec = playlog.motion_record(name, held)
    try:
        return _play(arm, ctx, chain, scene, beat_builder, bool(held), rec, anchors)
    except BaseException as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if log is not None:
            log(rec)


def _play(arm, ctx, chain, scene, beat_builder, held, rec, anchors):
    q, speed = _measure(arm)
    if not (speed < STILL_RAD_S):
        raise EngineError("not_still", "the arm is moving", recoverable=True)
    ctx.checkpoint()
    beat = beat_builder(q, held)
    if beat is None:
        raise EngineError("no_beat", "no beat to play here", recoverable=True)
    rec["beat"] = {"name": beat.name, "mood": beat.mood, "zone": str(beat.zone)}
    ctx.checkpoint()
    points = beat.points
    # the posture the beat is checked against and sent from is the one measured right now
    q, speed = _measure(arm)
    if not (speed < STILL_RAD_S):
        raise EngineError("not_still", "the arm is moving", recoverable=True)
    if anchors is not None and not _anchored(chain, q, anchors):
        raise EngineError(
            "drifted",
            "the arm has drifted from the stage pose; press Stage (or Rest) to re-seat it",
        )
    gap = float(np.abs(shortest_delta(chain, q, points[0][1])).max())
    if not (gap <= SHIFT_TOL_RAD):
        raise EngineError(
            "start_mismatch", f"the arm is {gap:.4f} rad from where the beat starts", recoverable=True
        )
    points = align_turns(chain, q, points)  # the driver reports continuous joints wrapped
    offset = q - np.asarray(points[0][1], dtype=float)  # under SHIFT_TOL_RAD on every joint
    points = [(t, np.asarray(pq, dtype=float) + offset, dq) for t, pq, dq in points]
    points[0] = (points[0][0], q.copy(), points[0][2])  # exactly the measured posture
    rec["shift_rad"] = float(gap)
    found = check_beat(points, q, chain, scene, beat.zone, held=held)
    rec["checks"] = [f"{v.rule}: {v.detail}" for v in found]
    if found:
        raise EngineError("check_failed", "; ".join(rec["checks"]), recoverable=True)
    rec["goal"] = {
        "points": len(points), "seconds": float(points[-1][0]), "control_mode": "IMPEDANCE", "gains": "default",
    }
    res = send_goal(arm, chain, points, rec, "beat")
    try:  # the summary is for the log only: after the send, so it never delays the move
        rec.update(playlog.summarize(points, chain, scene, held))
    except Exception as e:  # noqa: BLE001
        rec["summary_error"] = f"{type(e).__name__}: {e}"
    raise_for_result(arm, res, "beat")
    rec["ok"] = True
    return rec


def _estop_now(arm):
    """Whether /control_status says e-stopped right now (an unreadable status is "no")."""
    try:
        return bool(arm.control_status().estopped)
    except Exception:  # noqa: BLE001
        return False


def send_goal(arm, chain, points, rec, what):
    """Send one IMPEDANCE / "default" / QUEUE joint-trajectory goal and return its result.
    capability_engine's execute_joint_trajectory returns False (it does not raise) when the driver
    refuses the goal, including an e-stop already on at send time: that becomes EngineError("estopped")
    when /control_status says e-stopped, else a recoverable EngineError("refused", <arm.last_error>)."""
    try:
        res = arm.execute_joint_trajectory(
            chain.joint_names, points, control_mode=IMPEDANCE, gains="default", preemption=QUEUE
        )
    except (EStopped, ControlTaken) as e:
        raise _ended(e)
    except ArmError as e:
        rec["driver"] = {"refused": str(e)}
        raise EngineError("refused", f"the arm refused the {what}: {e}", recoverable=True)
    if res is False:
        last = str(getattr(arm, "last_error", "") or "")
        rec["driver"] = {"refused": last or "the driver refused the goal"}
        if _estop_now(arm):
            raise EngineError("estopped", f"the arm is e-stopped: the {what} was refused")
        raise EngineError(
            "refused", f"the driver refused the {what}: {last or 'no reason given'}", recoverable=True
        )
    rec["driver"] = {"error_code": int(res.error_code), "error_string": res.error_string}
    final_error = getattr(res, "final_error", None)
    if final_error is not None:
        rec["driver"]["final_error"] = [float(x) for x in np.asarray(final_error, dtype=float).ravel()]
    return res


def raise_for_result(arm, res, what):
    """Map a finished goal's non-zero error code: contact / estopped / control_taken / driver_error."""
    if res.error_code == 0:
        return
    name = ERRORS.get(int(res.error_code), res.error_string)
    if name == "PREEMPTED":
        raise EngineError("control_taken", f"the {what} was preempted: control was taken")
    if name in ("PATH_TOLERANCE_VIOLATED", "GOAL_TOLERANCE_VIOLATED"):
        raise EngineError("contact", f"the arm stopped the {what} early; possible contact; check the arm")
    if res.error_code == HALTED:  # the driver halted it: an e-stop, else control was taken
        if _estopped(arm):
            raise EngineError("estopped", f"the arm e-stopped during the {what}")
        raise EngineError("control_taken", f"the driver halted the {what}: control was taken")
    raise EngineError("driver_error", f"the driver ended the {what}: {res.error_string}", recoverable=True)


def _estopped(arm):
    deadline = time.monotonic() + HALT_STATUS_WAIT_S
    while True:
        try:
            if arm.control_status().estopped:
                return True
        except ArmError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _anchored(chain, q, anchors):
    return any(
        np.abs(shortest_delta(chain, np.asarray(a, dtype=float), q)).max() <= ANCHOR_TOL_RAD for a in anchors
    )
