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

from puppy_motion.validate import check_beat

try:
    from puppy_motion.joints import align_turns, shortest_delta
except ImportError:  # until Task 5's joints.py lands; same names and meaning, remove then

    def _continuous(chain):
        return ~np.isfinite(np.asarray(chain.pos_limits, dtype=float)).all(axis=1)

    def shortest_delta(chain, q_from, q_to):
        d = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
        m = _continuous(chain)
        d[m] = (d[m] + np.pi) % (2 * np.pi) - np.pi
        return d

    def align_turns(chain, q_ref, points):
        shift = np.zeros(len(q_ref))
        m = _continuous(chain)
        first = np.asarray(points[0][1], dtype=float)
        shift[m] = 2 * np.pi * np.round((np.asarray(q_ref, dtype=float) - first)[m] / (2 * np.pi))
        return [(t, np.asarray(q, dtype=float) + shift, dq) for t, q, dq in points]


from . import playlog

STILL_RAD_S = 0.02  # every |dq| must be below this to start a beat
START_TOL_RAD = 0.002  # the beat must start this close to the measured posture
HALT_STATUS_WAIT_S = 0.5  # after a HALTED goal, how long /control_status may take to show an e-stop


def _measure(arm):
    """(q, max |dq|) of the arm now. EStopped / ControlTaken end the action, other errors skip."""
    try:
        js = arm.joint_states()
    except (EStopped, ControlTaken) as e:
        raise _ended(e)
    except ArmError as e:
        raise EngineError("no_state", f"cannot read the arm: {e}", recoverable=True)
    return np.array(js.position, dtype=float), float(np.abs(np.array(js.velocity, dtype=float)).max())


def _ended(e):
    if isinstance(e, EStopped):
        return EngineError("estopped", str(e))
    return EngineError("control_taken", str(e))


def play(arm, ctx, chain, scene, beat_builder, held, log=None, name=None):
    """Play one beat; returns the log record. `beat_builder(q, held)` gives the Beat (or None when
    there is nothing to play) for the measured posture q. On a failed check nothing is sent and
    EngineError("check_failed", <rules>, recoverable=True) is raised. `log(rec)` is called with the
    record however this ends."""
    rec = playlog.motion_record(name, held)
    try:
        return _play(arm, ctx, chain, scene, beat_builder, bool(held), rec)
    except BaseException as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if log is not None:
            log(rec)


def _play(arm, ctx, chain, scene, beat_builder, held, rec):
    q, speed = _measure(arm)
    if speed >= STILL_RAD_S:
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
    if speed >= STILL_RAD_S:
        raise EngineError("not_still", "the arm is moving", recoverable=True)
    gap = float(np.abs(shortest_delta(chain, q, points[0][1])).max())
    if gap >= START_TOL_RAD:
        raise EngineError(
            "start_mismatch", f"the arm is {gap:.4f} rad from where the beat starts", recoverable=True
        )
    points = align_turns(chain, q, points)  # the driver reports continuous joints wrapped
    found = check_beat(points, q, chain, scene, beat.zone, held=held)
    rec["checks"] = [f"{v.rule}: {v.detail}" for v in found]
    if found:
        raise EngineError("check_failed", "; ".join(rec["checks"]), recoverable=True)
    rec.update(playlog.summarize(points, chain, scene, held))
    rec["goal"] = {
        "points": len(points), "seconds": float(points[-1][0]), "control_mode": "IMPEDANCE", "gains": "default",
    }
    try:
        res = arm.execute_joint_trajectory(
            chain.joint_names, points, control_mode=IMPEDANCE, gains="default", preemption=QUEUE
        )
    except (EStopped, ControlTaken) as e:
        raise _ended(e)
    except ArmError as e:
        rec["driver"] = {"refused": str(e)}
        raise EngineError("refused", f"the arm refused the beat: {e}", recoverable=True)
    rec["driver"] = {"error_code": int(res.error_code), "error_string": res.error_string}
    if res.error_code != 0:
        if res.error_code == HALTED:  # the driver halted it: an e-stop, else control was taken
            if _estopped(arm):
                raise EngineError("estopped", "the arm e-stopped during the beat")
            raise EngineError("control_taken", "the driver halted the beat: control was taken")
        raise EngineError("driver_error", f"the driver ended the beat: {res.error_string}", recoverable=True)
    rec["ok"] = True
    return rec


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
