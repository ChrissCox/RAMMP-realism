"""The playground's JSON-lines log: one line per request (and one per auto-mode motion), with the
action, params, peak speeds, clearances, check results, the goal sent and the driver result."""

import json
import time

import numpy as np

from puppy_motion.geometry import PARTS, clearances_batch

CLEARANCE_STRIDE = 4  # clearances are summarised on every 4th sample


def write(path, rec, warn=None):
    """Append `rec` as one JSON line to `path`; an unwritable log is warned about, never fatal."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    except OSError as e:
        if warn is not None:
            warn(f"cannot write {path}: {e}")


def summarize(points, chain, scene, held):
    """Peak joint speeds and the smallest clearance of each part over the beat's points."""
    dq = np.abs(np.array([p[2] for p in points], dtype=float))
    q = np.array([p[1] for p in points], dtype=float)[::CLEARANCE_STRIDE]
    clear = clearances_batch(chain, scene, q, held).min(axis=0)  # (2, parts)
    return {
        "seconds": float(points[-1][0]),
        "peak_joint_speed": dq.max(axis=0).tolist(),
        "min_clearance_person": dict(zip(PARTS, clear[0].tolist())),
        "min_clearance_obstacles": dict(zip(PARTS, clear[1].tolist())),
    }


def motion_record(name, held):
    return {"t": time.time(), "motion": name, "held": bool(held)}
