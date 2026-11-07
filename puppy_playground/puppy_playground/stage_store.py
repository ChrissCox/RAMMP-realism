"""The stage path on disk: points plus the facts needed to refuse a stale or malformed file.

File (JSON): version, urdf_sha256, q_rest (the pose the path was planned from), stage_q, margin_m
(the extra clearance the path was checked with when written: person and obstacle boxes grown by
it), person_floor_cap_m (0.0, or the attended floor cap the path was checked with: every person-clearance
floor was min(first sample, cap); absent in older files, which read as 0.0; a file only loads in an engine
running the same cap), points [[t, [q x7], [dq x7]], ...]. The path's own sha256 (of the file's bytes) keys the
runtime re-check cache."""

import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from puppy_motion.joints import shortest_delta

VERSION = 1
MARGIN_M = 0.002
END_TOL = 1e-6
# the only non-zero attended floor cap (see engine.ATTENDED_FLOOR_CAP_M, which is this): nobody in the chair
ATTENDED_FLOOR_CAP_M = 0.020


class StageStoreError(Exception):
    """`code` is the engine error code: no_stage_path, bad_stage_path, stale_stage_path or bad_endpoints."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass
class StagePath:
    points: list  # [(t, q (7,), dq (7,)), ...]
    q_rest: np.ndarray
    stage_q: np.ndarray
    urdf_sha256: str
    margin_m: float
    file_sha256: str
    method: str = ""
    person_floor_cap_m: float = 0.0


def urdf_hash(urdf_text):
    return hashlib.sha256(urdf_text.encode("utf-8")).hexdigest()


def save(path, points, urdf_text, stage_q, margin_m=MARGIN_M, method="", person_floor_cap_m=0.0):
    """Write the path atomically (temp file, then rename)."""
    doc = {
        "version": VERSION,
        "urdf_sha256": urdf_hash(urdf_text),
        "q_rest": [float(x) for x in points[0][1]],
        "stage_q": [float(x) for x in stage_q],
        "margin_m": float(margin_m),
        "method": str(method),
        "person_floor_cap_m": float(person_floor_cap_m),
        "points": [[float(t), [float(x) for x in q], [float(x) for x in dq]] for t, q, dq in points],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def check_endpoints(sp, chain, stage_q):
    """StageStoreError("bad_endpoints") unless the path starts at t=0 on its recorded rest pose and
    ends on `stage_q`, both within 1e-6 rad modulo whole turns of the continuous joints (the driver
    reports those wrapped). A NaN anywhere fails."""
    t0, q0, qn = sp.points[0][0], sp.points[0][1], sp.points[-1][1]
    if not (t0 == 0.0):
        raise StageStoreError("bad_endpoints", f"the path starts at t={t0}, not 0")
    if not np.all(np.abs(shortest_delta(chain, sp.q_rest, q0)) <= END_TOL):
        raise StageStoreError("bad_endpoints", "the path does not start at its recorded rest pose")
    if not np.all(np.abs(shortest_delta(chain, np.asarray(stage_q, dtype=float), qn)) <= END_TOL):
        raise StageStoreError("bad_endpoints", "the path does not end at the stage pose")
    if not np.all(np.abs(shortest_delta(chain, sp.stage_q, np.asarray(stage_q, dtype=float))) <= END_TOL):
        raise StageStoreError("bad_endpoints", "the path's stage pose is not the current stage_pose.json")


def load(path, urdf_text, stage_q=None, chain=None, floor_cap_m=0.0):
    """The StagePath in `path`. Raises StageStoreError if the file is missing (no_stage_path),
    unparseable (bad_stage_path), planned for another URDF (stale_stage_path), or, when `stage_q` is
    given (with `chain`), does not start at t=0 on its recorded rest pose and end within 1e-6 rad of `stage_q`
    (bad_endpoints). `floor_cap_m`: the engine's attended floor cap (0.0 = off); a file recorded with
    another cap is refused (bad_stage_path)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise StageStoreError("no_stage_path", f"cannot read {path}: {e}")
    try:
        doc = json.loads(raw)
        if doc["version"] != VERSION:
            raise ValueError(f"version {doc['version']}")
        pts = [(float(t), np.array(q, dtype=float), np.array(dq, dtype=float)) for t, q, dq in doc["points"]]
        if len(pts) < 2 or any(q.shape != (7,) or dq.shape != (7,) for _, q, dq in pts):
            raise ValueError("need >= 2 points of 7 joints")
        sp = StagePath(
            pts, np.array(doc["q_rest"], dtype=float), np.array(doc["stage_q"], dtype=float),
            str(doc["urdf_sha256"]), float(doc["margin_m"]), hashlib.sha256(raw).hexdigest(),
            str(doc.get("method", "")),
        )
        if not (np.isfinite(sp.q_rest).all() and np.isfinite(sp.stage_q).all()):
            raise ValueError("q_rest and stage_q must be finite")
        margin = doc["margin_m"]
        # The margin is not a file choice: a larger one makes the rules quieter, not stricter.
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin != MARGIN_M:
            raise ValueError(f"margin_m must be exactly {MARGIN_M}, not {margin!r}")
        cap = doc["person_floor_cap_m"] if "person_floor_cap_m" in doc else 0.0
        try:
            ok = not isinstance(cap, bool) and isinstance(cap, (int, float)) and math.isfinite(cap) and cap >= 0
        except OverflowError:  # an integer too large for a float
            ok = False
        if not ok:
            raise ValueError(f"person_floor_cap_m must be a finite number >= 0, not {cap!r:.60}")
        if cap not in (0.0, ATTENDED_FLOOR_CAP_M):
            raise ValueError(f"person_floor_cap_m must be 0.0 or {ATTENDED_FLOOR_CAP_M}, not {cap!r:.60}")
        if cap != floor_cap_m:
            raise ValueError(
                f"it was approved with person_floor_cap_m {cap}, this engine runs with {floor_cap_m}: "
                "a path only replays under the setting it was checked with"
            )
        sp.person_floor_cap_m = float(cap)
        if sp.q_rest.shape != (7,) or sp.stage_q.shape != (7,):
            raise ValueError("q_rest and stage_q need 7 joints")
        if not all(np.isfinite(x).all() for _, q, dq in pts for x in (q, dq)):
            raise ValueError("non-finite number")
    except (ValueError, KeyError, TypeError, OverflowError) as e:
        raise StageStoreError("bad_stage_path", f"{path} is not a stage path: {e}")
    if sp.urdf_sha256 != urdf_hash(urdf_text):
        raise StageStoreError("stale_stage_path", f"{path} was planned for another URDF")
    if stage_q is not None:
        if chain is None:
            raise TypeError("load: `chain` is required when `stage_q` is given")
        check_endpoints(sp, chain, stage_q)
    return sp
