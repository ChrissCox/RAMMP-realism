"""The stage path on disk: points plus the facts needed to refuse a stale or malformed file.

File (JSON): version, urdf_sha256, q_rest (the pose the path was planned from), stage_q, margin_m
(the extra clearance the path was checked with when written: person and obstacle boxes grown by
it), points [[t, [q x7], [dq x7]], ...]. The path's own sha256 (of the file's bytes) keys the
runtime re-check cache."""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

VERSION = 1
MARGIN_M = 0.002
END_TOL = 1e-6


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


def urdf_hash(urdf_text):
    return hashlib.sha256(urdf_text.encode("utf-8")).hexdigest()


def save(path, points, urdf_text, stage_q, margin_m=MARGIN_M):
    """Write the path atomically (temp file, then rename)."""
    doc = {
        "version": VERSION,
        "urdf_sha256": urdf_hash(urdf_text),
        "q_rest": [float(x) for x in points[0][1]],
        "stage_q": [float(x) for x in stage_q],
        "margin_m": float(margin_m),
        "points": [[float(t), [float(x) for x in q], [float(x) for x in dq]] for t, q, dq in points],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


def check_endpoints(sp):
    """StageStoreError("bad_endpoints") unless the path starts at t=0 on q_rest and ends on stage_q."""
    t0, q0, qn = sp.points[0][0], sp.points[0][1], sp.points[-1][1]
    if t0 != 0.0:
        raise StageStoreError("bad_endpoints", f"the path starts at t={t0}, not 0")
    if np.abs(q0 - sp.q_rest).max() > END_TOL:
        raise StageStoreError("bad_endpoints", "the path does not start at its recorded rest pose")
    if np.abs(qn - sp.stage_q).max() > END_TOL:
        raise StageStoreError("bad_endpoints", "the path does not end at the stage pose")


def load(path, urdf_text, stage_q=None):
    """The StagePath in `path`. Raises StageStoreError if the file is missing (no_stage_path),
    unparseable (bad_stage_path), planned for another URDF (stale_stage_path), or, when `stage_q` is
    given, does not start at t=0 on its recorded rest pose and end within 1e-6 rad of `stage_q`
    (bad_endpoints)."""
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
        )
        if sp.q_rest.shape != (7,) or sp.stage_q.shape != (7,):
            raise ValueError("q_rest and stage_q need 7 joints")
        if not all(np.isfinite(x).all() for _, q, dq in pts for x in (q, dq)):
            raise ValueError("non-finite number")
    except (ValueError, KeyError, TypeError) as e:
        raise StageStoreError("bad_stage_path", f"{path} is not a stage path: {e}")
    if sp.urdf_sha256 != urdf_hash(urdf_text):
        raise StageStoreError("stale_stage_path", f"{path} was planned for another URDF")
    if stage_q is not None:
        if np.abs(sp.stage_q - np.asarray(stage_q, dtype=float)).max() > END_TOL:
            raise StageStoreError("bad_endpoints", "the path's stage pose is not the current stage_pose.json")
        check_endpoints(sp)
    return sp
