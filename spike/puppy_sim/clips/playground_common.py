"""Shared by the playground_* previews: the flatbot chair scene, the stage pose, and tracks built
from puppy_motion's beats and paths (the q arrays only; the spike's Track and renderer play them).

Nothing here talks to a robot. The previews are what the user approves before the real arm moves:
the stage pose (stage_pose.json), the candidate rest->stage path
(escape.escape_path), every idle expression, and auto mode."""

import hashlib
import importlib.util
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import render, runs
from ..flatbot_scene import FLATBOT, LEGEND
from ..timeline import DT, FREE, NEAR_USER, Label, Track

REST_GRIP = 0.79
SEED = 0
BLEND_MIN_S = 0.5
CACHE_DIR = render.WORK_DIR / "playground_cache"
PM_DIR = Path(__file__).resolve().parents[3] / "puppy_motion" / "puppy_motion"
STAGE_POSE_JSON = Path(__file__).resolve().parents[3] / "puppy_playground" / "puppy_playground" / "data" / "stage_pose.json"

# cameras, registered for the playground clips (the head of the stage pose is near (0.5, 0.5, 0.56))
render.CAMERAS.setdefault(
    "playground_close",
    render.Camera(lookat=None, distance=0.78, azimuth=152.0, elevation=-6.0, fovy=40.0, head_offset=(-0.05, -0.04, -0.03)),
)
render.CAMERAS.setdefault(
    "playground_wide",
    render.Camera(lookat=(0.20, 0.0, 0.25), distance=2.6, azimuth=232.0, elevation=-18.0, fovy=40.0),
)
render.CAMERAS.setdefault(
    "playground_still",
    render.Camera(lookat=(0.15, 0.05, 0.25), distance=2.6, azimuth=205.0, elevation=-38.0, fovy=40.0),
)

__all__ = ["FLATBOT", "LEGEND"]


def urdf_text():
    """flatbot's URDF as the recorded runs carry it."""
    return json.loads((runs.RUNS_DIR / "grab_container_20261029_132745.json").read_text())["urdf"]


def chain():
    from puppy_motion import urdf

    return urdf.Chain.from_urdf(urdf_text())


def chair():
    from puppy_motion import flatbot

    return flatbot.chair_scene()


def stage_pose(ch=None, chair_scene=None):
    """The shipped stage pose (puppy_playground's stage_pose.json: the first find_stage_poses pose,
    seed 0, that has a checked escape path); the first find_stage_poses pose if there is no file."""
    from puppy_motion import stage

    if STAGE_POSE_JSON.exists():
        return np.array(json.loads(STAGE_POSE_JSON.read_text())["q"], dtype=float)
    return stage.find_stage_poses(ch or chain(), chair_scene or chair(), 1, SEED)[0]


def code_hash():
    """A hash of puppy_motion's source: cached beats are only reused while it is unchanged."""
    h = hashlib.sha256()
    for p in sorted(PM_DIR.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:16]


def cached(key, make):
    """make() cached on disk under (key, puppy_motion's code hash): results are plain pickles."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (hashlib.sha256(f"{key}|{code_hash()}".encode()).hexdigest()[:20] + ".pkl")
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = make()
    path.write_bytes(pickle.dumps(value))
    return value


def points_q(points):
    return np.array([p[1] for p in points], dtype=float)


def hold_rows(q, seconds):
    return np.tile(np.asarray(q, dtype=float), (int(round(seconds / DT)), 1))


class Assembler:
    """Joint rows end to end on the 50 Hz clock, with phases and beat chips. A segment that starts
    off where the last one ended is joined by a short minimum-jerk blend (a beat ends within
    millimetres of where it started)."""

    def __init__(self, q0):
        self.rows = [np.asarray(q0, dtype=float)[None]]
        self.zone = [np.array([FREE])]
        self.beats, self.phases = [], []

    @property
    def last(self):
        return self.rows[-1][-1]

    @property
    def n(self):
        return sum(len(r) for r in self.rows)

    @property
    def t(self):
        return (self.n - 1) * DT

    def add(self, q, zone=FREE, chip=None, phase=None):
        """Append rows q (n, 7); the first sample is joined to the last. The chip / phase label
        covers the span of the rows."""
        from puppy_motion.track import _min_jerk

        q = np.asarray(q, dtype=float)
        gap = float(np.abs(q[0] - self.last).max())
        if gap > 1e-9:
            m = max(2, int(round(max(BLEND_MIN_S, gap / 0.3) / DT)) + 1)
            s = _min_jerk(np.linspace(0.0, 1.0, m))[1:, None]
            self._append(self.last + s * (q[0] - self.last), FREE)
        t0 = self.t
        self._append(q[1:] if gap <= 1e-9 else q, zone)
        if chip:
            self.beats.append(Label(chip, t0, self.t))
        if phase:
            self.phases.append(Label(phase, t0, self.t))

    def hold(self, seconds, zone=FREE, chip=None, phase=None):
        rows = hold_rows(self.last, seconds)
        t0 = self.t
        self._append(rows, zone)
        if chip:
            self.beats.append(Label(chip, t0, self.t))
        if phase:
            self.phases.append(Label(phase, t0, self.t))

    def _append(self, q, zone):
        self.rows.append(np.asarray(q, dtype=float))
        self.zone.append(np.full(len(q), zone, dtype=int))

    def track(self, name):
        q = np.vstack(self.rows)
        n = len(q)
        return Track(
            q=q,
            grip=np.full(n, REST_GRIP),
            zone=np.concatenate(self.zone),
            held=np.zeros(n, dtype=bool),
            tau=np.arange(n) * DT,
            phases=self.phases,
            beats=self.beats,
            name=name,
        )


def title_case(name):
    return name.upper().replace("_", " ")


def zone_of(beat):
    return NEAR_USER if int(beat.zone) == NEAR_USER else FREE


def settings_json():
    return json.loads(STAGE_POSE_JSON.read_text()) if STAGE_POSE_JSON.exists() else None


# ------------------------------------------------------------ the robot's stage-path candidate
CANDIDATE_ENV = "PLAYGROUND_CANDIDATE"
STORE_PY = Path(__file__).resolve().parents[3] / "puppy_playground" / "puppy_playground" / "stage_store.py"


def candidate_env():
    """The candidate file named by PLAYGROUND_CANDIDATE, or None."""
    return os.environ.get(CANDIDATE_ENV) or None


def stage_store():
    """puppy_playground's stage_store, loaded from its file (the spike does not install the package)."""
    spec = importlib.util.spec_from_file_location("_pp_stage_store", STORE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Candidate:
    """A stage_path.candidate.json read with stage_store.load (same format, URDF and endpoint checks)
    and re-checked against the chair scene grown by the engine's margin. error: why it could not be
    read (then the other fields are empty); violations: 'rule: detail' lines, none to be approvable."""

    path: str
    method: str = ""
    points: list = None
    duration: float = 0.0
    margin_m: float = 0.0
    violations: list = None
    error: str = ""
    sha256: str = ""  # of the file's bytes, to compare with `sha256sum` before it is copied

    @property
    def approvable(self):
        return not self.error and not self.violations


def load_candidate(path):
    from puppy_motion import escape, stage

    store = stage_store()
    cand = Candidate(str(path), points=[], violations=[])
    try:
        cand.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        pass  # store.load reports the unreadable file below
    ch, sc = chain(), chair()
    try:
        sp = store.load(path, urdf_text(), stage_pose(ch, sc), ch)
    except store.StageStoreError as e:
        cand.error = f"{e.code}: {e}"
        return cand
    cand.method, cand.points, cand.margin_m = sp.method or "unlabelled", sp.points, sp.margin_m
    cand.duration = float(sp.points[-1][0])
    grown = escape.grown_scene(sc, max(sp.margin_m, store.MARGIN_M))
    cand.violations = [f"{v.rule}: {v.detail}" for v in stage.path_violations(sp.points, ch, grown)]
    return cand
