"""Functional tracks built from the recorded runs in tests/data (no simulator needed)."""

import glob
import json
import os

import numpy as np

from puppy_motion.track import DT, Label, Track, Zone
from puppy_motion.zones import zone_of

DATA = os.path.join(os.path.dirname(__file__), "data")
RUN_NAMES = (
    "grab_container_20261029_132745",
    "grab_container_20261029_131558",
    "grab_container_20261028_191809",
)
APPROACH_MESSAGE = "view -> pre-grasp -> grasp, one motion"
LEAD_IN_S = 1.0  # the track starts this long before the goal's first phase
TAIL_S = 2.0  # and ends this long after the goal's terminal status
HOME_JOINTS = (0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725)
GRIP_TO_KNUCKLE_RAD = 0.8
SHORT_NAMES = {("grasp", "opening"): "open", ("grasp", APPROACH_MESSAGE): "approach", ("grasp", "closing"): "close"}


def _goal(raw):
    counts = {}
    for _, msg in raw["feedback"]:
        key = tuple(msg["goal_id"]["uuid"])
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


def functional_track(name):
    """The recorded run `name` on the DT grid as a functional Track (joints linearly resampled,
    zones from the zone table, one label per task phase)."""
    with open(os.path.join(DATA, name + ".json")) as f:
        raw = json.load(f)
    goal = _goal(raw)
    feedback = [
        (float(t), m["feedback"]["progress"]["phase"], m["feedback"]["progress"]["message"])
        for t, m in raw["feedback"]
        if tuple(m["goal_id"]["uuid"]) == goal
    ]
    t = np.array(raw["t"], dtype=float)
    q8 = np.array(raw["q"], dtype=float)
    q = np.unwrap(q8[:, :7], axis=0)
    grip_raw = np.clip(q8[:, 7] / GRIP_TO_KNUCKLE_RAD, 0.0, 1.0)
    t0 = max(t[0], feedback[0][0] - LEAD_IN_S)
    t1 = min(t[-1], raw["status"][-1][0] + TAIL_S) if raw["status"] else t[-1]
    grid = t0 + np.arange(int(np.floor((t1 - t0) / DT)) + 1) * DT
    n = len(grid)
    marks = [(t0, "idle", "")] + [f for f in feedback if f[0] > t0]
    ends = [m[0] for m in marks[1:]] + [t1 + DT]
    zone = np.zeros(n, dtype=int)
    labels = []
    for (ts, phase, msg), te in zip(marks, ends):
        zone[(grid >= ts) & (grid < te)] = int(zone_of("grab_container", phase, msg).zone)
        labels.append(Label(SHORT_NAMES.get((phase, msg), phase), ts - t0, te - t0))
    return Track(
        q=np.column_stack([np.interp(grid, t, q[:, j]) for j in range(7)]),
        grip=np.interp(grid, t, grip_raw),
        zone=zone,
        held=np.zeros(n, dtype=bool),
        tau=np.arange(n) * DT,
        labels=labels,
    )


def all_run_names():
    return [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(DATA, "*.json")))]
