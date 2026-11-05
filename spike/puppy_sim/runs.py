"""Real grab_container runs (arm data from the robot's rosbags): load, resample to the 50 Hz
clock with light smoothing, the phase timeline from the capability's progress feedback, and the
zone marking (a per-capability table plus geometric rules). Offline JSON only.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .kin import ARM_PARTS
from .timeline import DT, FREE, GRIP_TO_KNUCKLE_RAD, NEAR_USER, PRECISION, Label, Track

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"
TERMINAL_STATUS = {4: "done", 5: "canceled", 6: "aborted"}  # action_msgs GoalStatus
SMOOTH_SIGMA_S = 0.03  # gaussian on the resampled joints: removes sensor noise, keeps the motion
GRIP_HELD = 0.6  # grip above this during or after 'closing': the container rides in the gripper
LEAD_IN_S = 1.0  # a replay starts this long before the goal's first phase
TAIL_S = 2.0  # and ends this long after the goal's terminal status
NEAR_USER_M = 0.35  # arm (past joint 2) closer than this to the person: NEAR_USER
FINAL_APPROACH_M = 0.11  # tool within this of the grasp pose: the final approach (STANDOFF 0.10 + blend)
APPROACH_MESSAGE = "view -> pre-grasp -> grasp, one motion"
DRY_RUN_MESSAGE = "dry run: planning"  # grasp.py's dry run: it plans the approach and the raise, nothing moves
HOME_JOINTS = (0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725)  # grab_container's home_joints

# (phase, message) -> zone. Unknown pairs are PRECISION (fail-safe). The geometric rules add the
# final approach (PRECISION) and NEAR_USER on top. The table knows the recorded runs' pairs and
# today's grasp.py, which reports the approach under its own phase.
ZONE_TABLES = {
    "grab_container": {
        ("idle", ""): FREE,  # before the goal: the arm rests at home
        ("home", ""): FREE,
        ("locate", ""): PRECISION,  # the scene camera segments the container: keep out of its view
        ("view", ""): FREE,
        ("detect", ""): PRECISION,  # wrist-camera perception hold: frames must agree to 5 mm
        ("grasp", "opening"): FREE,
        ("grasp", APPROACH_MESSAGE): FREE,  # the recorded runs: free until the final approach (geometric rule)
        ("approach", APPROACH_MESSAGE): FREE,  # grasp.py today: the same approach
        ("grasp", "closing"): PRECISION,
        ("raise", ""): PRECISION,  # lifting it clear of the counter
        ("approach", DRY_RUN_MESSAGE): PRECISION,  # a dry run moves nothing, so neither does Puppy Mode
        ("raise", DRY_RUN_MESSAGE): PRECISION,
        ("done", ""): FREE,  # holding it, goal succeeded
        ("aborted", ""): FREE,  # the arm holds where it stopped
    },
}
SHORT_NAMES = {
    ("grasp", "opening"): "open",
    ("grasp", APPROACH_MESSAGE): "approach",
    ("approach", APPROACH_MESSAGE): "approach",
    ("grasp", "closing"): "close",
    ("approach", DRY_RUN_MESSAGE): "plan (dry run)",
    ("raise", DRY_RUN_MESSAGE): "plan the raise (dry run)",
}


@dataclass(frozen=True)
class Run:
    """One run as recorded. t: (N,) s since bag start; q: (N, 7) rad (continuous joints
    unwrapped); grip: (N,) 0 open..1 closed (knuckle / 0.8); feedback: [(t, phase, message)] of
    the run's goal; terminal: (t, name) or None; ee_t, ee_pos, ee_quat (xyzw): /ee_state."""

    name: str
    capability: str
    t: np.ndarray
    q: np.ndarray
    grip: np.ndarray
    feedback: tuple
    terminal: tuple | None
    ee_t: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray


def _goal_key(goal_id):
    return tuple(goal_id["uuid"])


def _main_goal(raw):
    """The goal with the most feedback (a bag can hold the tail of a previous goal)."""
    counts = {}
    for _, msg in raw["feedback"]:
        key = _goal_key(msg["goal_id"])
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


def _terminal(raw, goal):
    """(time, name) of the goal's first terminal status, or None."""
    for t, msg in raw["status"]:
        for s in msg["status_list"]:
            if _goal_key(s["goal_info"]["goal_id"]) == goal and s["status"] in TERMINAL_STATUS:
                return float(t), TERMINAL_STATUS[s["status"]]
    return None


@lru_cache(maxsize=8)
def load_run(name):
    """Parse data/runs/<name>.json."""
    raw = json.loads((RUNS_DIR / f"{name}.json").read_text())
    goal = _main_goal(raw)
    feedback = tuple(
        (float(t), m["feedback"]["progress"]["phase"], m["feedback"]["progress"]["message"])
        for t, m in raw["feedback"]
        if _goal_key(m["goal_id"]) == goal
    )
    q = np.array(raw["q"], dtype=float)
    ee = raw["ee"]
    return Run(
        name=name,
        capability=name.rsplit("_", 2)[0],
        t=np.array(raw["t"], dtype=float),
        q=np.unwrap(q[:, :7], axis=0),
        grip=q[:, 7] / GRIP_TO_KNUCKLE_RAD,
        feedback=feedback,
        terminal=_terminal(raw, goal),
        ee_t=np.array([e[0] for e in ee], dtype=float),
        ee_pos=np.array([[e[1]["pose"]["position"][k] for k in "xyz"] for e in ee]),
        ee_quat=np.array([[e[1]["pose"]["orientation"][k] for k in "xyzw"] for e in ee]),
    )


def phase_spans(run):
    """[(phase, message, t0, t1)] in run time: idle before the goal, each feedback until the next,
    then the terminal state until the data ends."""
    marks = [(run.t[0], "idle", "")] + list(run.feedback)
    if run.terminal is not None:
        marks.append((run.terminal[0], run.terminal[1], ""))
    ends = [m[0] for m in marks[1:]] + [run.t[-1]]
    return [(phase, msg, t0, t1) for (t0, phase, msg), t1 in zip(marks, ends) if t1 > t0]


def closing_index(run):
    """Raw sample index where the gripper first holds: grip > GRIP_HELD at or after 'closing'."""
    t_close = next(t for t, phase, msg in run.feedback if msg == "closing")
    after = np.nonzero((run.t >= t_close) & (run.grip > GRIP_HELD))[0]
    return int(after[0])


def replay_window(run):
    """(t0, t1) in run time: LEAD_IN_S before the first phase to TAIL_S after the terminal status."""
    t0 = max(run.t[0], run.feedback[0][0] - LEAD_IN_S)
    end = run.terminal[0] + TAIL_S if run.terminal is not None else run.t[-1]
    return t0, min(end, run.t[-1])


def resample(run, t0, t1):
    """Joints and grip on the DT grid over [t0, t1], joints lightly smoothed."""
    grid = t0 + np.arange(int(np.floor((t1 - t0) / DT)) + 1) * DT
    q = np.column_stack([np.interp(grid, run.t, run.q[:, j]) for j in range(7)])
    q = gaussian_filter1d(q, SMOOTH_SIGMA_S / DT, axis=0, mode="nearest")
    return grid, q, np.clip(np.interp(grid, run.t, run.grip), 0.0, 1.0)


def zone_of(capability, phase, message):
    """The table's zone for a (phase, message); PRECISION when the pair is unknown, including a new
    message in a known phase (fail-safe: a message the table has not seen may be a precise step)."""
    return ZONE_TABLES.get(capability, {}).get((phase, message), PRECISION)


def functional_track(run, scene_kin, window=None):
    """The run as a functional Track on the DT clock (t = 0 at the window start), with phases,
    the held flag and zones (table, final approach, NEAR_USER)."""
    t0, t1 = window or replay_window(run)
    grid, q, grip = resample(run, t0, t1)
    spans = [s for s in phase_spans(run) if s[3] > t0 and s[2] < t1]
    zone = np.full(len(grid), PRECISION, dtype=int)
    phases = []
    for phase, msg, a, b in spans:
        zone[(grid >= a) & (grid < b)] = zone_of(run.capability, phase, msg)
        phases.append(Label(SHORT_NAMES.get((phase, msg), phase), max(a, t0) - t0, min(b, t1) - t0))
    phases[-1] = Label(phases[-1].name, phases[-1].t0, len(grid) * DT)  # the last phase runs to the end
    held = np.zeros(len(grid), dtype=bool)
    if any(msg == "closing" for _, _, msg in run.feedback):
        held = grid >= run.t[closing_index(run)]
    track = Track(q=q, grip=grip, zone=zone, held=held, tau=np.arange(len(grid)) * DT, phases=phases, name=run.name)
    _mark_final_approach(track, run, scene_kin)
    mark_near_user(track, scene_kin)
    return track


def _mark_final_approach(track, run, k):
    """PRECISION while the tool closes in on the grasp: within FINAL_APPROACH_M of where the
    gripper closed, before it holds the container (afterwards the raise and the table rule)."""
    if not any(msg == "closing" for _, _, msg in run.feedback):
        return
    p_grasp, _ = k.fk(run.q[closing_index(run)], "tool")
    p_tool, _ = k.fk_track(track.q, "tool")
    track.zone[(np.linalg.norm(p_tool - p_grasp, axis=1) < FINAL_APPROACH_M) & ~track.held] = PRECISION


def mark_near_user(track, k):
    """NEAR_USER where a FREE sample brings the head (bracelet, face, gripper, held container)
    within NEAR_USER_M of the person. The other arm parts are guarded by the per-part clearance
    floor instead: the folded home pose keeps the elbow beside the user's shoulder all the time."""
    person = k.clearance_track(track)[:, 0, ARM_PARTS.index("head")]
    track.zone[(track.zone == FREE) & (person < NEAR_USER_M)] = NEAR_USER
