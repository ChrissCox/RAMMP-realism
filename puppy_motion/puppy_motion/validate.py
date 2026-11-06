"""The beat validator: the last independent check before a beat goes to the driver.

check_beat() takes a beat as the driver would get it, a list of (t, q, dq) points, and returns
every rule it breaks (one Violation per rule, at its worst sample). It is a second implementation:
it shares nothing with compose.py's fitting or checking code, only the data tables (caps, zones,
the Chain and the clearance geometry). Speeds and accelerations are forward differences on the
points, so a sample-to-sample zig-zag shows in full (a central difference would cancel it).

A beat starts and ends at the robot's measured posture `q_start`, at rest, and nothing upstream
checks the step into or out of it, so the `start` and `end` rules are that check. The whole beat
motion is expressive (the task is stopped), so it is held to the zone's caps as well as to the URDF
velocity limits, joint ranges and RAMMP's joint guards.
"""

from dataclasses import dataclass

import numpy as np

from .caps import CAP_PRESETS, DEFAULT_CAPS, URDF_GUARDS
from .geometry import clearances
from .track import DT, Zone

REST_TOL = 1e-9  # first/last point equal q_start, and are at rest, within this (rad, rad/s)
DT_TOL = 1e-9  # s, how far a time step may be from DT
CAP_TOL = 1e-9  # a cap is broken when exceeded by more than this (float noise)
VELOCITY_TOL = 0.05  # rad/s: the velocity a point carries vs the forward-difference speed (a smooth
# beat's forward and central differences differ by about accel * DT / 2, 0.03 rad/s at 3 rad/s^2)
NEAR_USER_M = 0.35
PERSON_SLACK_M = 0.005  # a part may come this much closer to the person than min(task, 0.35)
OBSTACLE_KEEP_M = 0.03
OBSTACLE_TOL_M = 0.0005  # numerical tolerance on the obstacle rule, the same as the composer's
HEAD_SITES = ("eye", "tool")
RULES = (
    "shape", "zone", "start", "end", "dt", "velocity", "speed", "accel", "head_speed",
    "urdf_speed", "limits", "guards", "person", "obstacle",
)


@dataclass(frozen=True)
class Violation:
    """A rule a beat breaks: `rule` (one of RULES), the worst `sample` index, and a `detail` line."""

    rule: str
    sample: int
    detail: str


def _worst(excess, what):
    """(sample, detail) of the largest entry of `excess` (rows are samples), None if none is > 0."""
    if excess.size == 0 or not np.any(excess > 0.0):
        return None
    i, j = np.unravel_index(np.argmax(excess), excess.shape)
    return int(i), f"{what} column {int(j)}: over by {excess[i, j]:.6g}"


def check_beat(points, q_start, chain, scene, zone, caps=DEFAULT_CAPS, task_clearance=None):
    """Every rule the beat `points` [(t, q (7,), dq (7,)), ...] breaks, as Violations (empty when
    it is fit to send). `zone` picks the caps (CAP_PRESETS[caps][zone]); `task_clearance` (2, 4) is
    the clearance at q_start (row 0 person, row 1 obstacles; computed when None). Clearances are
    taken without a held object."""
    if caps not in CAP_PRESETS:
        raise ValueError(f"unknown caps preset {caps!r}; expected one of {sorted(CAP_PRESETS)}")
    zone = Zone(zone)
    q_start = np.asarray(q_start, dtype=float)
    try:
        t = np.array([p[0] for p in points], dtype=float)
        q = np.array([p[1] for p in points], dtype=float)
        dq = np.array([p[2] for p in points], dtype=float)
    except (TypeError, ValueError, IndexError):
        return [Violation("shape", 0, "points are not (t, q, dq) triples of numbers")]
    n = len(t)
    if n < 2 or q.shape != (n, 7) or dq.shape != (n, 7) or q_start.shape != (7,):
        return [Violation("shape", 0, f"need >= 2 points with 7 joints, got q {q.shape}, dq {dq.shape}")]
    if not (np.isfinite(t).all() and np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(q_start).all()):
        return [Violation("shape", 0, "a time, position or velocity is not finite")]

    out = []

    def add(rule, found):
        if found is not None:
            out.append(Violation(rule, found[0], found[1]))

    if zone == Zone.PRECISION:
        out.append(Violation("zone", 0, "no beat plays in PRECISION"))
    limits = CAP_PRESETS[caps][zone]

    # the ends: at q_start, at rest
    gap = np.abs(q[0] - q_start)
    if gap.max() > REST_TOL or np.abs(dq[0]).max() > REST_TOL:
        out.append(Violation("start", 0, f"first point off q_start by {gap.max():.3g} rad, speed {np.abs(dq[0]).max():.3g}"))
    gap = np.abs(q[-1] - q_start)
    if gap.max() > REST_TOL or np.abs(dq[-1]).max() > REST_TOL:
        out.append(Violation("end", n - 1, f"last point off q_start by {gap.max():.3g} rad, speed {np.abs(dq[-1]).max():.3g}"))

    # the clock
    off = np.abs(np.diff(t) - DT)
    add("dt", _worst(np.maximum(off - DT_TOL, 0.0)[:, None], "time step"))

    # forward differences (speed per step, acceleration per step including the stop at each end)
    v = np.diff(q, axis=0) / DT
    v_rest = np.vstack([np.zeros((1, 7)), v, np.zeros((1, 7))])
    a = np.diff(v_rest, axis=0) / DT
    add("velocity", _worst(np.abs(dq[:-1] - v) - VELOCITY_TOL, "carried velocity vs forward difference, joint"))
    add("speed", _worst(np.abs(v) - limits.joint_speed - CAP_TOL, "joint speed"))
    add("accel", _worst(np.abs(a) - limits.joint_accel - CAP_TOL, "joint accel"))
    add("urdf_speed", _worst(np.abs(v) - chain.vel_limits - CAP_TOL, "URDF velocity limit, joint"))

    # head speed at the eye and tool points
    memo = {}

    def sites(qi):
        key = qi.tobytes()
        if key not in memo:
            memo[key] = np.array([chain.fk(qi, s)[0] for s in HEAD_SITES])
        return memo[key]

    pos = np.array([sites(qi) for qi in q])  # (n, sites, 3)
    head = np.linalg.norm(np.diff(pos, axis=0), axis=2) / DT
    add("head_speed", _worst(head - limits.head_speed - CAP_TOL, "head speed, site"))

    # joint ranges, then RAMMP's guards on what is inside the range
    lo, hi = chain.pos_limits[:, 0], chain.pos_limits[:, 1]
    beyond = (q < lo) | (q > hi)
    add("limits", _worst(np.where(beyond, np.maximum(lo - q, q - hi), 0.0), "joint range, joint"))
    guard = np.full(7, np.inf)
    for j, g in URDF_GUARDS.items():
        guard[j] = g
    add("guards", _worst(np.where(beyond, 0.0, np.abs(q) - guard), "joint guard, joint"))

    # clearances
    if task_clearance is None:
        task_clearance = clearances(chain, scene, q_start, False)
    task = np.asarray(task_clearance, dtype=float)
    floor = np.empty((2, 4))
    floor[0] = np.minimum(task[0], NEAR_USER_M) - PERSON_SLACK_M
    floor[1] = np.minimum(task[1], OBSTACLE_KEEP_M) - OBSTACLE_TOL_M
    seen = {}
    short = np.zeros((n, 2, 4))
    for i, qi in enumerate(q):
        key = qi.tobytes()
        if key not in seen:
            seen[key] = clearances(chain, scene, qi, False)
        short[i] = floor - seen[key]
    add("person", _worst(short[:, 0], "person clearance, part"))
    add("obstacle", _worst(short[:, 1], "obstacle clearance, part"))
    return out
