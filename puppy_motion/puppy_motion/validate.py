"""The beat validator: the last independent check before a beat goes to the driver.

check_beat() takes a beat as the driver would get it, a list of (t, q, dq) points, and returns
every rule it breaks (one Violation per rule, at its worst sample). It is a second implementation:
it shares nothing with compose.py's fitting or checking code, only the data tables (caps, zones,
the Chain and the clearance geometry). Speeds and accelerations are forward differences on the
points, so a sample-to-sample zig-zag shows in full (a central difference would cancel it).

Velocity convention: the points carry CENTRAL-difference velocities, dq[i] = (q[i+1] - q[i-1]) / 2 DT,
with the first and last exactly 0 (the driver interpolates between points, and central differences
reproduce a constant acceleration). The `velocity` rule holds them to that within
VELOCITY_TOL = 1e-6 rad/s (producers compute them exactly, the driver feeds them forward), and each
carried |dq| must itself be inside the effective zone's joint speed caps and the URDF limits. The
speed and accel caps on the positions stay on forward differences.

The zone is not trusted: whenever the head (with the held object when `held`) is under 0.35 m from
the person at q_start, the stricter of the given zone and NEAR_USER applies.

check_beat never raises on a malformed beat or argument: that is a `shape` violation.

A beat starts and ends at the robot's measured posture `q_start`, at rest, and nothing upstream
checks the step into or out of it, so the `start` and `end` rules are that check. The whole beat
motion is expressive (the task is stopped), so it is held to the zone's caps as well as to the URDF
velocity limits, joint ranges and RAMMP's joint guards.
"""

from dataclasses import dataclass

import numpy as np

from .caps import CAP_PRESETS, DEFAULT_CAPS, URDF_GUARDS
from .geometry import clearances, clearances_batch
from .track import DT, Zone

REST_TOL = 1e-9  # first/last point equal q_start, and are at rest, within this (rad, rad/s)
DT_TOL = 1e-9  # s, how far a time step may be from DT
CAP_TOL = 1e-9  # a cap is broken when exceeded by more than this (float noise)
VELOCITY_TOL = 1e-6  # rad/s: carried dq vs the central difference of q
NEAR_USER_M = 0.35
PERSON_SLACK_M = 0.005  # a part may come this much closer to the person than min(task, 0.35)
OBSTACLE_KEEP_M = 0.03
OBSTACLE_TOL_M = 0.0005  # numerical tolerance on the obstacle rule, the same as the composer's
HEAD_COLUMN = 3  # geometry.PARTS.index("head")
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


def _worst(excess, what, first=0):
    """(sample, detail) of the largest entry of `excess` (rows are samples, the first one is sample
    `first`), None if none is > 0."""
    if excess.size == 0 or not np.any(excess > 0.0):
        return None
    i, j = np.unravel_index(np.argmax(excess), excess.shape)
    return int(i) + first, f"{what} column {int(j)}: over by {excess[i, j]:.6g}"


def check_beat(points, q_start, chain, scene, zone, caps=DEFAULT_CAPS, task_clearance=None, held=False):
    """Every rule the beat `points` [(t, q (7,), dq (7,)), ...] breaks, as Violations (empty when
    it is fit to send). `zone` picks the caps (CAP_PRESETS[caps][zone], or NEAR_USER's when the
    head is within 0.35 m of the person at q_start); `task_clearance` (2, 4) is the clearance at
    q_start (row 0 person, row 1 obstacles; computed when None); `held` says the container rides
    in the gripper (clearances then include it). A malformed beat or argument is a `shape`
    violation; only an unknown `caps` preset raises."""
    if caps not in CAP_PRESETS:
        raise ValueError(f"unknown caps preset {caps!r}; expected one of {sorted(CAP_PRESETS)}")
    try:
        return _check(points, q_start, chain, scene, zone, caps, task_clearance, bool(held))
    except Exception as e:  # a malformed argument must not take the caller down
        return [Violation("shape", 0, f"cannot check the beat: {type(e).__name__}: {e}")]


def _check(points, q_start, chain, scene, zone, caps, task_clearance, held):
    zone = Zone(zone)
    q_start = np.asarray(q_start, dtype=float)
    t = np.array([p[0] for p in points], dtype=float)
    q = np.array([p[1] for p in points], dtype=float)
    dq = np.array([p[2] for p in points], dtype=float)
    n = len(t)
    if n < 2 or t.shape != (n,) or q.shape != (n, 7) or dq.shape != (n, 7) or q_start.shape != (7,):
        return [Violation("shape", 0, f"need >= 2 points with 7 joints, got q {q.shape}, dq {dq.shape}")]
    if not (np.isfinite(t).all() and np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(q_start).all()):
        return [Violation("shape", 0, "a time, position or velocity is not finite")]

    # the clearance at q_start: the validator's own reading, and the task's as given (checked)
    own = clearances(chain, scene, q_start, held)
    if task_clearance is None:
        task = own
    else:
        task = np.asarray(task_clearance, dtype=float)
        if task.shape != (2, 4) or np.isnan(task).any() or (task == -np.inf).any():
            return [Violation("shape", 0, "task_clearance must be (2, 4) without NaN")]
        if not np.isfinite(task[0]).all() and not (task[0] == np.inf).all():
            return [Violation("shape", 0, "task_clearance person row must be finite")]

    out = []

    def add(rule, found):
        if found is not None:
            out.append(Violation(rule, found[0], found[1]))

    if zone == Zone.PRECISION:
        out.append(Violation("zone", 0, "no beat plays in PRECISION"))
    if not own[0, HEAD_COLUMN] >= NEAR_USER_M:  # under 0.35 m, or NaN: near the user whatever the caller says
        zone = max(zone, Zone.NEAR_USER)
    limits = CAP_PRESETS[caps][zone]
    no_person = not np.isfinite(task[0]).any() or not np.isfinite(own[0]).any()
    if no_person:
        out.append(Violation("person", 0, "no finite clearance to the person at q_start"))

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
    central = (q[2:] - q[:-2]) / (2.0 * DT)  # what the interior carried velocities must be
    add("velocity", _worst(np.abs(dq[1:-1] - central) - VELOCITY_TOL, "carried velocity vs central difference, joint", 1))
    over = np.maximum(np.abs(v), np.abs(dq[:-1])) - limits.joint_speed  # the step and the carried dq
    add("speed", _worst(over - CAP_TOL, "joint speed (step or carried dq)"))
    add("accel", _worst(np.abs(a) - limits.joint_accel - CAP_TOL, "joint accel"))
    over = np.maximum(np.abs(v), np.abs(dq[:-1])) - chain.vel_limits
    add("urdf_speed", _worst(over - CAP_TOL, "URDF velocity limit (step or carried dq), joint"))

    # head speed at the eye and tool points
    memo = {}

    def sites(qi):
        key = qi.tobytes()
        if key not in memo:
            memo[key] = np.array([chain.fk(qi, s)[0] for s in HEAD_SITES])
        return memo[key]

    pos = np.array([sites(qi) for qi in q])  # (n, sites, 3)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=2) / DT
    add("head_speed", _worst(speed - limits.head_speed - CAP_TOL, "head speed, site"))

    # joint ranges, then RAMMP's guards on what is inside the range
    lo, hi = chain.pos_limits[:, 0], chain.pos_limits[:, 1]
    beyond = (q < lo) | (q > hi)
    add("limits", _worst(np.where(beyond, np.maximum(lo - q, q - hi), 0.0), "joint range, joint"))
    guard = np.full(7, np.inf)
    for j, g in URDF_GUARDS.items():
        guard[j] = g
    add("guards", _worst(np.where(beyond, 0.0, np.abs(q) - guard), "joint guard, joint"))

    # clearances: a NaN reading is a violation
    floor = np.empty((2, 4))
    floor[0] = np.minimum(task[0], NEAR_USER_M) - PERSON_SLACK_M
    floor[1] = np.minimum(task[1], OBSTACLE_KEEP_M) - OBSTACLE_TOL_M
    keys = [qi.tobytes() for qi in q]
    first = {}  # each distinct posture once, batched
    for key, qi in zip(keys, q):
        first.setdefault(key, qi)
    seen = dict(zip(first, clearances_batch(chain, scene, np.array(list(first.values())), held)))
    short = np.zeros((n, 2, 4))
    for i, key in enumerate(keys):
        short[i] = floor - seen[key]
    short = np.where(np.isnan(short), np.inf, short)
    if not no_person:
        add("person", _worst(short[:, 0], "person clearance, part"))
    add("obstacle", _worst(short[:, 1], "obstacle clearance, part"))
    return out
