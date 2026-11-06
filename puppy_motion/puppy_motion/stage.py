"""Stage poses and stage paths: where the arm waits to perform, and how it gets there and back.

A stage pose is a posture held well away from the occupant, looking forward and level. A stage
path is a joint-space path between rest and a stage pose. Both are held to rules (the playground's
global constraints), checked here and nowhere else:

Stage pose (stage_violations): the forearm, wrist and head are >= 0.35 m from the person; the head
is >= 0.10 m from the seat and platform; the gaze (ee +z) is within 15 deg of horizontal; the face's
up (ee +y) is within 20 deg of world up; every joint is 10 deg inside the guards and range limits.
The upper arm is exempt from the person rule: it stays near the occupant in every posture.

Stage path (path_violations), on points [(t, q, dq), ...] in check_beat's convention (DT clock,
central-difference dq, zero velocity at both ends):
  path_person_floor     every part's person clearance stays >= its value at the first sample
  path_person_monotone  once the forearm, wrist and head each reach 0.35 m, they stay >= 0.35 m
  path_obstacle         obstacle clearance stays >= min(first sample, 0.03) - 0.0005
  path_caps             FREE caps (joint speed, accel, head speed), URDF limits, ranges and guards
  path_ends             starts and ends at rest, on the clock, well formed
retime_path only ever slows a path down, uniformly, until it meets the FREE caps.
"""

import numpy as np

from .caps import CAP_PRESETS, DEFAULT_CAPS, guard_limits
from .geometry import PARTS, clearances, clearances_batch
from .track import DT, Zone, _min_jerk
from .validate import Violation

# flatbot's rest pose (the recorded runs' home joints)
REST_Q = np.array([0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725])

PERSON_STAGE_M = 0.35
OBSTACLE_STAGE_M = 0.10
GAZE_MAX_DEG = 15.0
UP_MAX_DEG = 20.0
GUARD_MARGIN = np.radians(10.0)
OBSTACLE_KEEP_M = 0.03
OBSTACLE_TOL_M = 0.0005
CAP_TOL = 1e-9
REST_TOL = 1e-9
DT_TOL = 1e-9
FLOOR_TOL = 1e-9
HEAD_SITES = ("eye", "tool")
_FAR_PARTS = [PARTS.index(p) for p in ("forearm", "wrist", "head")]
_HEAD = PARTS.index("head")
# forward and level, face up: ee z -> +x, ee y -> +z
_TARGET_R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def stage_limits(chain):
    """(7,) |q| allowed on a stage pose: the guards and range limits, less 10 deg."""
    return guard_limits(chain) - GUARD_MARGIN


def _angles(R):
    gaze = np.degrees(np.arcsin(np.clip(R[2, 2], -1.0, 1.0)))  # elevation of ee +z
    up = np.degrees(np.arccos(np.clip(R[2, 1], -1.0, 1.0)))  # ee +y from world up
    return gaze, up


def stage_violations(chain, scene, q):
    """Every stage-pose rule `q` breaks, as Violations (empty when it is a stage pose)."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        return [Violation("stage_guard", 0, "q must be 7 finite joints")]
    out = []
    over = np.abs(q) - stage_limits(chain)
    if np.any(over > 0.0):
        j = int(np.argmax(over))
        out.append(Violation("stage_guard", 0, f"joint {j} is {over[j]:.4g} rad past the stage limit"))
    _, R = chain.fk(q, "ee")
    gaze, up = _angles(R)
    if not abs(gaze) <= GAZE_MAX_DEG:
        out.append(Violation("stage_gaze", 0, f"gaze is {gaze:.1f} deg from horizontal, limit {GAZE_MAX_DEG}"))
    if not up <= UP_MAX_DEG:
        out.append(Violation("stage_up", 0, f"face up is {up:.1f} deg from world up, limit {UP_MAX_DEG}"))
    c = clearances(chain, scene, q, False)
    for j in _FAR_PARTS:
        if not c[0, j] >= PERSON_STAGE_M:
            out.append(Violation("stage_person", 0, f"{PARTS[j]} is {c[0, j]:.3f} m from the person, need {PERSON_STAGE_M}"))
    if not c[1, _HEAD] >= OBSTACLE_STAGE_M:
        out.append(Violation("stage_obstacle", 0, f"head is {c[1, _HEAD]:.3f} m from the seat/platform, need {OBSTACLE_STAGE_M}"))
    return out


def _orientation_error(R):
    return 0.5 * sum(np.cross(R[:, k], _TARGET_R[:, k]) for k in range(3))


def _refine(chain, q, lim, iters=30):
    """Damped least squares on the orientation only, toward forward and level, inside `lim`."""
    for _ in range(iters):
        _, R = chain.fk(q, "ee")
        e = _orientation_error(R)
        if np.linalg.norm(e) < 0.02:
            break
        J = chain.jacobian(q, "ee")[3:]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), e)
        q = np.clip(q + dq, -lim, lim)
    return q


def find_stage_poses(chain, scene, n, seed, tries=20000):
    """Up to `n` stage poses: random postures inside the stage limits, refined by damped least
    squares toward forward and level, keeping those with no violations. Sorted by the smallest head
    travel from REST_Q; poses closer than 0.15 rad (max joint) to a better one are dropped."""
    rng = np.random.default_rng(seed)
    lim = np.minimum(stage_limits(chain), np.pi)  # continuous joints: one turn
    rest_head = chain.fk(REST_Q, "tool")[0]
    found = []
    for _ in range(tries):
        q = _refine(chain, rng.uniform(-lim, lim), lim)
        if stage_violations(chain, scene, q):
            continue
        found.append((float(np.linalg.norm(chain.fk(q, "tool")[0] - rest_head)), q))
        if len(found) >= max(4 * n, 12):
            break
    found.sort(key=lambda f: f[0])
    out = []
    for _, q in found:
        if all(np.abs(q - o).max() > 0.15 for o in out):
            out.append(q)
        if len(out) == n:
            break
    return out


# ---- paths -------------------------------------------------------------------------------------


def points_from_q(q):
    """check_beat points [(t, q, dq)] from joint samples (n, 7) on the DT clock: central-difference
    velocities, zero at both ends."""
    q = np.asarray(q, dtype=float)
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2.0 * DT)
    return [(k * DT, q[k].copy(), dq[k].copy()) for k in range(len(q))]


def min_jerk_path(q0, q1, seconds):
    """A minimum-jerk joint move from q0 to q1 over `seconds` (rounded to the DT clock), as points."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    s = _min_jerk(np.linspace(0.0, 1.0, max(2, int(round(seconds / DT)) + 1)))
    return points_from_q(q0 + s[:, None] * (q1 - q0))


def _free_caps(caps):
    return CAP_PRESETS[caps][Zone.FREE]


def _arrays(points):
    q = np.array([p[1] for p in points], dtype=float)
    dq = np.array([p[2] for p in points], dtype=float)
    return q, dq


def _head_speed(chain, q):
    pos = np.array([[chain.fk(qi, s)[0] for s in HEAD_SITES] for qi in q])
    return np.linalg.norm(np.diff(pos, axis=0), axis=2) / DT


def _cap_ratios(q, chain, caps):
    """Ratios of the path's motion to the FREE caps (> 1 breaks a cap): (speed, accel, head)."""
    lim = _free_caps(caps)
    v = np.diff(q, axis=0) / DT
    a = np.diff(np.vstack([np.zeros((1, 7)), v, np.zeros((1, 7))]), axis=0) / DT
    central = (q[2:] - q[:-2]) / (2.0 * DT)
    return (
        max(float(np.max(np.abs(v) / lim.joint_speed)), float(np.max(np.abs(central) / lim.joint_speed, initial=0.0))),
        float(np.max(np.abs(a) / lim.joint_accel)),
        float(np.max(_head_speed(chain, q) / lim.head_speed)),
    )


def _hermite(q, dq, tau):
    """Cubic Hermite interpolation of samples (DT apart, carried velocities dq) at fractional
    sample indices tau."""
    i = np.minimum(np.floor(tau).astype(int), len(q) - 2)
    u = (tau - i)[:, None]
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * q[i] + h10 * DT * dq[i] + h01 * q[i + 1] + h11 * DT * dq[i + 1]


def retime_path(points, chain, caps=DEFAULT_CAPS):
    """The same path, uniformly stretched in time (never faster) until it meets the FREE caps."""
    if caps not in CAP_PRESETS:
        raise ValueError(f"unknown caps preset {caps!r}; expected one of {sorted(CAP_PRESETS)}")
    q, dq = _arrays(points)
    n = len(q)
    rv, ra, rh = _cap_ratios(q, chain, caps)
    s = max(1.0, rv, rh, np.sqrt(ra))  # stretching by s divides speeds by s and accelerations by s^2
    for _ in range(200):
        n_new = int(np.ceil((n - 1) * s - 1e-9)) + 1
        qn = _hermite(q, dq, np.linspace(0.0, n - 1.0, n_new))
        qn[0], qn[-1] = q[0], q[-1]
        if max(_cap_ratios(qn, chain, caps)) <= 1.0 - 1e-6:
            return points_from_q(qn)
        s *= 1.01
    raise ValueError("cannot retime the path to the FREE caps")


def _worst(excess, what, names=None):
    if excess.size == 0 or not np.any(excess > 0.0):
        return None
    i, j = np.unravel_index(np.argmax(excess), excess.shape)
    name = names[j] if names else j
    return int(i), f"{what} {name}: over by {excess[i, j]:.6g}"


def path_violations(points, chain, scene, caps=DEFAULT_CAPS, person_floor_ref=None):
    """Every stage-path rule the path `points` [(t, q, dq), ...] breaks, as Violations.
    `person_floor_ref` (shape (4,), one per part): the person clearances the path may not fall below;
    default None = the first sample's own clearances."""
    try:
        return _path_violations(points, chain, scene, caps, person_floor_ref)
    except Exception as e:  # a malformed path must not take the caller down
        return [Violation("path_ends", 0, f"cannot check the path: {type(e).__name__}: {e}")]


def _path_violations(points, chain, scene, caps, person_floor_ref=None):
    t = np.array([p[0] for p in points], dtype=float)
    q, dq = _arrays(points)
    n = len(t)
    if n < 2 or q.shape != (n, 7) or dq.shape != (n, 7):
        return [Violation("path_ends", 0, f"need >= 2 points with 7 joints, got q {q.shape}")]
    if not (np.isfinite(t).all() and np.isfinite(q).all() and np.isfinite(dq).all()):
        return [Violation("path_ends", 0, "a time, position or velocity is not finite")]
    out = []

    def add(rule, found):
        if found is not None:
            out.append(Violation(rule, found[0], found[1]))

    if np.abs(dq[0]).max() > REST_TOL or np.abs(dq[-1]).max() > REST_TOL:
        out.append(Violation("path_ends", 0, "the path does not start and end at rest"))
    if np.abs(np.diff(t) - DT).max() > DT_TOL:
        out.append(Violation("path_ends", 0, "time steps are not DT"))

    lim = _free_caps(caps)
    v = np.diff(q, axis=0) / DT
    a = np.diff(np.vstack([np.zeros((1, 7)), v, np.zeros((1, 7))]), axis=0) / DT
    central = (q[2:] - q[:-2]) / (2.0 * DT)
    if n > 2 and np.abs(dq[1:-1] - central).max() > 1e-6:
        out.append(Violation("path_caps", 1, "carried velocity is not the central difference of q"))
    step = np.maximum(np.abs(v), np.abs(dq[:-1]))
    add("path_caps", _worst(step - lim.joint_speed - CAP_TOL, "joint speed, joint"))
    add("path_caps", _worst(np.abs(a) - lim.joint_accel - CAP_TOL, "joint accel, joint"))
    add("path_caps", _worst(step - chain.vel_limits - CAP_TOL, "URDF velocity limit, joint"))
    add("path_caps", _worst(_head_speed(chain, q) - lim.head_speed - CAP_TOL, "head speed, site", HEAD_SITES))
    lo, hi = chain.pos_limits[:, 0], chain.pos_limits[:, 1]
    add("path_caps", _worst(np.maximum(lo - q, q - hi), "joint range, joint"))
    add("path_caps", _worst(np.abs(q) - guard_limits(chain) - CAP_TOL, "joint guard, joint"))

    c = clearances_batch(chain, scene, q, False)  # (n, 2, 4)
    if np.isnan(c).any():
        out.append(Violation("path_person_floor", int(np.argwhere(np.isnan(c))[0][0]), "NaN clearance"))
        return out
    person, obstacle = c[:, 0, :], c[:, 1, :]
    ref = person[0] if person_floor_ref is None else np.asarray(person_floor_ref, dtype=float)
    add("path_person_floor", _worst(ref - FLOOR_TOL - person, "person clearance below its first sample,", PARTS))
    far = person[:, _FAR_PARTS]
    reached = np.maximum.accumulate(far >= PERSON_STAGE_M, axis=0)  # each part has reached 0.35 m by here
    far_names = [PARTS[j] for j in _FAR_PARTS]
    add(
        "path_person_monotone",
        _worst(np.where(reached, PERSON_STAGE_M - far, 0.0) - FLOOR_TOL, "back under 0.35 m from the person,", far_names),
    )
    floor = np.minimum(obstacle[0], OBSTACLE_KEEP_M) - OBSTACLE_TOL_M
    add("path_obstacle", _worst(floor - obstacle, "obstacle clearance,", PARTS))
    return out
