"""The escape leg of a stage path: how the arm leaves rest, where it starts inside the torso box.

A straight minimum-jerk move from rest to some stage poses dips deeper into the person boxes than
rest already is, which the stage-path rules refuse. escape_path first climbs the person-clearance
gradient in small joint-space steps (a part never ends a step below its clearance at rest or more
than CLIMB_DIP_M below the step before; obstacle floors are kept) until the forearm, wrist and head
are all clear of the person by PERSON_STAGE_M plus MARGIN_M. It then routes to the stage pose through
postures that keep every person floor (the straight line if that is feasible, else a seeded RRT,
shortcut and smoothed), times the polyline with a cosine ease and retimes it to the FREE caps.
It raises ValueError when no route is found. The result is meant to pass stage.path_violations;
callers check.

The rest move replays the stored path backwards: the same samples in reverse order, so it visits
exactly the postures the forward path was checked on (see stage.path_violations).
"""

import numpy as np

from .caps import DEFAULT_CAPS, guard_limits
from .geometry import PARTS, Box, Scene, clearances
from .joints import continuous, shortest_delta
from .stage import (
    OBSTACLE_KEEP_M, OBSTACLE_TOL_M, PERSON_STAGE_M, _FAR_PARTS, points_from_q, retime_path,
)
from .track import DT

STEP_RAD = 0.02  # largest joint step of the climb (max-norm)
MARGIN_M = 0.01  # climb to this far past PERSON_STAGE_M
MAX_STEPS = 400
NOMINAL_RAD_S = 0.30  # speed the polyline is first timed at, before retime_path slows it down
FD_EPS = 1e-4
BLEND_STEP_RAD = 0.01
OBSTACLE_EXTRA_M = 0.003
CLIMB_LIFT_M = 1e-5  # far parts end every step this far above their clearance at rest
CLIMB_DIP_M = 0.0005  # a step may lower a part this much, never below its clearance at rest


def grown_scene(scene, m):
    """The scene with every box grown by `m` metres: a path that passes the rules against it holds
    them with `m` to spare (stage paths are checked only at the 50 Hz samples)."""
    def grow(boxes):
        return [Box(b.center, b.half + m, b.R) for b in boxes]

    return Scene(person=grow(scene.person), obstacles=grow(scene.obstacles))


def _person(chain, scene, q):
    return clearances(chain, scene, q, False)[0]


def _obstacle_floor(chain, scene, q_rest):
    c = clearances(chain, scene, q_rest, False)[1]
    # the cap is raised by the extra margin stage paths are checked with (boxes grown by 2 mm)
    return np.minimum(c, OBSTACLE_KEEP_M + OBSTACLE_EXTRA_M) - OBSTACLE_TOL_M


def _ok(chain, scene, q, c_first, c_prev, floor, lim):
    """Whether the step to q keeps every part's person clearance (no decrease) and the obstacle
    floors, and stays inside the guards."""
    if np.any(np.abs(q) > lim):
        return False, c_prev
    c = clearances(chain, scene, q, False)
    if np.isnan(c).any():
        return False, c_prev
    return bool(np.all(c[0] >= np.maximum(c_first + CLIMB_LIFT_M, c_prev - CLIMB_DIP_M)) and np.all(c[1] >= floor)), c[0]


def _gradient(chain, scene, q):
    """(4, 7) d(person clearance per part) / dq, by central differences."""
    g = np.zeros((len(PARTS), 7))
    for j in range(7):
        d = np.zeros(7)
        d[j] = FD_EPS
        g[:, j] = (_person(chain, scene, q + d) - _person(chain, scene, q - d)) / (2 * FD_EPS)
    return g


def _climb(chain, scene, q_rest, lim, floor):
    """Waypoints from q_rest: steps up the clearance gradient of the far parts still under the
    target, never lowering any part's clearance. Raises ValueError when it gets stuck."""
    target = PERSON_STAGE_M + MARGIN_M
    q = np.asarray(q_rest, dtype=float).copy()
    path = [q.copy()]
    c_prev = _person(chain, scene, q)
    c_first = c_prev.copy()
    c_first[0] -= CLIMB_LIFT_M  # the upper arm only has to hold
    for _ in range(MAX_STEPS):
        deficit = np.array([max(0.0, target - c_prev[j]) for j in _FAR_PARTS])
        if not deficit.any():
            return path
        g = _gradient(chain, scene, q)
        active = [j for j, d in zip(_FAR_PARTS, deficit) if d > 0.0]
        # least-norm step raising every active part at the same rate; the other parts must not drop
        A = g[active]
        rhs = np.minimum(deficit[deficit > 0.0], 0.05)
        step = A.T @ np.linalg.solve(A @ A.T + 1e-4 * np.eye(len(active)), rhs)
        # damp toward a step that also helps (or leaves alone) the parts that are not active
        for _ in range(8):
            others = [j for j in range(len(PARTS)) if j not in active]
            bad = [j for j in others if g[j] @ step < 0.0]
            if not bad:
                break
            B = g[bad]
            step = step - B.T @ np.linalg.solve(B @ B.T + 1e-6 * np.eye(len(bad)), B @ step)
        m = np.abs(step).max()
        if m < 1e-9:
            raise ValueError("escape_path: no direction raises the far parts without lowering another")
        step = step * min(1.0, STEP_RAD / m)
        for _ in range(12):
            q_new = q + step
            ok, c_new = _ok(chain, scene, q_new, c_first, c_prev, floor, lim)
            if ok:
                break
            step = step * 0.5
        else:
            raise ValueError("escape_path: the climb is stuck")
        q, c_prev = q_new, c_new
        path.append(q.copy())
    raise ValueError("escape_path: the climb did not clear the person")


def unwrap_to(chain, q_ref, q):
    """`q` with each continuous joint (no range limit) turned by whole revolutions to the value
    nearest `q_ref`: the same posture, the shortest way round."""
    q_ref = np.asarray(q_ref, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.where(continuous(chain), q_ref + shortest_delta(chain, q_ref, q), q)


FAR_FLOOR_M = PERSON_STAGE_M + 0.005  # the far parts stay this far out once clear
EDGE_RES_RAD = 0.02  # feasibility is checked along every edge at this spacing
RRT_STEP_RAD = 0.2
RRT_ITERS = 2000
RRT_GOAL_BIAS = 0.25
SHORTCUTS = 150
SMOOTH_ROUNDS = 2


def _feasible(chain, scene, q, person_floor, obstacle_floor, lim):
    if np.any(np.abs(q) > lim):
        return False
    c = clearances(chain, scene, q, False)
    return bool((not np.isnan(c).any()) and np.all(c[0] >= person_floor) and np.all(c[1] >= obstacle_floor))


def _edge_ok(feasible, a, b):
    n = max(1, int(np.ceil(np.abs(b - a).max() / EDGE_RES_RAD)))
    return all(feasible(a + (b - a) * k / n) for k in range(1, n + 1))


def _plan(feasible, q0, q_goal, lim, seed=0):
    """Waypoints q0 -> q_goal through the feasible set: the straight line when it is feasible, else a
    goal-biased RRT (seeded, so the same inputs give the same path), shortcut and smoothed.
    Raises ValueError when no path is found."""
    if _edge_ok(feasible, q0, q_goal):
        return [q0.copy(), q_goal.copy()]
    rng = np.random.default_rng(seed)
    span = np.where(np.isfinite(lim), lim, np.pi)
    nodes, parent = [q0.copy()], [-1]
    goal_i = None
    for _ in range(RRT_ITERS):
        target = q_goal if rng.random() < RRT_GOAL_BIAS else rng.uniform(-span, span)
        i = int(np.argmin(np.abs(np.array(nodes) - target).max(axis=1)))
        d = target - nodes[i]
        m = np.abs(d).max()
        if m < 1e-9:
            continue
        new = nodes[i] + d * min(1.0, RRT_STEP_RAD / m)
        if feasible(new) and _edge_ok(feasible, nodes[i], new):
            nodes.append(new)
            parent.append(i)
            if np.abs(q_goal - new).max() <= RRT_STEP_RAD and _edge_ok(feasible, new, q_goal):
                nodes.append(q_goal.copy())
                parent.append(len(nodes) - 2)
                goal_i = len(nodes) - 1
                break
    if goal_i is None:
        raise ValueError("escape_path: no feasible route to the stage pose found")
    way, i = [], goal_i
    while i >= 0:
        way.append(nodes[i])
        i = parent[i]
    way = way[::-1]
    for _ in range(SHORTCUTS):  # drop waypoints the straight edge can skip
        if len(way) < 3:
            break
        i, j = sorted(rng.choice(len(way), size=2, replace=False))
        if j - i > 1 and _edge_ok(feasible, way[i], way[j]):
            way = way[: i + 1] + way[j:]
    for _ in range(SMOOTH_ROUNDS):  # round the corners (Chaikin), keeping a round only if it is feasible
        if len(way) < 3:
            break
        cut = [way[0]]
        for a_, b_ in zip(way[:-1], way[1:]):
            cut += [0.75 * a_ + 0.25 * b_, 0.25 * a_ + 0.75 * b_]
        cut.append(way[-1])
        if all(_edge_ok(feasible, x, y) for x, y in zip(cut[:-1], cut[1:])):
            way = cut
    return way


def _densify(waypoints, max_step):
    out = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / max_step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def escape_path(chain, scene, q_rest, q_stage, caps=DEFAULT_CAPS):
    """Points [(t, q, dq), ...] from q_rest to q_stage: the climb out of the person, the route to
    q_stage, one zero-to-zero timing, retimed to the FREE caps. Continuous joints
    take the whole-turn equivalent of q_stage nearest q_rest, so the path ends on that (the same
    posture; callers who need the exact end values pass an already-unwrapped q_stage)."""
    q_rest = np.asarray(q_rest, dtype=float)
    q_stage = unwrap_to(chain, q_rest, q_stage)  # the path ends on this equivalent of the stage pose
    lim = guard_limits(chain)
    floor = _obstacle_floor(chain, scene, q_rest)
    way = _climb(chain, scene, q_rest, lim, floor)
    # on to the stage pose, holding every person floor (far parts stay clear of 0.35 m)
    person_floor = _person(chain, scene, q_rest).copy()
    person_floor[list(_FAR_PARTS)] = FAR_FLOOR_M

    def feasible(x):
        return _feasible(chain, scene, x, person_floor, floor, lim)

    way = way + _plan(feasible, way[-1], q_stage, lim)[1:]
    poly = _densify(np.array(way), BLEND_STEP_RAD)
    arc = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(poly, axis=0)).max(axis=1))])
    seconds = max(2 * DT, arc[-1] / NOMINAL_RAD_S)
    m = max(2, int(round(seconds / DT)) + 1)
    # Cosine ease, zero velocity at both ends. Unlike min-jerk it leaves rest quickly enough that the
    # first samples clear the ~1e-8 m noise of the capsule-box distance, which the path floor rule
    # (first sample, 1e-9 m) would otherwise read as a dip.
    pos = (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, m))) / 2.0 * arc[-1]
    q = np.stack([np.interp(pos, arc, poly[:, j]) for j in range(7)], axis=1)
    q[0], q[-1] = q_rest, q_stage
    return retime_path(points_from_q(q), chain, caps)
