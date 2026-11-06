"""Capsule model of the arm and its clearance to the person and to obstacles (pure numpy)."""
import json
from dataclasses import dataclass, field
from typing import List

import numpy as np

PARTS = ("upper_arm", "forearm", "wrist", "head")
RADII = {"upper_arm": 0.06, "forearm": 0.055, "wrist": 0.05, "head": 0.065}  # head grown from the brief's 0.06 so it reads no more clearance than the spike mesh (0.377 m)
HELD_RADIUS = 0.09
HELD_OFFSET = 0.17  # along the end-effector z axis
# Link-frame origins (URDF link name suffixes) joined into each part's capsule chain.
PART_LINKS = {
    "upper_arm": ("half_arm_1_link", "half_arm_2_link", "forearm_link"),
    "forearm": ("forearm_link", "spherical_wrist_1_link"),
    "wrist": ("spherical_wrist_1_link", "spherical_wrist_2_link", "bracelet_link"),
}


def _v(x):
    return np.asarray(x, dtype=float).reshape(3)


@dataclass
class Sphere:
    c: np.ndarray
    r: float

    def __post_init__(self):
        self.c, self.r = _v(self.c), float(self.r)


@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    r: float

    def __post_init__(self):
        self.a, self.b, self.r = _v(self.a), _v(self.b), float(self.r)


@dataclass
class Box:
    center: np.ndarray
    half: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.center, self.half = _v(self.center), _v(self.half)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)


def _point_box(points, box):
    """Distance from each point (n, 3) to the solid box (0 inside)."""
    local = (points - box.center) @ box.R
    out = np.maximum(np.abs(local) - box.half, 0.0)
    return np.linalg.norm(out, axis=1)


def _seg_seg(p1, q1, p2, q2):
    """Distance between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps * a * e else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _seg_box(a, b, box, rounds=6, n=17):
    """Distance from segment ab to the solid box. The point-to-box distance is convex along the
    segment, so repeatedly sampling n points and narrowing to the neighbours of the best one
    converges to the minimum."""
    lo, hi = 0.0, 1.0
    best = np.inf
    for _ in range(rounds):
        ts = np.linspace(lo, hi, n)
        d = _point_box(a + ts[:, None] * (b - a), box)
        i = int(np.argmin(d))
        best = min(best, float(d[i]))
        step = ts[1] - ts[0]
        lo, hi = max(0.0, ts[i] - step), min(1.0, ts[i] + step)
        if best == 0.0:
            break
    return best


def _as_capsule(s):
    return Capsule(s.c, s.c, s.r) if isinstance(s, Sphere) else s


def distance(x, y):
    """Gap between a Capsule or Sphere and a Sphere, Capsule or Box: the distance between the core
    segments (or the segment and the solid box) minus the radii. Positive when apart, <= 0 when
    overlapping; it is not a penetration depth (e.g. a segment inside a box returns -r)."""
    if isinstance(x, Box):
        x, y = y, x
    if isinstance(y, Box):
        c = _as_capsule(x)
        return _seg_box(c.a, c.b, y) - c.r
    cx, cy = _as_capsule(x), _as_capsule(y)
    return _seg_seg(cx.a, cx.b, cy.a, cy.b) - cx.r - cy.r


def _shapes(group):
    group = group or {}
    out: list = [Sphere(s[:3], s[3]) for s in group.get("spheres", [])]
    out += [Capsule(c[:3], c[3:6], c[6]) for c in group.get("capsules", [])]
    out += [Box(b[:3], b[3:6]) for b in group.get("boxes", [])]
    return out


@dataclass
class Scene:
    person: List = field(default_factory=list)
    obstacles: List = field(default_factory=list)

    @classmethod
    def from_param(cls, text):
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as e:
            raise ValueError(f"scene is not valid JSON: {e}")
        if not isinstance(data, dict):
            raise ValueError("scene must be a JSON object")
        try:
            person = _shapes(data.get("person"))
            obstacles = _shapes(data.get("obstacles"))
            for shape in person + obstacles:
                for value in vars(shape).values():
                    if not np.all(np.isfinite(value)):
                        raise ValueError(f"non-finite number in a {type(shape).__name__}")
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            raise ValueError(f"malformed scene shapes: {e}")
        if not person:
            raise ValueError("scene has no person shapes")
        return cls(person, obstacles)

    def person_head(self):
        spheres = [s for s in self.person if isinstance(s, Sphere)]
        if not spheres:
            raise ValueError("person has no sphere to serve as head")
        return max(spheres, key=lambda s: s.c[2]).c.copy()


def arm_capsules(chain, q, held):
    """Capsules per arm part at posture q (URDF root frame)."""
    out = {}
    for part, links in PART_LINKS.items():
        pts = [chain.link_pose(q, name)[0] for name in links]
        out[part] = [Capsule(p0, p1, RADII[part]) for p0, p1 in zip(pts[:-1], pts[1:])]
    ee, R = chain.fk(q, "ee")
    tool, _ = chain.fk(q, "tool")
    head = [Capsule(ee, tool, RADII["head"])]
    if held:
        c = ee + HELD_OFFSET * R[:, 2]
        head.append(Capsule(c, c, HELD_RADIUS))
    out["head"] = head
    return out


# ----------------------------------------------------------------------- vectorized clearances
# The capsule model above is evaluated for many postures and every (capsule, shape) pair at once.
# _gap_* reproduce _seg_seg / _seg_box (the scalar versions kept above as the specification), branch
# for branch and sample for sample, so the numbers agree to rounding.
_EPS = 1e-12
_LINK_NAMES = tuple(dict.fromkeys(n for links in PART_LINKS.values() for n in links))
_SEGMENTS = (  # (part column, first point, second point) in [links..., ee, tool, held centre] order
    *((PARTS.index(part), _LINK_NAMES.index(a), _LINK_NAMES.index(b))
      for part, links in PART_LINKS.items() for a, b in zip(links[:-1], links[1:])),
)
_EE, _TOOL, _HELD = len(_LINK_NAMES), len(_LINK_NAMES) + 1, len(_LINK_NAMES) + 2
_HEAD_COL = PARTS.index("head")
_CHUNK = 16  # postures per vectorized block: small enough that the work arrays stay in cache


def _gap_seg_seg(p1, q1, p2, q2):
    """_seg_seg over broadcast arrays (..., 3): the distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = np.einsum("...i,...i->...", d1, d1), np.einsum("...i,...i->...", d2, d2)
    f, c = np.einsum("...i,...i->...", d2, r), np.einsum("...i,...i->...", d1, r)
    b = np.einsum("...i,...i->...", d1, d2)
    a_le, e_le = a <= _EPS, e <= _EPS
    den = a * e - b * b
    s_gen = np.where(den > _EPS * a * e, np.clip((b * f - c * e) / den, 0, 1), 0.0)
    t_gen = (b * s_gen + f) / e
    s_low = np.clip(-c / a, 0, 1)
    low, high = t_gen < 0, t_gen > 1
    s_gen = np.where(low, s_low, np.where(high, np.clip((b - c) / a, 0, 1), s_gen))
    t_gen = np.where(low, 0.0, np.where(high, 1.0, t_gen))
    s = np.where(a_le, 0.0, np.where(e_le, s_low, s_gen))
    t = np.where(a_le, np.clip(f / e, 0, 1), np.where(e_le, 0.0, t_gen))
    sep = (p1 + s[..., None] * d1) - (p2 + t[..., None] * d2)
    return np.where(a_le & e_le, np.sqrt(np.einsum("...i,...i->...", r, r)), np.sqrt(np.einsum("...i,...i->...", sep, sep)))


def _gap_seg_box(a, b, center, half, R, rounds=6, n=17):
    """_seg_box over broadcast arrays: segments a, b (..., 1, 3) against boxes center/half (K, 3), R (K, 3, 3).
    The narrowing search runs the same `rounds` for every pair (a pair that reached 0 keeps it)."""
    shape = np.broadcast_shapes(a.shape[:-1], center.shape[:-1])
    lo, hi = np.zeros(shape), np.ones(shape)
    best = np.full(shape, np.inf)  # squared, until the end
    ramp = np.arange(n)
    # the sampled points a + t (b - a) in the box frame are linear in t: rotate the two ends once
    base = np.matmul((a - center)[..., None, :], R)[..., 0, :]
    slope = np.matmul((b - a)[..., None, :], R)[..., 0, :]
    bx, by, bz = (np.broadcast_to(base[..., k], shape)[..., None] for k in range(3))
    sx, sy, sz = (np.broadcast_to(slope[..., k], shape)[..., None] for k in range(3))
    hx, hy, hz = (half[:, k][:, None] for k in range(3))
    for _ in range(rounds):
        step = (hi - lo) / (n - 1)
        ts = lo[..., None] + ramp * step[..., None]
        ts[..., -1] = hi
        ox = np.maximum(np.abs(bx + ts * sx) - hx, 0.0)
        oy = np.maximum(np.abs(by + ts * sy) - hy, 0.0)
        oz = np.maximum(np.abs(bz + ts * sz) - hz, 0.0)
        d2 = ox * ox + oy * oy + oz * oz
        i = np.argmin(d2, axis=-1)[..., None]
        dmin = np.take_along_axis(d2, i, axis=-1)[..., 0]
        best = np.where(dmin < best, dmin, best)  # min(best, d): a NaN d never replaces best
        tmin = np.take_along_axis(ts, i, axis=-1)[..., 0]
        width = ts[..., 1] - ts[..., 0]
        lo, hi = np.maximum(0.0, tmin - width), np.minimum(1.0, tmin + width)
    return np.sqrt(best)


class _Pack:
    """A shape group as arrays: spheres and capsules as segments, boxes apart."""

    def __init__(self, shapes):
        segs = [_as_capsule(s) for s in shapes if not isinstance(s, Box)]
        boxes = [s for s in shapes if isinstance(s, Box)]
        self.p = np.array([c.a for c in segs]).reshape(-1, 3)
        self.q = np.array([c.b for c in segs]).reshape(-1, 3)
        self.r = np.array([c.r for c in segs], dtype=float)
        self.center = np.array([b.center for b in boxes]).reshape(-1, 3)
        self.half = np.array([b.half for b in boxes]).reshape(-1, 3)
        self.R = np.array([b.R for b in boxes]).reshape(-1, 3, 3)


def _arm_points(chain, Q, held):
    """(B, 8 or 9, 3) capsule end points per posture: the link origins, ee, tool and (when held)
    the held-object centre, in the order _SEGMENTS indexes."""
    links, sites, R = chain.points_batch(Q, _LINK_NAMES, ("ee", "tool"))
    pts = [*links, *sites]
    if held:
        pts.append(sites[0] + HELD_OFFSET * R[:, :, 2])
    return np.stack(pts, axis=1)


def clearances_batch(chain, scene, Q, held):
    """(B, 2, 4): clearances() of each posture of Q (B, 7), computed together (one FK per posture,
    all capsules against all shapes at once)."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[1] != 7:
        raise ValueError(f"Q must have shape (B, 7), got {Q.shape}")
    held = bool(held)
    segs = list(_SEGMENTS) + [(_HEAD_COL, _EE, _TOOL)]
    radii = [RADII[PARTS[col]] for col, _, _ in segs]
    if held:
        segs.append((_HEAD_COL, _HELD, _HELD))
        radii.append(HELD_RADIUS)
    cols = np.array([col for col, _, _ in segs])
    ia, ib = np.array([a for _, a, _ in segs]), np.array([b for _, _, b in segs])
    radii = np.array(radii)
    packs = [_Pack(scene.person), _Pack(scene.obstacles)]
    out = np.full((len(Q), 2, len(PARTS)), np.inf)
    with np.errstate(all="ignore"):
        for lo in range(0, len(Q), _CHUNK):
            pts = _arm_points(chain, Q[lo:lo + _CHUNK], held)
            A, Bp = pts[:, ia, :], pts[:, ib, :]  # (B, C, 3)
            for row, pack in enumerate(packs):
                parts = []
                if len(pack.r):
                    d = _gap_seg_seg(A[:, :, None, :], Bp[:, :, None, :], pack.p, pack.q)
                    parts.append(d - radii[None, :, None] - pack.r)  # (B, C, S): - cx.r - cy.r
                if len(pack.half):
                    d = _gap_seg_box(A[:, :, None, :], Bp[:, :, None, :], pack.center, pack.half, pack.R)
                    parts.append(d - radii[None, :, None])
                if not parts:
                    continue
                gap = np.concatenate(parts, axis=2).reshape(len(pts), len(segs), -1)
                for j in range(len(PARTS)):
                    out[lo:lo + _CHUNK, row, j] = gap[:, cols == j, :].reshape(len(pts), -1).min(axis=1)
    return out


def clearances(chain, scene, q, held):
    """(2, 4): row 0 distance to the person, row 1 to obstacles (+inf if none), per PARTS column.
    A NaN distance (a malformed shape) gives a NaN clearance."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"q must have shape (7,), got {q.shape}")
    return clearances_batch(chain, scene, q[None], held)[0]
