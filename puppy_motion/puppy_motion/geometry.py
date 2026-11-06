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


def clearances(chain, scene, q, held):
    """(2, 4): row 0 distance to the person, row 1 to obstacles (+inf if none), per PARTS column.
    A NaN distance (a malformed shape) gives a NaN clearance."""
    caps = arm_capsules(chain, q, held)
    out = np.full((2, len(PARTS)), np.inf)
    for j, part in enumerate(PARTS):
        for row, shapes in enumerate((scene.person, scene.obstacles)):
            for c in caps[part]:
                for s in shapes:
                    d = distance(c, s)
                    # a NaN distance (a malformed shape) must not be dropped by min(): it propagates
                    out[row, j] = np.nan if np.isnan(d) or np.isnan(out[row, j]) else min(out[row, j], d)
    return out
