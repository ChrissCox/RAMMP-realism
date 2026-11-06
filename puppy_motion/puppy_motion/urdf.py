"""Serial-chain kinematics read from the URDF text the driver publishes."""
import xml.etree.ElementTree as ET

import numpy as np

# Offsets of each site from the end-effector frame, expressed in that frame.
SITE_OFFSETS = {
    "ee": np.zeros(3),
    "tool": np.array([0.0, 0.0, 0.120]),
    "eye": np.array([0.0, 0.065, 0.015]),
    "neck": np.array([0.0, 0.0, -0.1675]),
}
N_JOINTS = 7


def _floats(text, default):
    return np.array(text.split(), dtype=float) if text else np.array(default, dtype=float)


def _rpy_to_matrix(rpy):
    """URDF fixed-axis roll-pitch-yaw to a rotation matrix, R = Rz(y) Ry(p) Rx(r)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _skew(axis):
    return np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])


def _axis_angle(axis, angle):
    """Rodrigues rotation about a unit axis."""
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class _Joint:
    def __init__(self, name, origin_R, origin_p, axis, vel_limit, pos_limit, child=None):
        self.name = name
        self.child = child  # name of the child link
        self.origin_R = origin_R
        self.origin_p = origin_p
        self.axis = axis  # None for a fixed joint
        self.vel_limit = vel_limit
        self.pos_limit = pos_limit


class Chain:
    """Root link -> end-effector link, with seven actuated joints."""

    def __init__(self, path, movable):
        self._path = path          # every joint from the root to the ee link, in order
        self._movable = movable    # the seven actuated joints, in order
        self.joint_names = tuple(j.name for j in movable)
        self.vel_limits = np.array([j.vel_limit for j in movable])
        self.pos_limits = np.array([j.pos_limit for j in movable])

    @classmethod
    def from_urdf(cls, urdf_text):
        root = ET.fromstring(urdf_text)
        by_child = {}
        for el in root.findall("joint"):
            origin = el.find("origin")
            rpy = _floats(origin.get("rpy") if origin is not None else None, [0, 0, 0])
            xyz = _floats(origin.get("xyz") if origin is not None else None, [0, 0, 0])
            kind = el.get("type")
            axis = vel = None
            lim = [-np.inf, np.inf]
            if kind in ("revolute", "continuous"):
                axis_el = el.find("axis")
                axis = _floats(axis_el.get("xyz") if axis_el is not None else None, [1, 0, 0])
                axis = axis / np.linalg.norm(axis)
                limit = el.find("limit")
                if limit is not None:
                    vel = float(limit.get("velocity"))
                if kind == "revolute" and limit is not None:
                    lim = [float(limit.get("lower")), float(limit.get("upper"))]
            joint = _Joint(el.get("name"), _rpy_to_matrix(rpy), xyz, axis, vel, lim, el.find("child").get("link"))
            by_child[el.find("child").get("link")] = (el.find("parent").get("link"), joint)

        ee_links = [n for n in by_child if n.endswith("end_effector_link")]
        if len(ee_links) != 1:
            raise ValueError(f"expected one *end_effector_link, found {ee_links}")
        path = []
        link = ee_links[0]
        while link in by_child:
            link, joint = by_child[link]
            path.append(joint)
        path.reverse()

        movable = []
        for i in range(1, N_JOINTS + 1):
            match = [j for j in path if j.axis is not None and j.name.endswith(f"joint_{i}")]
            if len(match) != 1 or match[0].vel_limit is None:
                raise ValueError(f"expected one limited actuated joint named *joint_{i}")
            movable.append(match[0])
        if [j for j in path if j.axis is not None] != movable:
            raise ValueError("actuated joints on the chain are not joint_1..joint_7 in order")
        return cls(path, movable)

    def _forward(self, q):
        """Return the ee pose plus each actuated joint's world axis and origin."""
        q = np.asarray(q, dtype=float)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"q must have shape ({N_JOINTS},), got {q.shape}")
        R, p = np.eye(3), np.zeros(3)
        axes, origins = [], []
        k = 0
        for joint in self._path:
            p = p + R @ joint.origin_p
            R = R @ joint.origin_R
            if joint.axis is not None:
                axes.append(R @ joint.axis)
                origins.append(p.copy())
                R = R @ _axis_angle(joint.axis, q[k])
                k += 1
        return p, R, axes, origins

    @staticmethod
    def _site_offset(site):
        try:
            return SITE_OFFSETS[site]
        except KeyError:
            raise ValueError(f"unknown site {site!r}; expected one of {sorted(SITE_OFFSETS)}")

    def fk(self, q, site="ee"):
        """Position (3,) and rotation (3, 3) of a site in the URDF root frame."""
        offset = self._site_offset(site)
        p, R, _, _ = self._forward(q)
        return p + R @ offset, R

    def link_pose(self, q, link_name):
        """Position (3,) and rotation (3, 3) of a URDF link frame in the root frame. `link_name`
        is the full link name, or a suffix of it that names exactly one link on the chain
        (e.g. "half_arm_1_link")."""
        q = np.asarray(q, dtype=float)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"q must have shape ({N_JOINTS},), got {q.shape}")
        matches = [j for j in self._path if j.child == link_name]
        if not matches:
            matches = [j for j in self._path if j.child and j.child.endswith(link_name)]
        if len(matches) != 1:
            raise ValueError(f"{link_name!r} does not name exactly one link on the chain")
        R, p = np.eye(3), np.zeros(3)
        k = 0
        for joint in self._path:
            p = p + R @ joint.origin_p
            R = R @ joint.origin_R
            if joint.axis is not None:
                R = R @ _axis_angle(joint.axis, q[k])
                k += 1
            if joint is matches[0]:
                return p, R
        raise AssertionError("unreachable")

    def _resolve_link(self, link_name):
        """Index into the path of the joint whose child is `link_name` (link_pose's matching rule)."""
        matches = [i for i, j in enumerate(self._path) if j.child == link_name]
        if not matches:
            matches = [i for i, j in enumerate(self._path) if j.child and j.child.endswith(link_name)]
        if len(matches) != 1:
            raise ValueError(f"{link_name!r} does not name exactly one link on the chain")
        return matches[0]

    def points_batch(self, Q, links=(), sites=()):
        """One forward-kinematics sweep over B postures Q (B, 7): the positions (B, 3) of each link
        frame in `links` (as link_pose names them) and of each site in `sites` (as fk names them),
        plus the end-effector rotations (B, 3, 3). Returns (link_points, site_points, R_ee) with
        the points stacked as (len, B, 3); equal to link_pose / fk row by row."""
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[1] != N_JOINTS:
            raise ValueError(f"Q must have shape (B, {N_JOINTS}), got {Q.shape}")
        want = {}
        for i, name in enumerate(links):
            want.setdefault(self._resolve_link(name), []).append(i)
        offsets = [self._site_offset(s) for s in sites]
        last = max(want, default=-1) if not sites else len(self._path) - 1
        n = len(Q)
        cos, sin = np.cos(Q), np.sin(Q)
        R, p = np.broadcast_to(np.eye(3), (n, 3, 3)), np.zeros((n, 3))
        link_pts = np.empty((len(links), n, 3))
        k = 0
        for idx, joint in enumerate(self._path[: last + 1]):
            p = p + R @ joint.origin_p
            R = R @ joint.origin_R
            if joint.axis is not None:
                K = _skew(joint.axis)
                R = R @ (np.eye(3) + sin[:, k, None, None] * K + (1 - cos[:, k])[:, None, None] * (K @ K))
                k += 1
            for i in want.get(idx, ()):
                link_pts[i] = p
        site_pts = np.empty((len(sites), n, 3))
        for i, off in enumerate(offsets):
            site_pts[i] = p + R @ off
        return link_pts, site_pts, R

    def jacobian(self, q, site="ee"):
        """6x7 world-frame Jacobian of a site: linear rows first, then angular."""
        offset = self._site_offset(site)
        p, R, axes, origins = self._forward(q)
        p_site = p + R @ offset
        J = np.zeros((6, N_JOINTS))
        for i, (z, o) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(z, p_site - o)
            J[3:, i] = z
        return J
