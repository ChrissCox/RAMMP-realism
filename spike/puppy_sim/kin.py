"""Kinematics on the scene's MuJoCo model: FK, Jacobians, damped-least-squares IK (poses, head
offsets, look-at) and clearance queries. Pure computation: nothing here talks to a robot.

Head frame = the end-effector link (site 'ee'): forward = ee +z (the gaze), up = ee +y (the eyes
sit above the gripper), left = ee +x. A head offset is [dx, dy, dz, droll, dpitch, dyaw]: a world
translation (m) and a pan-tilt-roll in the head frame (rad), applied in that order like a gimbal:
yaw about the head's up (+ looks to the face's left), then pitch about the turned eye line (+ looks
down), then roll about the resulting gaze (+ tips the head to its right). Offsets from different
behaviors add component-wise, and a roll always tilts about wherever the head is looking.

Head turns pivot about the neck: the Gen3's wrist centre, where the joint 5, 6 and 7 axes meet
(0.167 m behind the end-effector origin, on its z axis). Turning about it is wrist motion only, so
the shoulder and elbow stay put; a turn about the end-effector origin would swing the elbow.

Gravity turns (gravity_turn, aim) describe a turn in the world instead: pan about world z (+ to
the left seen from above), tilt about the horizontal axis across the gaze (+ looks down), roll
about the gaze. At home the face is upright and they equal yaw/pitch/roll; at the task's view and
grasp postures the face is rolled 90 deg, where a head-frame pitch would swing the gaze sideways.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import curves
from .timeline import DT, GRIP_TO_KNUCKLE_RAD

ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
VEL_LIMITS = np.array([1.3963] * 4 + [1.2218] * 3)  # URDF arm_joint_1..7 velocity limits, rad/s
# RAMMP's guards (rad): |joint_4| <= 144 deg, |joint_6| <= 115 deg, joint_2 within the URDF +-2.24
GUARD_LIMITS = {1: 2.24, 3: np.radians(144.0), 5: np.radians(115.0)}  # 0-based joint index -> |q| max
NECK_BODY = "spherical_wrist_2_link"  # its origin is the wrist centre (joints 5-7 meet within 0.2 mm)
DLS_DAMPING = 0.03
HEAD_IK_ITERS = 4
# Head offsets are the neck's job: a joint's motion costs this much (joints 1..7), so turns stay in
# the (spherical) wrist and the shoulder and elbow move only for translations (a breath, a lean).
HEAD_JOINT_WEIGHTS = np.array([25.0, 25.0, 15.0, 8.0, 1.0, 1.0, 1.0])
IK_ITERS = 200
IK_TOL = 1e-6
IK_STEP_MAX_RAD = 0.2  # largest joint step per IK iteration
LOOK_AT_ITERS = 4
CLEARANCE_CUTOFF_M = 0.6  # distances beyond this read as the cutoff
TOUCH_M = 1e-4  # pairs closer than this count as touching
DISTANCE_BISECTIONS = 13  # ~0.07 mm on a 0.6 m cutoff
ARM_PARTS = ("upper_arm", "forearm", "wrist", "head")  # scene.ARM_PART_BODIES; the head carries a held container
UP = np.array([0.0, 0.0, 1.0])
VERTICAL_GAZE_SIN = 1e-6  # gaze this close to vertical: tilt about the eye line instead
HEAD_TRAVEL_SITES = ("eye", "tool")  # the head points whose travel paces a move (layer.HEAD_SPEED_SITES)


@dataclass
class RobotOnly:
    """A model without the scene's props: FK/IK only (no gripper posing, no clearance groups)."""

    model: mujoco.MjModel
    linkage: dict | None = None
    groups: dict = field(default_factory=dict)


def dls(J, err, damping=DLS_DAMPING, weights=None):
    """Damped least squares step W^-1 J^T (J W^-1 J^T + damping^2 I)^-1 err (W = diag(weights))."""
    w_inv = np.ones(J.shape[1]) if weights is None else 1.0 / np.asarray(weights)
    JW = J * w_inv
    return JW.T @ np.linalg.solve(JW @ J.T + damping**2 * np.eye(J.shape[0]), err)


def rot_err(R_target, R):
    """World-frame rotation vector taking R to R_target."""
    return Rotation.from_matrix(R_target @ R.T).as_rotvec()


def head_rotation(offset):
    """The head-frame rotation of a head offset: yaw (about y), then pitch (x), then roll (z)."""
    roll, pitch, yaw = offset[3], offset[4], offset[5]
    return Rotation.from_euler("YXZ", [yaw, pitch, roll]).as_matrix()


def pan_tilt(direction):
    """(yaw, pitch) that turn the head's gaze (+z) onto a head-frame direction."""
    x, y, z = direction
    return np.arctan2(x, z), np.arctan2(-y, np.hypot(x, z))


def across_gaze(R):
    """The horizontal axis across the gaze of ee rotation R: a positive turn about it tips the
    gaze down. Falls back to the eye line (ee x) when the gaze is vertical."""
    across = np.cross(UP, R[:, 2])
    norm = np.linalg.norm(across)
    return across / norm if norm > VERTICAL_GAZE_SIN else R[:, 0]


def gravity_rotation(R, pan, tilt, roll):
    """Head-frame rotation (3x3) of a gravity turn at ee rotation R: tilt about the horizontal
    axis across the gaze, then pan about world z, then roll about the new gaze."""
    world = Rotation.from_rotvec(pan * UP) * Rotation.from_rotvec(tilt * across_gaze(R))
    return R.T @ world.as_matrix() @ R @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()


def elevation_azimuth(v):
    """(elevation, azimuth) of a world direction (rad)."""
    v = v / np.linalg.norm(v)
    return float(np.arcsin(np.clip(v[2], -1.0, 1.0))), float(np.arctan2(v[1], v[0]))


def wrap(angle):
    """An angle in (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class Kin:
    """FK/IK/clearance on one MjData. Build with Kin(scene) (full scene) or Kin.robot_only(model)."""

    def __init__(self, scene):
        self.scene = scene
        self.m = scene.model
        self.d = mujoco.MjData(self.m)
        ids = [self.m.joint(j).id for j in ARM_JOINTS]
        self.qadr = self.m.jnt_qposadr[ids]
        self.dadr = self.m.jnt_dofadr[ids]
        limited = self.m.jnt_limited[ids].astype(bool)
        self.q_lo = np.where(limited, self.m.jnt_range[ids, 0], -np.inf)
        self.q_hi = np.where(limited, self.m.jnt_range[ids, 1], np.inf)
        self._grip_adr = self._gripper_addresses()
        self._jp, self._jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        p, R = self.fk(np.zeros(7), "ee")
        self.neck_local = R.T @ (self.d.xpos[self.m.body(NECK_BODY).id] - p)

    @classmethod
    def robot_only(cls, model):
        """FK/IK on a model without the scene's props (no gripper posing, no clearance)."""
        return cls(RobotOnly(model))

    def _gripper_addresses(self):
        """qpos addresses of the 2F-85 joints by role ('driver', ...), both sides."""
        out = {}
        for role in ("driver", "coupler", "spring_link", "follower"):
            names = [f"g_{side}_{role}_joint" for side in ("left", "right")]
            found = [n for n in names if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
            out[role] = [self.m.joint(n).qposadr[0] for n in found]
        return out

    # ------------------------------------------------------------------------------ posing
    def pose(self, q, grip=0.0, held=False):
        """Write the arm, the gripper linkage and the container into the model and run FK."""
        self.d.qpos[self.qadr] = q
        self._pose_gripper(grip)
        if "container" in self.scene.groups:
            self._pose_container(held)
        mujoco.mj_kinematics(self.m, self.d)
        return self

    def _pose_gripper(self, grip):
        table = self.scene.linkage
        if table is None:
            return
        knuckle = float(np.clip(grip, 0.0, 1.0)) * GRIP_TO_KNUCKLE_RAD
        for role, adrs in self._grip_adr.items():
            self.d.qpos[adrs] = knuckle if role == "driver" else np.interp(knuckle, table["driver"], table[role])

    def _pose_container(self, held):
        """Resting on the counter, or riding rigidly in the tool frame when held."""
        sc = self.scene
        mocap = self.m.body("container").mocapid[0]
        if not held:
            self.d.mocap_pos[mocap] = sc.placement.center
            self.d.mocap_quat[mocap] = sc.container_quat
            return
        mujoco.mj_kinematics(self.m, self.d)
        p, R = self.site("tool")
        pos_in_tool, quat_in_tool = sc.carry_in_tool
        tool_quat, quat = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(tool_quat, R.ravel())
        mujoco.mju_mulQuat(quat, tool_quat, quat_in_tool)
        self.d.mocap_pos[mocap] = p + R @ pos_in_tool
        self.d.mocap_quat[mocap] = quat

    def site(self, name):
        """(position, rotation matrix) of a site at the current pose."""
        sid = self.m.site(name).id
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def fk(self, q, site="ee"):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.site(site)

    def jac(self, site="ee"):
        """(6, 7) Jacobian [linear; angular] of a site, world frame, at the current pose."""
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.m.site(site).id)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def fk_track(self, q, site="ee"):
        """Positions (N, 3) and rotations (N, 3, 3) of a site along joint rows q."""
        P, R = np.empty((len(q), 3)), np.empty((len(q), 3, 3))
        for i, qi in enumerate(q):
            P[i], R[i] = self.fk(qi, site)
        return P, R

    # ------------------------------------------------------------------------------------ IK
    def clamp(self, q):
        """q inside the joint ranges and RAMMP's guards."""
        q = np.clip(q, self.q_lo, self.q_hi)
        for j, lim in GUARD_LIMITS.items():
            q[j] = np.clip(q[j], -lim, lim)
        return q

    def ik(self, q_seed, p_target, R_target, site="tool"):
        """DLS IK to a site pose from q_seed. Returns (q, position error m, rotation error rad)."""
        q = np.array(q_seed, dtype=float)
        for _ in range(IK_ITERS):
            p, R = self.fk(q, site)
            err = np.concatenate([p_target - p, rot_err(R_target, R)])
            if np.linalg.norm(err) < IK_TOL:
                break
            step = dls(self.jac(site), err)
            q = self.clamp(q + step * min(1.0, IK_STEP_MAX_RAD / np.abs(step).max()))
        p, R = self.fk(q, site)
        return q, float(np.linalg.norm(p_target - p)), float(np.linalg.norm(rot_err(R_target, R)))

    def cartesian_path(self, q0, p1, R1, duration_s, site="tool"):
        """Joint rows (N, 7) moving `site` on a straight line from its pose at q0 to (p1, R1),
        orientation slerped, min-jerk timed; IK at every sample, warm-started from the last.
        Returns (q, worst position error m)."""
        p0, R0 = self.fk(q0, site)
        s = curves.min_jerk(np.linspace(0.0, 1.0, int(round(duration_s / DT)) + 1))
        turn = Slerp([0.0, 1.0], Rotation.from_matrix([R0, R1]))
        q, rows, worst = np.asarray(q0, dtype=float), [], 0.0
        for si in s:
            q, err, _ = self.ik(q, p0 + si * (np.asarray(p1) - p0), turn(si).as_matrix(), site)
            rows.append(q)
            worst = max(worst, err)
        return np.array(rows), worst

    def head_target(self, p, R, offset):
        """The ee pose after a head offset at ee pose (p, R): turned about the neck, then shifted."""
        R_new = R @ head_rotation(offset)
        neck = p + R @ self.neck_local
        return neck - R_new @ self.neck_local + offset[:3], R_new

    def head_dq(self, q, offset, iters=HEAD_IK_ITERS):
        """Joint offset that realises a head offset at posture q: DLS on the ee Jacobian at q,
        iterated so the achieved offset matches. Zero offset gives exactly zero."""
        dq = np.zeros(7)
        if not np.any(offset):
            return dq
        p_t, R_t = self.head_target(*self.fk(q, "ee"), offset)
        for _ in range(iters):
            p, R = self.fk(q + dq, "ee")
            dq = dq + dls(self.jac("ee"), np.concatenate([p_t - p, rot_err(R_t, R)]), weights=HEAD_JOINT_WEIGHTS)
        return dq

    def look_at_offset(self, q, target, lean_m=0.0, max_turn_rad=np.pi, mode="shortest"):
        """Head offset that turns the gaze (the ray from 'eye' along ee z) onto `target`, at most
        max_turn_rad off the current gaze, plus a lean of lean_m toward the target. The turn
        pivots about the neck. mode 'shortest': the least rotation (a head-frame pan and tilt);
        'pan_tilt': a gravity pan and tilt, like a camera on a pan-tilt head (natural for targets
        to the side or behind); 'pan': about world z only, which keeps a held container level."""
        if mode == "shortest":
            return self._shortest_look(q, target, lean_m, max_turn_rad)
        pan, tilt, d = self.aim(q, target)
        tilt = 0.0 if mode == "pan" else tilt
        turn = self.gaze_change(q, pan, tilt)
        if turn > max_turn_rad:
            pan, tilt = pan * max_turn_rad / turn, tilt * max_turn_rad / turn
        return np.concatenate([lean_m * d, self.gravity_turn(q, pan, tilt)])

    def _shortest_look(self, q, target, lean_m, max_turn_rad):
        p, R = self.fk(q, "ee")
        eye_local = R.T @ (self.site("eye")[0] - p)
        offset = np.zeros(6)
        for _ in range(LOOK_AT_ITERS):  # the eye swings with the turn: re-aim from where it lands
            p_new, R_new = self.head_target(p, R, offset)
            eye = p_new + R_new @ eye_local
            d = (target - eye) / np.linalg.norm(target - eye)
            yaw, pitch = pan_tilt(R.T @ d)
            offset = np.array([*(lean_m * d), 0.0, pitch, yaw])
        turn = np.arccos(np.clip(np.cos(offset[4]) * np.cos(offset[5]), -1.0, 1.0))
        if turn > max_turn_rad:
            offset[4:] *= max_turn_rad / turn
        return offset

    def gravity_turn(self, q, pan, tilt, roll=0.0):
        """(3,) head-offset rotation [roll, pitch, yaw] realising a gravity turn at posture q."""
        _, R = self.fk(q, "ee")
        yaw, pitch, roll_h = Rotation.from_matrix(gravity_rotation(R, pan, tilt, roll)).as_euler("YXZ")
        return np.array([roll_h, pitch, yaw])

    def aim(self, q, target):
        """(pan, tilt, unit direction) of the gravity turn that puts the gaze on `target`, re-aimed
        from where the eye lands after the turn (it swings with the head about the neck)."""
        p, R = self.fk(q, "ee")
        eye_local = R.T @ (self.site("eye")[0] - p)
        el0, az0 = elevation_azimuth(R[:, 2])
        pan = tilt = 0.0
        for _ in range(LOOK_AT_ITERS):
            offset = np.concatenate([np.zeros(3), self.gravity_turn(q, pan, tilt)])
            p_new, R_new = self.head_target(p, R, offset)
            d = target - (p_new + R_new @ eye_local)
            el, az = elevation_azimuth(d)
            pan, tilt = wrap(az - az0), el0 - el
        return pan, tilt, d / np.linalg.norm(d)

    def gaze_change(self, q, pan, tilt):
        """Angle (rad) between the gaze at q and the gaze after a gravity pan and tilt."""
        _, R = self.fk(q, "ee")
        g = R[:, 2]
        g_new = R @ gravity_rotation(R, pan, tilt, 0.0)[:, 2]
        return float(np.arccos(np.clip(g @ g_new, -1.0, 1.0)))

    def head_travel(self, q, dq):
        """Largest displacement (m) of the head points (eye, tool) between q and q + dq."""
        return max(np.linalg.norm(self.fk(q + dq, s)[0] - self.fk(q, s)[0]) for s in HEAD_TRAVEL_SITES)

    # ----------------------------------------------------------------------------- clearance
    def geom_distance(self, ga, gb, distmax, fromto):
        """mj_geomDistance, guarded. It sometimes returns 0 for well separated mesh/box pairs when
        distmax exceeds their distance, but its "farther than D" answer is reliable: a 0 from a
        pair that is not touching is replaced by bisecting on that answer, keeping the lower
        bracket (never more than the true distance; fromto then comes from a call just above it,
        or is zeroed)."""
        dist = mujoco.mj_geomDistance(self.m, self.d, ga, gb, distmax, fromto)
        if dist > 0.0 or mujoco.mj_geomDistance(self.m, self.d, ga, gb, TOUCH_M, None) < TOUCH_M:
            return dist
        lo, hi = TOUCH_M, distmax
        for _ in range(DISTANCE_BISECTIONS):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if mujoco.mj_geomDistance(self.m, self.d, ga, gb, mid, None) < mid else (mid, hi)
        if mujoco.mj_geomDistance(self.m, self.d, ga, gb, hi, fromto) <= 0.0:
            fromto[:] = 0.0
        return lo

    def closest(self, geoms_a, geoms_b, cutoff=CLEARANCE_CUTOFF_M):
        """(distance, geom of a, point on a, point on b) of the closest pair at the current pose;
        (cutoff, -1, None, None) when nothing is within cutoff. Bounding-sphere lower bounds first,
        exact distances only where they could beat the best so far."""
        best = (cutoff, -1, None, None)
        if len(geoms_a) == 0 or len(geoms_b) == 0:
            return best
        xa, xb = self.d.geom_xpos[geoms_a], self.d.geom_xpos[geoms_b]
        rb_a, rb_b = self.m.geom_rbound[geoms_a], self.m.geom_rbound[geoms_b]
        lower = np.linalg.norm(xa[:, None] - xb[None], axis=2) - rb_a[:, None] - rb_b[None]
        fromto = np.zeros(6)
        for flat in np.argsort(lower, axis=None):
            i, j = np.unravel_index(flat, lower.shape)
            if lower[i, j] >= best[0]:
                break
            dist = self.geom_distance(int(geoms_a[i]), int(geoms_b[j]), best[0], fromto)
            if dist < best[0]:
                best = (dist, int(geoms_a[i]), fromto[:3].copy(), fromto[3:].copy())
        return best

    def part_geoms(self, part, held):
        """Geoms of an arm part; the head includes the container while it is held."""
        g = self.scene.groups
        return np.concatenate([g[part], g["container"]]) if held and part == "head" else g[part]

    def obstacle_geoms(self, held):
        """Counter, panel, chair, and the container while it rests on the counter."""
        g = self.scene.groups
        return np.concatenate([g["counter"], g["panel"], g["chair"]] + ([] if held else [g["container"]]))

    def _targets(self, held):
        """The two things the arm keeps clear of: (person geoms, obstacle geoms)."""
        return self.scene.groups["person"], self.obstacle_geoms(held)

    def clearances(self, q, grip, held):
        """(2, len(ARM_PARTS)) m: row 0 to the person, row 1 to the obstacles, per arm part."""
        self.pose(q, grip, held)
        out = np.empty((2, len(ARM_PARTS)))
        for row, target in enumerate(self._targets(held)):
            for j, part in enumerate(ARM_PARTS):
                out[row, j] = self.closest(self.part_geoms(part, held), target)[0]
        return out

    def clearance_track(self, track):
        """(N, 2, len(ARM_PARTS)) clearances along a track (see clearances)."""
        return np.array([self.clearances(q, g, h) for q, g, h in zip(track.q, track.grip, track.held)])

    def clearance_gradients(self, q, grip, held):
        """(2, len(ARM_PARTS), 7): d(clearance)/dq per target (person, obstacles) and arm part, at
        posture q. Zero where the pair is beyond the cutoff or already touching."""
        self.pose(q, grip, held)
        mujoco.mj_comPos(self.m, self.d)
        out = np.zeros((2, len(ARM_PARTS), 7))
        for row, target in enumerate(self._targets(held)):
            for j, part in enumerate(ARM_PARTS):
                dist, geom, pa, pb = self.closest(self.part_geoms(part, held), target)
                if geom >= 0 and dist > TOUCH_M and np.linalg.norm(pa - pb) > TOUCH_M:
                    out[row, j] = self._point_gradient(geom, pa, (pa - pb) / np.linalg.norm(pa - pb))
        return out

    def _point_gradient(self, geom, point, normal):
        """How fast `point` (on geom's body) moves along `normal` per unit joint motion (7,). A held
        container is a mocap body riding on the end effector: its points move with 'ee'."""
        body = self.m.geom_bodyid[geom]
        if body == self.m.body("container").id:
            body = self.m.body("ee").id
        mujoco.mj_jac(self.m, self.d, self._jp, None, point, body)
        return normal @ self._jp[:, self.dadr]
