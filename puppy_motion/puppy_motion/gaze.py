"""Kinematic look-at helpers on top of Chain + compose.head_dq (pure numpy).

Head frame = the end-effector link (site 'ee'): forward = ee +z (the gaze), up = ee +y, left = ee +x.
A head offset is [dx, dy, dz, droll, dpitch, dyaw]: a world translation (m) and a pan-tilt-roll in
the head frame (rad) applied yaw -> pitch -> roll like a gimbal (compose.head_rotation). Head turns
pivot about the neck (the wrist centre), so a turn is wrist motion only.

Gravity turns (gravity_turn, aim) describe a turn in the world instead: pan about world z (+ to
the left seen from above), tilt about the horizontal axis across the gaze (+ looks down), roll
about the gaze. At home the face is upright and they equal yaw/pitch/roll; at the task's view and
grasp postures the face is rolled 90 deg, where a head-frame pitch would swing the gaze sideways.
"""

import numpy as np

from .compose import HEAD_SPEED_SITES, head_target, rot_err
from .urdf import SITE_OFFSETS

UP = np.array([0.0, 0.0, 1.0])
VERTICAL_GAZE_SIN = 1e-6  # gaze this close to vertical: tilt about the eye line instead
LOOK_AT_ITERS = 4
GIMBAL_COS = 1e-9
FALLBACK_AIM_M = 0.35  # with no target an aim point is this far from the eye along the gaze
SMOOTH_TRUNCATE = 4.0  # sigmas the smoothing kernel reaches


def rotation_about(axis, angle):
    """Rodrigues rotation matrix about a unit axis (a rotation vector axis * angle)."""
    axis = np.asarray(axis, dtype=float)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotvec_matrix(v):
    """Rotation matrix of a rotation vector."""
    v = np.asarray(v, dtype=float)
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-15 else rotation_about(v / a, a)


def rotation_angle(M):
    """The angle (rad) of a rotation matrix."""
    return float(np.linalg.norm(rot_err(M, np.eye(3))))


def rotation_vector(M):
    """The rotation vector of a rotation matrix."""
    return rot_err(M, np.eye(3))


def euler_yxz(R):
    """(yaw, pitch, roll) with R = Ry(yaw) Rx(pitch) Rz(roll) (compose.head_rotation's order)."""
    pitch = float(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
    if np.cos(pitch) > GIMBAL_COS:
        return float(np.arctan2(R[0, 2], R[2, 2])), pitch, float(np.arctan2(R[1, 0], R[1, 1]))
    return float(np.arctan2(-R[2, 0], R[0, 0])), pitch, 0.0  # gimbal lock: all of it in the yaw


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
    world = rotation_about(UP, pan) @ rotation_about(across_gaze(R), tilt)
    return R.T @ world @ R @ rotation_about([0.0, 0.0, 1.0], roll)


def elevation_azimuth(v):
    """(elevation, azimuth) of a world direction (rad)."""
    v = v / np.linalg.norm(v)
    return float(np.arcsin(np.clip(v[2], -1.0, 1.0))), float(np.arctan2(v[1], v[0]))


def wrap(angle):
    """An angle in (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def gravity_turn(chain, q, pan, tilt, roll=0.0):
    """(3,) head-offset rotation [roll, pitch, yaw] realising a gravity turn at posture q."""
    _, R = chain.fk(q, "ee")
    yaw, pitch, roll_h = euler_yxz(gravity_rotation(R, pan, tilt, roll))
    return np.array([roll_h, pitch, yaw])


def aim(chain, q, target):
    """(pan, tilt, unit direction) of the gravity turn that puts the gaze on `target`, re-aimed
    from where the eye lands after the turn (it swings with the head about the neck)."""
    p, R = chain.fk(q, "ee")
    eye_local = SITE_OFFSETS["eye"]
    el0, az0 = elevation_azimuth(R[:, 2])
    pan = tilt = 0.0
    for _ in range(LOOK_AT_ITERS):
        offset = np.concatenate([np.zeros(3), gravity_turn(chain, q, pan, tilt)])
        p_new, R_new = head_target(chain, p, R, offset)
        d = target - (p_new + R_new @ eye_local)
        el, az = elevation_azimuth(d)
        pan, tilt = wrap(az - az0), el0 - el
    return pan, tilt, d / np.linalg.norm(d)


def gaze_change(chain, q, pan, tilt):
    """Angle (rad) between the gaze at q and the gaze after a gravity pan and tilt."""
    _, R = chain.fk(q, "ee")
    g = R[:, 2]
    g_new = R @ gravity_rotation(R, pan, tilt, 0.0)[:, 2]
    return float(np.arccos(np.clip(g @ g_new, -1.0, 1.0)))


def look_at_offset(chain, q, target, lean_m=0.0, max_turn_rad=np.pi, mode="shortest"):
    """Head offset that turns the gaze (the ray from 'eye' along ee z) onto `target`, at most
    max_turn_rad off the current gaze, plus a lean of lean_m toward the target. The turn
    pivots about the neck. mode 'shortest': the least rotation (a head-frame pan and tilt);
    'pan_tilt': a gravity pan and tilt, like a camera on a pan-tilt head (natural for targets
    to the side or behind); 'pan': about world z only, which keeps a held container level."""
    if mode == "shortest":
        return _shortest_look(chain, q, target, lean_m, max_turn_rad)
    pan, tilt, d = aim(chain, q, target)
    tilt = 0.0 if mode == "pan" else tilt
    turn = gaze_change(chain, q, pan, tilt)
    if turn > max_turn_rad:
        pan, tilt = pan * max_turn_rad / turn, tilt * max_turn_rad / turn
    return np.concatenate([lean_m * d, gravity_turn(chain, q, pan, tilt)])


def _shortest_look(chain, q, target, lean_m, max_turn_rad):
    p, R = chain.fk(q, "ee")
    eye_local = SITE_OFFSETS["eye"]
    offset = np.zeros(6)
    for _ in range(LOOK_AT_ITERS):  # the eye swings with the turn: re-aim from where it lands
        p_new, R_new = head_target(chain, p, R, offset)
        eye = p_new + R_new @ eye_local
        d = (target - eye) / np.linalg.norm(target - eye)
        yaw, pitch = pan_tilt(R.T @ d)
        offset = np.array([*(lean_m * d), 0.0, pitch, yaw])
    turn = np.arccos(np.clip(np.cos(offset[4]) * np.cos(offset[5]), -1.0, 1.0))
    if turn > max_turn_rad:
        offset[4:] *= max_turn_rad / turn
    return offset


def head_travel(chain, q, dq):
    """Largest displacement (m) of the head points (eye, tool) between q and q + dq."""
    return max(np.linalg.norm(chain.fk(q + dq, s)[0] - chain.fk(q, s)[0]) for s in HEAD_SPEED_SITES)


def gaze_point(chain, q, distance_m=FALLBACK_AIM_M):
    """The point distance_m from the eye along the gaze at posture q: where a behavior aims when
    it has no target of its own (the wrist camera is already looking at the object)."""
    eye, R = chain.fk(q, "eye")
    return eye + distance_m * R[:, 2]


def gaussian_smooth(x, sigma_samples):
    """x (N, k) smoothed along axis 0 by a Gaussian of sigma_samples (edges repeat; the kernel
    reaches SMOOTH_TRUNCATE sigmas): scipy.ndimage.gaussian_filter1d's 'nearest' mode."""
    radius = int(SMOOTH_TRUNCATE * sigma_samples + 0.5)
    k = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (k / sigma_samples) ** 2)
    w /= w.sum()
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], w, mode="valid") for j in range(x.shape[1])], axis=1)
