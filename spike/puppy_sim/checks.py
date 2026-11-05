"""Safety checks on clip tracks, independent of how they were composed: the caps, precision
identity, the time-warp (against the functional track it warped, not the warped copy compose was
handed), clearances to the user and the obstacles (with the composite gripper), the gripper rule
and continuity. Plus the model's FK against the robot's recorded end-effector poses.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import layer, scene
from .kin import ARM_PARTS, VEL_LIMITS
from .timeline import DT, NEAR_USER, PRECISION, interp_rows, speed

CONTINUITY_TOL = 1.05  # one sample's step may be this much over URDF speed x DT (difference noise)
DTAU_TOL = 1e-9
PATH_TOL_RAD = 1e-9  # the functional track under the composite vs its source's path at tau
SAMPLE_SNAP = 1e-6  # a tau this close to a source sample's time (in samples) is that sample
WARP_ACCEL_CAP = layer.EXPR_JOINT_ACCEL  # rad/s^2 a pause or slow-in may add to the functional motion


@dataclass(frozen=True)
class Check:
    """One check on one track: the worst value seen against its limit."""

    name: str
    ok: bool
    shown: str  # compact value for the table
    detail: str = ""


def _ratio_check(name, values, limit, fmt="{:.2f}"):
    worst = float(np.max(values)) if np.size(values) else 0.0
    return Check(name, worst <= limit, f"{fmt.format(worst)}/{fmt.format(limit)}")


def check_track(track, kin):
    """All checks for a track; the expressive ones only for a composed (Puppy Mode) track."""
    out = _robot_checks(track, kin)
    if track.func is not None:
        out += _expressive_checks(track, kin)
        out += _precision_checks(track)
        out += _path_checks(track)
        out += _clearance_checks(track, kin)
        out.append(_gripper_check(track))
    return out


def _robot_checks(track, kin):
    """Joint speed under the URDF limits, no jumps, inside the joint ranges and RAMMP's guards."""
    v = np.abs(np.gradient(track.q, DT, axis=0))
    step = np.abs(np.diff(track.q, axis=0)) / (VEL_LIMITS * DT)
    lim = layer.guard_limits(kin)
    margin = lim - np.abs(track.q)
    worst = int(np.argmin(margin.min(axis=0)))
    return [
        _ratio_check("joint speed", (v / VEL_LIMITS).max(axis=1), 1.0),
        Check("continuity", bool(step.max() <= CONTINUITY_TOL), f"{step.max():.2f}/{CONTINUITY_TOL:.2f}"),
        Check(
            "guards",
            bool(margin.min() >= 0.0),
            f"{np.degrees(margin.min()):.0f} deg",
            f"least margin {np.degrees(margin.min()):.1f} deg on joint {worst + 1}",
        ),
    ]


def _expressive_checks(track, kin):
    """The expressive part alone: joint speed, joint acceleration, head speed, gripper speed."""
    expr = track.q - track.func.q
    v = np.gradient(expr, DT, axis=0)
    a = np.gradient(v, DT, axis=0)
    return [
        _ratio_check("expr speed", np.abs(v), layer.EXPR_JOINT_SPEED),
        _ratio_check("expr accel", np.abs(a), layer.EXPR_JOINT_ACCEL),
        _ratio_check("head speed", layer.expressive_head_speed(kin, track.func, track.q), layer.EXPR_HEAD_SPEED),
        _ratio_check("grip speed", layer.gripper_rate(track.grip - track.func.grip), layer.EXPR_GRIP_SPEED),
    ]


def _precision_checks(track):
    """Inside PRECISION: q bit-identical to the functional track, gain exactly 0, dtau/dt in [0, 1]."""
    prec = track.zone == PRECISION
    identical = track.q[prec].tobytes() == track.func.q[prec].tobytes() and bool(np.all(track.gain[prec] == 0.0))
    s = speed(track)
    s_prec = s[prec] if prec.any() else np.zeros(1)
    in_range = bool(np.all((s_prec >= -DTAU_TOL) & (s_prec <= 1.0 + DTAU_TOL)))
    return [
        Check("precision", identical, f"{'same' if identical else 'DIFF'} x{int(prec.sum())}"),
        Check("dtau/dt", in_range and bool(np.all(s <= 1.0 + DTAU_TOL)), f"{s.min():.2f}..{s.max():.2f}"),
    ]


def _path_checks(track):
    """For a time-warped track (its .func has a source): the functional track under the composite
    is the source's path at tau, every sample at a precise source sample is PRECISION, and the
    warp's pauses and slow-ins add at most WARP_ACCEL_CAP (|dq/dtau| |d2tau/dt2|) to the motion."""
    func, src = track.func, track.func.source
    if src is None:
        return []
    u = np.interp(func.tau, src.tau, np.arange(src.n, dtype=float))
    u = np.where(np.abs(u - np.round(u)) < SAMPLE_SNAP, np.round(u), u)
    off = float(np.abs(func.q - interp_rows(src.q, u)).max())
    lo, hi = np.floor(u).astype(int), np.ceil(u).astype(int)
    precise = (src.zone[lo] == PRECISION) | (src.zone[hi] == PRECISION)
    covered = bool(np.all(track.zone[precise] == PRECISION))
    v = interp_rows(np.gradient(src.q, DT, axis=0), u)
    added = (np.abs(v) * np.abs(np.gradient(np.gradient(func.tau, DT), DT))[:, None]).max(axis=1)
    ok = off <= PATH_TOL_RAD and covered
    detail = "" if ok else f"{off:.1e} rad off the source path; precise samples {'kept' if covered else 'NOT kept'}"
    return [Check("path", ok, "same" if ok else "OFF", detail), _ratio_check("warp accel", added, WARP_ACCEL_CAP)]


def _clearance_checks(track, kin):
    """Every arm part (gripper as composed) at or above its floor: person min(functional, 0.35 m)
    - 5 mm, obstacles min(functional, 3 cm)."""
    func_clear = kin.clearance_track(track.func)
    comp = layer.composite_clearances(kin, track.func, track.q, func_clear, track.grip)
    floor = layer.clearance_floor(func_clear)
    out = []
    for row, name in ((0, "user clr"), (1, "obstacle clr")):
        margin = comp[:, row] - floor[:, row]
        i, j = np.unravel_index(np.argmin(margin), margin.shape)
        out.append(
            Check(
                name,
                bool(margin.min() >= 0.0),
                f"{comp[i, row, j]:.3f}>={floor[i, row, j]:.3f}",
                f"tightest: {ARM_PARTS[j]} at {i * DT:.2f} s",
            )
        )
    return out


def _gripper_check(track):
    """No gripper expression while it holds something, near the user, or in PRECISION."""
    locked = track.held | (track.zone == NEAR_USER) | (track.zone == PRECISION)
    same = track.grip[locked].tobytes() == track.func.grip[locked].tobytes()
    return Check("gripper", same, "ok" if same else "MOVED")


def passed(checks):
    return all(c.ok for c in checks)


def fk_residuals(run, kin):
    """FK of the end-effector link at the run's joints vs the recorded /ee_state (arm_mount frame):
    (max position error m, mean position error m, max rotation error deg)."""
    q = np.column_stack([np.interp(run.ee_t, run.t, run.q[:, j]) for j in range(7)])
    pos, rot = [], []
    for qi, p_rec, quat in zip(q, run.ee_pos, run.ee_quat):
        p, R = kin.fk(qi, "ee")
        pos.append(np.linalg.norm(p - p_rec))
        rot.append(np.degrees(Rotation.from_matrix(R.T @ Rotation.from_quat(quat).as_matrix()).magnitude()))
    return float(np.max(pos)), float(np.mean(pos)), float(np.max(rot))


def format_table(rows):
    """rows: [(clip, track, duration, checks)] -> a compact fixed-width table."""
    names = []
    for *_, checks in rows:
        names += [c.name for c in checks if c.name not in names]
    header = ["clip", "track", "dur s", *names, "result"]
    body = []
    for clip, track, duration, checks in rows:
        shown = {c.name: c.shown + ("" if c.ok else " !") for c in checks}
        body.append(
            [clip, track, f"{duration:.1f}", *[shown.get(n, "-") for n in names], "PASS" if passed(checks) else "FAIL"]
        )
    widths = [max(len(str(r[k])) for r in [header, *body]) for k in range(len(header))]
    lines = ["  ".join(str(v).ljust(w) for v, w in zip(r, widths)) for r in [header, *body]]
    return "\n".join([lines[0], "  ".join("-" * w for w in widths), *lines[1:]])


def check_named(name):
    """Worker: build the registered clip `name` and check its tracks;
    [(clip, track label, duration, checks, composition log)]."""
    from .clips import registry

    clip = registry()[name]
    kin = scene.build(clip.run).kin()
    built = clip.build()
    tracks = built if isinstance(built, tuple) else (built,)
    labels = ("plain", "puppy") if len(tracks) == 2 else ("puppy" if tracks[0].func is not None else "plain",)
    return [(name, label, t.duration, check_track(t, kin), t.log) for label, t in zip(labels, tracks)]
