"""Safety checks on clip tracks, independent of how they were composed: the zone-tiered caps of the
track's preset, precision identity, the time-warp (against the functional track it warped, not the
warped copy compose was handed) and the time it adds, clearances to the user and the obstacles
(with the composite gripper), the gripper rule and continuity. Plus the model's FK against the
robot's recorded end-effector poses.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import layer, scene
from .kin import ARM_PARTS, VEL_LIMITS
from .timeline import BOUNDARY, DT, FREE, NEAR_USER, PRECISION, STOP_RAD_S, interp_rows, joint_speed, speed

CONTINUITY_TOL = 1.05  # one sample's step may be this much over URDF speed x DT (difference noise)
CAP_TOL = 1e-9
DTAU_TOL = 1e-9
PATH_TOL_RAD = 1e-9  # the functional track under the composite vs its source's path at tau
SAMPLE_SNAP = 1e-6  # a tau this close to a source sample's time (in samples) is that sample


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


def _cap_check(name, value, cap, where):
    """value (N, k) against cap (N, k) on the samples `where`: the worst sample's value/cap, '-'
    when there are none."""
    if not where.any():
        return Check(name, True, "-")
    ratio = value[where] / cap[where]
    i, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    return Check(name, bool(ratio[i, j] <= 1.0 + CAP_TOL), f"{value[where][i, j]:.2f}/{cap[where][i, j]:.2f}")


def check_track(track, kin):
    """All checks for a track; the expressive ones only for a composed (Puppy Mode) track."""
    out = _robot_checks(track, kin)
    if track.func is not None:
        out += _expressive_checks(track, kin)
        out += _precision_checks(track)
        out += _path_checks(track)
        out += _clearance_checks(track, kin)
        out.append(_gripper_check(track))
        out.append(_tier_check(track))
        out += _time_checks(track)
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
    """The expressive part alone under the caps of the track's preset, tier by tier (layer.cap_tier):
    joint speed, joint acceleration and head speed where FREE's caps apply ('free ...') and where
    the gentle ones do ('near ...': NEAR_USER and the ramps into it or into PRECISION); and the
    gripper speed."""
    caps = track.caps or layer.DEFAULT_CAPS
    limits = layer.sample_caps(track.zone, caps)
    expr = track.q - track.func.q
    v = np.abs(np.gradient(expr, DT, axis=0))
    a = np.abs(np.gradient(np.gradient(expr, DT, axis=0), DT, axis=0))
    head = layer.expressive_head_speed(kin, track.func, track.q)[:, None]
    out = [Check("caps", True, caps)]
    for tier, where in (("free", limits.tier == FREE), ("near", limits.tier != FREE)):
        out += [
            _cap_check(f"{tier} speed", v, limits.joint_speed, where),
            _cap_check(f"{tier} accel", a, limits.joint_accel, where),
            _cap_check(f"{tier} head", head, limits.head_speed[:, None], where),
        ]
    out.append(_ratio_check("grip speed", layer.gripper_rate(track.grip - track.func.grip), layer.EXPR_GRIP_SPEED))
    return out


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
    warp's pauses and slow-ins add (|dq/dtau| |d2tau/dt2|) at most the default preset's joint
    acceleration cap of each sample's tier to the motion."""
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
    added = np.abs(v) * np.abs(np.gradient(np.gradient(func.tau, DT), DT))[:, None]
    ok = off <= PATH_TOL_RAD and covered
    detail = "" if ok else f"{off:.1e} rad off the source path; precise samples {'kept' if covered else 'NOT kept'}"
    accel_cap = layer.sample_caps(track.zone).joint_accel
    return [
        Check("path", ok, "same" if ok else "OFF", detail),
        _cap_check("warp accel", added, accel_cap, np.ones(track.n, dtype=bool)),
    ]


def _tier_check(track):
    """What a composed track does to the functional moves: 'boundary' when every moving sample of
    the source (a joint at STOP_RAD_S or faster) plays exactly as recorded, at dtau/dt = 1 on both
    sides and with no expressive offset; else how many it retimes and how many it layers onto. A
    track that promises the boundary tier (track.tier) fails if it does either."""
    func = track.func
    src = func.source if func.source is not None else func
    u = np.interp(func.tau, src.tau, np.arange(src.n, dtype=float))
    moving = joint_speed(src) >= STOP_RAD_S
    on_move = moving[np.floor(u).astype(int)] | moving[np.ceil(u).astype(int)]
    step = np.abs(np.diff(u) - 1.0) <= SAMPLE_SNAP
    steady = np.append(step, True) & np.insert(step, 0, True)
    retimed = int(np.sum(on_move & ~steady))
    layered = int(np.sum(on_move & np.any(track.q != func.q, axis=1)))
    shown = "boundary" if not (retimed or layered) else f"moves: {retimed} retimed, {layered} layered"
    promised = track.tier == BOUNDARY
    return Check(
        "tier", not (promised and (retimed or layered)), shown, f"promised: {track.tier}" if track.tier else ""
    )


def _time_checks(track):
    """For a time-warped track: the time its pauses add to the plain task, within its budget when it
    has one (a directed track)."""
    if track.func.source is None:
        return []
    added = track.duration - track.plain_duration
    budget = track.budget
    ok = budget is None or added <= budget.seconds + CAP_TOL
    limit = f"/{budget.seconds:.1f} {budget.name}" if budget is not None else ""
    return [Check("added s", ok, f"+{added:.1f}{limit}")]


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


def replay_checks(plain, puppy):
    """The checks of a replay.direct_run track against the plain track it came from: every spliced
    beat passes puppy_motion.validate.check_beat at the posture it was given (what the hook would
    send), sits in the track as composed, the PRECISION samples play exactly as recorded, the rest
    of the path is the plain path at tau, and the beats' time is within the preset's budget."""
    from puppy_motion.validate import check_beat

    chain, person = scene.motion_chain(plain.name), scene.person_scene()
    found = []
    for sp in puppy.spliced:
        b = sp.beat
        bad = check_beat(b.points, sp.q_now, chain, person, b.zone, puppy.caps or "default", held=sp.held)
        found += [f"{b.name} at {sp.i0 * DT:.2f} s: {v.rule}: {v.detail}" for v in bad]
    out = [Check("beats", not found, f"{len(puppy.spliced) - len(found)}/{len(puppy.spliced)} valid", "; ".join(found))]
    in_track = all(
        np.array_equal(puppy.q[sp.i0 : sp.i0 + len(sp.beat.points)], np.array([p[1] for p in sp.beat.points]))
        for sp in puppy.spliced
    )
    out.append(Check("splice", in_track, "as composed" if in_track else "ALTERED"))
    u = puppy.tau / DT
    on_path = interp_rows(plain.q, u)
    beat_span = np.zeros(puppy.n, dtype=bool)
    for sp in puppy.spliced:
        beat_span[sp.i0 : sp.i0 + len(sp.beat.points)] = True
    prec = puppy.zone == PRECISION
    whole = np.abs(u - np.round(u)) < SAMPLE_SNAP
    exact = bool(np.array_equal(puppy.q[prec & whole], plain.q[np.round(u[prec & whole]).astype(int)]))
    lo, hi = np.floor(u + SAMPLE_SNAP).astype(int), np.ceil(u - SAMPLE_SNAP).astype(int)
    hi = np.clip(hi, 0, plain.n - 1)
    played = bool(np.all(puppy.zone[(plain.zone[lo] == PRECISION) | (plain.zone[hi] == PRECISION)] == PRECISION))
    ok = exact and played and not (prec & beat_span).any() and bool(np.abs(puppy.q - on_path)[prec].max(initial=0.0) <= PATH_TOL_RAD)
    out.append(Check("precision", ok, "same" if ok else "DIFF", "" if ok else f"exact {exact}, precise samples kept {played}"))
    off = float(np.abs(puppy.q - on_path)[~beat_span].max())
    out.append(Check("path", off <= PATH_TOL_RAD, "same" if off <= PATH_TOL_RAD else "OFF", f"{off:.1e} rad off the plain path"))
    beat_s = sum(sp.beat.duration for sp in puppy.spliced)
    limit = puppy.budget.seconds if puppy.budget is not None else float("inf")
    out.append(
        Check("beat s", beat_s <= limit + DT + CAP_TOL, f"{beat_s:.1f}/{limit:.1f} {puppy.budget.name}", f"+{puppy.duration - plain.duration:.1f} s in all")
    )
    return out


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
    rows = []
    for label, t in zip(labels, tracks):
        if t.spliced is not None:  # a puppy_motion replay: its own checks
            rows.append((name, label, t.duration, [*check_track(t, kin), *replay_checks(tracks[0], t)], t.log))
        elif t.commanded is None:
            rows.append((name, label, t.duration, check_track(t, kin), t.log))
        else:  # an achieved track (dynamics.achieved): the checks hold its command
            cmd = t.commanded
            rows.append((name, t.control, cmd.duration, [*check_track(cmd, kin), achieved_note(t, kin)], cmd.log))
    return rows


def achieved_note(track, kin):
    """An informational Check on an achieved track: how far the arm strays from its command, and
    its least margin (mm) over the command's clearance floors to the user and to the obstacles,
    with the plain task's under the same control for comparison (the checks hold the command; the
    simulated arm has no contacts, so it can dip into the counter)."""
    cmd = track.commanded
    err = 1000.0 * max(
        float(np.linalg.norm(kin.fk_track(track.q, s)[0] - kin.fk_track(cmd.q, s)[0], axis=1).max())
        for s in ("eye", "tool")
    )
    floor = layer.clearance_floor(kin.clearance_track(cmd.func if cmd.func is not None else cmd))
    sides = [track] + ([track.func] if track.func is not None else [])
    clear = [kin.clearance_track(t) for t in sides]
    margin = [1000.0 * (c - floor).min(axis=(0, 2)) for c in clear]
    words = []
    for row, target in ((0, "user"), (1, "obstacles")):
        over = clear[0][:, row] - floor[:, row]
        i, j = np.unravel_index(int(np.argmin(over)), over.shape)
        there = f" ({ARM_PARTS[j]} {1000.0 * clear[0][i, row, j]:.1f} mm clear at {i * DT:.2f} s)"
        plain = f"; the plain task under it: {margin[1][row]:+.1f}" if len(margin) > 1 else ""
        words.append(f"{target} floor margin {margin[0][row]:+.1f} mm{there}{plain}")
    detail = (
        f"under {track.control} the face strays from its command by up to {err:.1f} mm; "
        f"{'; '.join(words)}; the checks above hold the command"
    )
    return Check("achieved", True, f"{err:.1f} mm", detail)
