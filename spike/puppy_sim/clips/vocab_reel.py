"""The vocabulary reel: every Puppy Mode primitive once, at home in the close camera, its name in
the beat chip. Each plays for as long as it takes under RAMMP's caps (about 2 s for the quick
ones, longer for big turns), with a short stillness between them.

The motion-craft primitives ride on short scripted moves: a turn to the side with a wind-up, a
lead gaze and a follow-through, then the same move back plain for comparison; and a short
approach straight ahead whose last few centimetres are PRECISION, with the lock-on stare at the
boundary and the slow-in through it.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .. import behaviors as B
from .. import layer, runs, scene
from ..kin import UP, across_gaze
from ..layer import EXPR_JOINT_ACCEL
from ..timeline import (
    DT,
    PRECISION,
    Hold,
    Label,
    Slow,
    concat,
    constant_track,
    hold_ramp_s,
    joint_move,
    output_time,
    scripted_track,
    still_span,
    warp,
)
from . import Clip

REST_GRIP = 0.79  # the reference run's gripper while it waited at home
GAP_S = 0.6  # stillness between primitives
MEASURE_S = 200.0  # long enough for the whole rest part
BREATHE_S = 4.6
MICRO_S = 3.2
GAZE_HOLD_S = 0.4
TILT_HOLD_S = 0.6
DROOP_EYES_AFTER_S = 1.8
DROOP_EYES_HOLD_S = 0.8
PROUD_S = 6.0
DOZE_S = 4.5
DOUBLE_TAKE_SPOT = (-0.02, 0.17, 0.0)  # glance-away spot on the counter, from the container (m)

SIDE_SHIFT_M = (0.0, 0.10, -0.03)  # the turn to the side: the tool moves left and a little down
SIDE_TURN_RAD = (np.radians(25.0), np.radians(10.0))  # (pan left, tilt down)
MOVE_S = 2.2
SIDE_REST_S = 2.2  # at the side: the follow-through settles
WIND_UP_PAUSE_S = 0.9
APPROACH_M = 0.12  # forward, along the gaze
APPROACH_S = 2.4
APPROACH_PRECISION_M = 0.04  # the last part of the approach is PRECISION
LOCK_PAUSE_S = 1.6
LOCK_SLOW = 0.75
LOCK_MARGIN_S = 0.06
LOCK_TARGET_AHEAD_M = 0.3  # the stare is at a point straight ahead
PRECISE_REST_S = 1.0
RETREAT_S = 2.2
END_REST_S = 1.2


def _rest_slots():
    """(chip, play(ctx, t0) -> [Contribution]) for the primitives at rest, in reel order."""

    def droop_and_eyes(ctx, t):
        eyes = B.puppy_eyes(ctx, t + DROOP_EYES_AFTER_S, t + DROOP_EYES_AFTER_S + DROOP_EYES_HOLD_S, beat="PUPPY EYES")
        sag = B.droop(ctx, t, eyes.span()[1])
        sag.beats = [Label("DROOP", t, t + DROOP_EYES_AFTER_S)]
        eyes.beats = [Label("PUPPY EYES", t + DROOP_EYES_AFTER_S, sag.span()[1])]
        return [sag, eyes]

    def double_take(ctx, t):
        return [B.double_take(ctx, ctx.container, ctx.container + np.array(DOUBLE_TAKE_SPOT), t, beat="DOUBLE TAKE")]

    return (
        ("BREATHE", lambda ctx, t: [B.breathe(ctx, t, t + BREATHE_S)]),
        ("MICRO LIFE", lambda ctx, t: [B.micro_life(ctx, t, t + MICRO_S)]),
        ("GAZE", lambda ctx, t: [B.gaze_to(ctx, ctx.container, t, t + GAZE_HOLD_S)]),
        ("HEAD TILT", lambda ctx, t: [B.head_tilt(ctx, t, t + TILT_HOLD_S)]),
        ("PERK", lambda ctx, t: [B.perk(ctx, t)]),
        ("NOD", lambda ctx, t: [B.nod(ctx, t)]),
        ("BOUNCE", lambda ctx, t: [B.bounce(ctx, t)]),
        ("WIGGLE", lambda ctx, t: [B.wiggle(ctx, t)]),
        ("CLACK", lambda ctx, t: [B.clack(ctx, t)]),
        ("SNIFF", lambda ctx, t: [B.sniff(ctx, ctx.container, t)]),
        ("DOUBLE TAKE", double_take),
        ("SEARCH", lambda ctx, t: [B.search(ctx, t)]),
        ("PROUD", lambda ctx, t: [B.proud(ctx, t, t + PROUD_S)]),
        ("STARTLE", lambda ctx, t: [B.startle(ctx, t)]),
        ("SHAKE OFF", lambda ctx, t: [B.shake_off(ctx, t)]),
        (None, droop_and_eyes),  # DROOP, then PUPPY EYES while drooped
        (None, lambda ctx, t: [B.doze_and_wake(ctx, t, DOZE_S, "DOZE", "WAKE")]),
    )


def _labelled(chip, parts, t0):
    """Parts with the chip over [t0, their end] (chip None: they carry their own beats)."""
    if chip is None:
        return parts
    end = max(p.span()[1] for p in parts if p.span())
    for p in parts:
        p.beats = []
    parts[0].beats = [Label(chip, t0, end)]
    return parts


def _rest_plan(sc, kin, home):
    """[(chip, play, t0)] for the rest part and its length, measured on a long still track."""
    ctx = B.Context.of(constant_track(home, REST_GRIP, MEASURE_S), sc, kin)
    plan, t = [], GAP_S
    for chip, play in _rest_slots():
        plan.append((chip, play, t))
        ends = [p.span()[1] for p in play(ctx, t) if p.span()]
        t = max(ends) + GAP_S
    return plan, t


def _side_pose(kin, home):
    """Joints with the tool shifted SIDE_SHIFT_M and the gaze panned and tilted SIDE_TURN_RAD
    (a gravity turn: tilt down about the axis across the gaze, then pan about world z)."""
    p, R = kin.fk(home, "tool")
    pan, tilt = SIDE_TURN_RAD
    turn = Rotation.from_rotvec(pan * UP) * Rotation.from_rotvec(tilt * across_gaze(R))
    q, _, _ = kin.ik(home, p + np.array(SIDE_SHIFT_M), turn.as_matrix() @ R)
    return q


def _approach(kin, home):
    """A straight tool move forward along the gaze (straight toward the container would drive
    joint 4 into RAMMP's 144 deg guard), its last APPROACH_PRECISION_M marked PRECISION, a
    precise rest at its end, and the retreat home."""
    p, R = kin.fk(home, "tool")
    end = p + APPROACH_M * R[:, 2]
    q, _ = kin.cartesian_path(home, end, R, APPROACH_S)
    move = scripted_track(q, REST_GRIP, name="approach")
    tool = kin.fk_track(move.q, "tool")[0]
    move.zone[np.linalg.norm(tool - end, axis=1) < APPROACH_PRECISION_M] = PRECISION
    rest = constant_track(q[-1], REST_GRIP, PRECISE_REST_S, zone=PRECISION, name="at it")
    back = joint_move(q[-1], home, RETREAT_S, REST_GRIP, name="back")
    return move, rest, back


def _functional(kin, home, rest_s):
    """The reel's functional track: rest, the side turn there and back, the approach and back."""
    side = _side_pose(kin, home)
    move, precise, back = _approach(kin, home)
    parts = [
        constant_track(home, REST_GRIP, rest_s, name="rest at home"),
        joint_move(home, side, MOVE_S, REST_GRIP, name="turn"),
        constant_track(side, REST_GRIP, SIDE_REST_S, name="at the side"),
        joint_move(side, home, MOVE_S, REST_GRIP, name="back (plain)"),
        constant_track(home, REST_GRIP, GAP_S, name="rest"),
        move,
        precise,
        back,
        constant_track(home, REST_GRIP, END_REST_S, name="rest"),
    ]
    func = concat(parts, name="vocabulary")
    runs.mark_near_user(func, kin)
    return func, [p.duration for p in parts]


def build():
    sc = scene.build()
    kin = sc.kin()
    home = np.array(runs.HOME_JOINTS)
    plan, rest_s = _rest_plan(sc, kin, home)
    func, lengths = _functional(kin, home, rest_s)
    t_turn = lengths[0]
    t_turned = t_turn + lengths[1]
    t_approach = sum(lengths[:5])
    boundary = int(np.argmax((func.zone == PRECISION) & (func.t >= t_approach))) * DT - LOCK_MARGIN_S
    t_at = sum(lengths[:6])
    holds = [Hold(t_turn, WIND_UP_PAUSE_S, hold_ramp_s(func, t_turn, EXPR_JOINT_ACCEL, 0.3))]
    ramp = hold_ramp_s(func, boundary, EXPR_JOINT_ACCEL, 0.3)
    slows = [Slow(boundary, t_at, LOCK_SLOW, ramp, hold_s=LOCK_PAUSE_S)]
    w = warp(func, holds, slows)
    ctx = B.Context.of(w, sc, kin)
    parts = []
    for chip, play, t0 in plan:  # the warp only pauses after the rest part: same times there
        parts += _labelled(chip, play(ctx, t0), t0)
    parts += _motion_craft(ctx, w, t_turn, t_turned, sum(lengths[:3]), sum(lengths[:4]), boundary)
    return layer.compose(w, parts, kin)


def _motion_craft(ctx, w, t_turn, t_turned, t_back, t_home, boundary):
    """Wind-up, lead gaze and follow-through on the turn out; PLAIN MOVE on the way back; LOCK ON
    and FOCUS on the approach (times are functional; placed on the warped clock)."""
    go = output_time(w, t_turn) + WIND_UP_PAUSE_S
    arrive = output_time(w, t_turned)
    stare = still_span(w, boundary)
    precise = int(np.argmax((w.zone == PRECISION) & (w.t >= stare[1]))) * DT
    lead = B.lead_gaze(ctx, go - B.LEAD_S - B.LEAD_FADE_S, arrive)
    wind = B.wind_up(ctx, go - WIND_UP_PAUSE_S, go)
    follow = B.follow_through(ctx, arrive)
    eye, R = ctx.kin.fk(ctx.q(stare[0]), "eye")
    lock = B.lock_on(ctx, eye + LOCK_TARGET_AHEAD_M * R[:, 2], stare[0], stare[1])
    focus_end = (precise_run_end(w, precise) + 1) * DT
    labels = ctx.empty(
        "labels",
        [
            Label("WIND UP", go - WIND_UP_PAUSE_S, go),
            Label("LEAD GAZE", go, arrive),
            Label("FOLLOW THROUGH", arrive, arrive + B.FOLLOW_SETTLE_S),
            Label("PLAIN MOVE", output_time(w, t_back), output_time(w, t_home)),
            Label("LOCK ON", stare[0], stare[1]),
            Label("FOCUS", precise, focus_end),
        ],
    )
    return [wind, lead, follow, lock, labels]


def precise_run_end(track, t):
    """Index of the last sample of the PRECISION run that starts at time t."""
    i = int(round(t / DT))
    after = np.nonzero(track.zone[i:] != PRECISION)[0]
    return i + (int(after[0]) - 1 if after.size else track.n - 1 - i)


CLIP = Clip(
    name="vocab_reel",
    title="Puppy Mode: the vocabulary",
    subtitle="each primitive once, at RAMMP's speed caps; the chip names it",
    camera="close",
    build=build,
)
