"""Proud of itself, both ways, in one pick-and-put-back at the counter (the camera following the
head, the user behind it).

(b) Holding: the real run grab_container_20261029_132745 from its raise to its end. After the raise,
holding the container, the proud is muted: a level look at the user (a pan only, rising as it
turns) and two small bobs; no tilt, no wag, no chomp, so the container never tips.
(a) Empty-handed: the arm then sets the container back down, lets go and backs out (scripted, all
precise), and moves to a ready pose with the face upright. There the proud is free: a bigger look at
the user, rising as it turns, and a happy wag with chomps of the open gripper while it holds the
look. (A head tilt on top would take the clip past the 'demo' budget's 6 s.)

Both play only while the arm is stopped between moves (the boundary tier): a pause (time-warp hold)
inside the stop makes room for each, every move plays exactly as recorded or scripted, and the
pauses stay within the 'demo' budget (check holds the track to it).

The module also holds the staging helpers the other gallery clips share: a beat's length measured
at a stop's posture, the holds that make room for it, and the fence that keeps beats between moves.
"""

from dataclasses import dataclass

import numpy as np

from .. import behaviors as B
from .. import curves, director, layer, runs, scene
from ..kin import UP
from ..timeline import (
    BOUNDARY,
    DT,
    FREE,
    PRECISION,
    Hold,
    Label,
    concat,
    constant_track,
    hold_ramp_s,
    joint_move,
    output_time,
    scripted_track,
    warp,
)
from . import Clip, durations

# ------------------------------------------------------------------------- staging (shared)
MEASURE_PAD_S = 0.5  # a beat is measured on a still track from this time
MEASURE_S = 40.0
FENCE_NOTE = "faded out where the arm moves or before PRECISION"


def beat_length(play, track, t, sc, kin, caps=layer.DEFAULT_CAPS, seed=0):
    """Seconds the beats play(ctx, t0) -> [Contribution] take at the posture at time t of `track`
    (its zone, gripper and held flag), measured on a still copy (as director._measure does)."""
    i = min(int(round(t / DT)), track.n - 1)
    still = constant_track(track.q[i], track.grip[i], MEASURE_S, zone=int(track.zone[i]))
    still.held[:] = track.held[i]
    parts = play(B.Context.of(still, sc, kin, seed, caps), MEASURE_PAD_S)
    ends = [p.span()[1] for p in parts if p.span()] + [b.t1 for p in parts for b in p.beats]
    return max(ends, default=MEASURE_PAD_S) - MEASURE_PAD_S + DT


def room_hold(func, stage, need_s):
    """The Hold (or None) that makes `stage` long enough for need_s of beats between its margins:
    in the middle of the stop, so no motion is retimed, its ramps sized by hold_ramp_s."""
    short = stage.start_margin + need_s + stage.end_margin - (stage.t1 - stage.t0)
    if short <= 0.0:
        return None
    at = DT * round(0.5 * (stage.t0 + stage.t1) / DT)
    return Hold(at, short, hold_ramp_s(func, at, layer.FREE_JOINT_ACCEL, director.HOLD_RAMP_MIN_S))


def between_moves(warped):
    """(N,) the director's boundary fence: 1 inside the stops of the warped track, easing to 0
    before the arm moves and before PRECISION."""
    still = director.stage_mask(warped)
    return np.minimum(B.precision_fence(warped), B.fence(warped, ~still, director.BOUNDARY_FADE_S))


def staged(func, plays, sc, kin, caps=layer.DEFAULT_CAPS, seed=0, finales=None):
    """Puppy Mode on a functional track, between moves only. plays: {stage index (director.stages)
    -> play(ctx, t0) -> [Contribution]}, beats that start at the stop's start margin, measured at
    its posture on a still copy. finales: {stage index -> (play(ctx, t1), length(ctx, t1))}, beats
    that end at the stop's end margin just before the next move (a wind-up), their seconds measured
    by length() on the functional track itself, where that move is known. Each stop gets a pause
    long enough for both; the beats play on the warped clock, fenced to the stops, and are composed
    once. Returns (composed track promising the boundary tier, {stage: (t0, t1)} on its clock)."""
    stops = director.stages(func)
    finales = finales or {}
    func_ctx = B.Context.of(func, sc, kin, seed, caps)
    holds = []
    for k in sorted(set(plays) | set(finales)):
        s = stops[k]
        need = beat_length(plays[k], func, s.t0 + s.start_margin, sc, kin, caps, seed) if k in plays else 0.0
        need += finales[k][1](func_ctx, s.t1 - s.end_margin) if k in finales else 0.0
        holds += [h for h in [room_hold(func, s, need)] if h is not None]
    warped = warp(func, holds)
    ctx = B.Context.of(warped, sc, kin, seed, caps)
    fence = between_moves(warped)
    parts, windows = [], {}
    for k in sorted(set(plays) | set(finales)):
        s = stops[k]
        windows[k] = (output_time(warped, s.t0) + s.start_margin, output_time(warped, s.t1) - s.end_margin)
        played = (plays[k](ctx, windows[k][0]) if k in plays else []) + (
            finales[k][0](ctx, windows[k][1]) if k in finales else []
        )
        for c in played:
            director.fenced(c, fence, FENCE_NOTE)
            parts.append(c)
    puppy = layer.compose(warped, parts, kin, caps)
    puppy.tier = BOUNDARY
    return puppy, windows


def stop_at(func, phase):
    """Index (director.stages) of the stop the arm is in when the phase named `phase` starts."""
    t = next(p.t0 for p in func.phases if p.name == phase)
    return next(k for k, s in enumerate(director.stages(func)) if s.t0 - 2 * DT <= t <= s.t1)


def chip(ctx, name, t0, parts):
    """A contribution carrying only the chip `name` from t0 until `parts` stop moving."""
    return ctx.empty(name, [Label(name, t0, max(p.span()[1] for p in parts if p.span()))])


def min_jerk_grip(g0, g1, seconds):
    """(N,) a gripper command easing from g0 to g1 over `seconds`."""
    return g0 + (g1 - g0) * curves.min_jerk(np.linspace(0.0, 1.0, int(round(seconds / DT)) + 1))


# ------------------------------------------------------------------------------- the clip
RAISE_FROM_S = 12.6  # the reference run's own clock: the clip starts late in its raise
LOWER_S = 0.9  # set it back down where it was picked up (the run's closing pose)
LET_GO_S = 0.45
OPEN_GRIP = 0.3  # the gripper opens this far to let go
BACK_OUT_M = 0.06  # straight out along the approach, clear of the brim
BACK_OUT_S = 0.6
READY_BACK_M = 0.13  # the ready pose: this far back from the grasp along the approach,
READY_UP_M = 0.04  # this much higher,
READY_DOWN_RAD = 0.0  # the face upright, looking along the approach's heading and this much down
READY_S = 1.7  # the face rolls upright on the way (joint 7 turns about 1.5 rad)
READY_REST_S = 0.4
HOLDING_LOOK_RAD = np.radians(12.0)  # the muted look at the user: a pan, the container level
EMPTY_LOOK_RAD = np.radians(18.0)  # empty-handed: from the ready pose the user is 167 deg behind
EMPTY_CYCLES, EMPTY_CHOMPS = 2.0, 2  # the wag and the chomps while it looks (the tilt does not fit the budget)
HOLDING_HOLD_S = 0.1  # the look holds this long after the bobs
FLOURISH_GAP_S = 0.1  # the tilt goes back this long after the wag and the chomps
TILT_OVERLAP = 0.2  # share of the tilt that overlaps the end of the look (and its return the look back)
HOLDING_CHIP, EMPTY_CHIP = "PROUD (holding)", "PROUD"


def proud_level(look_rad, name=HOLDING_CHIP):
    """play(ctx, t0) for (b), the muted proud while holding: a level look at the user (a pan, at
    most look_rad) rising as it turns, two bobs, and back. The chip `name` spans it."""

    def play(ctx, t0):
        t_look = t0 + B.PROUD_LOOK_AT_S
        t_there = t_look + B.user_turn_s(ctx, t_look, True, look_rad, B.PROUD_LIFT_M)
        hop_s = B.paced_bump(ctx, t_there, [0.0, 0.0, B.PROUD_BOB_M, 0.0, 0.0, 0.0], B.BOUNCE_HOP_S)
        proud = B.proud(ctx, t0, t_there + B.PROUD_BOBS * hop_s + HOLDING_HOLD_S, max_turn_rad=look_rad)
        return [proud, chip(ctx, name, t0, [proud])]

    return play


def proud_flourish(look_rad, name=EMPTY_CHIP, tilt=True, cycles=B.WIGGLE_CYCLES, chomps=B.CLACK_CHOMPS):
    """play(ctx, t0) for (a), the empty-handed proud with every flourish: the proud look at the user
    (at most look_rad) rising as it turns (proud without its own flourish), the head tilting as the
    look settles (unless tilt is False), then while both hold a happy wag with chomps of the
    gripper (`cycles` swings, `chomps` chomps); the tilt comes back as the look turns back. The
    tilt overlaps the ends of the look a little (it rolls the wrist's last joint; the look pans
    mostly with joint 5). The chip `name` spans it all."""

    def play(ctx, t0):
        t_look = t0 + B.PROUD_LOOK_AT_S
        turn_s = B.user_turn_s(ctx, t_look, False, look_rad, B.PROUD_LIFT_M)
        roll = [0.0, 0.0, 0.0, B.PROUD_TILT_RAD, 0.0, 0.0]
        tilt_s = B.paced_move(ctx, t_look, roll, B.TILT_TURN_S, B.TILT_OVERSHOOT) if tilt else 0.0
        t_tilt = t_look + turn_s - TILT_OVERLAP * tilt_s
        t_play = t_tilt + tilt_s
        mouth = B.clack(ctx, t_play, chomps=chomps)
        t_untilt = t_play + max(cycles / B.WIGGLE_HZ, mouth.span()[1] - t_play) + FLOURISH_GAP_S
        t_back = t_untilt + (1.0 - TILT_OVERLAP) * tilt_s
        parts = [B.proud(ctx, t0, t_back, flourish=None, max_turn_rad=look_rad), B.wiggle(ctx, t_play, cycles), mouth]
        if tilt:
            parts.append(B.head_tilt(ctx, t_tilt, t_untilt, angle_rad=B.PROUD_TILT_RAD))
        return parts + [chip(ctx, name, t0, parts)]

    return play


def put_back(kin, sc, q_from, grip):
    """Scripted, after the run: set the container down at the run's closing pose (held), open the
    gripper (let go), back straight out along the approach (all PRECISION), then move to the ready
    pose with the face upright (FREE once 0.11 m from the grasp pose) and stop there."""
    run = runs.load_run(scene.REFERENCE_RUN)
    q_close = run.q[runs.closing_index(run)]  # the container's pose in the gripper is defined here
    lower = joint_move(q_from, q_close, LOWER_S, grip, PRECISION, "set down")
    lower.held[:] = True
    n = int(round(LET_GO_S / DT)) + 1
    let_go = scripted_track(np.tile(q_close, (n, 1)), min_jerk_grip(grip, OPEN_GRIP, LET_GO_S), PRECISION, "let go")
    pl = sc.placement
    p, R = kin.fk(q_close, "tool")
    q_out, _ = kin.cartesian_path(q_close, p - BACK_OUT_M * pl.approach, R, BACK_OUT_S)
    out = scripted_track(q_out, OPEN_GRIP, PRECISION, "back out")
    q_ready, _, _ = kin.ik(q_out[-1], *ready_pose(p, pl.approach))
    ready = joint_move(q_out[-1], q_ready, READY_S, OPEN_GRIP, FREE, "to ready")
    tool = kin.fk_track(ready.q, "tool")[0]
    ready.zone[np.linalg.norm(tool - p, axis=1) < runs.FINAL_APPROACH_M] = PRECISION
    rest = constant_track(q_ready, OPEN_GRIP, READY_REST_S, name="ready")
    return [lower, let_go, out, ready, rest]


def ready_pose(p_grasp, approach):
    """(p, R) of the tool at the ready pose: back and up from the grasp, the face upright (ee y up),
    looking along the approach's heading and READY_DOWN_RAD down."""
    heading = np.array([approach[0], approach[1], 0.0]) / np.linalg.norm(approach[:2])
    gaze = np.cos(READY_DOWN_RAD) * heading - np.sin(READY_DOWN_RAD) * UP
    up = UP - (UP @ gaze) * gaze
    up = up / np.linalg.norm(up)
    p = p_grasp - READY_BACK_M * approach + READY_UP_M * UP
    return p, np.column_stack([np.cross(up, gaze), up, gaze])


def functional(sc, kin):
    """The plain task: the run from late in its raise to its end, then the scripted put-back."""
    run = runs.load_run(scene.REFERENCE_RUN)
    start, end = runs.replay_window(run)
    tail = runs.functional_track(run, kin, window=(start + RAISE_FROM_S, end))
    func = concat([tail, *put_back(kin, sc, tail.q[-1], float(tail.grip[-1]))], name="pick and put back")
    runs.mark_near_user(func, kin)
    return func


def build():
    sc = scene.build()
    kin = sc.kin()
    func = functional(sc, kin)
    stops = director.stages(func)
    holding = next(k for k, s in enumerate(stops) if func.held[s.i0])
    empty = len(stops) - 1  # the ready pose, the last stop
    empty_handed = proud_flourish(EMPTY_LOOK_RAD, tilt=False, cycles=EMPTY_CYCLES, chomps=EMPTY_CHOMPS)
    puppy, _ = staged(func, {holding: proud_level(HOLDING_LOOK_RAD), empty: empty_handed}, sc, kin)
    puppy.budget = director.BUDGETS[director.DEFAULT_BUDGET]  # check holds the added time to it
    return puppy


@dataclass(frozen=True)
class TimedClip(Clip):
    """A single Puppy Mode clip whose subtitle adds both durations and the budget (as a pair's does)."""

    def subtitle_for(self, tracks):
        (puppy,) = tracks
        return [self.subtitle, durations(puppy.func.source, puppy)]


CLIP = TimedClip(
    name="success_proud",
    title="Proud of itself",
    subtitle="after a success, while the arm waits between moves: holding the container it only looks at you "
    "(level) and bobs; empty-handed it also wags and chomps",
    camera="close_follow",
    build=build,
)
