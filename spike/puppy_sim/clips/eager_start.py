"""Eager to start, plain vs Puppy Mode: the first seconds of the real run
grab_container_20261029_132745, from just before the task is requested until the arm has arrived
at the view pose. Plain: the recorded motion. Puppy Mode: the same moves, each exactly as
recorded, with beats only where the arm is stopped between them (the boundary tier):

- asked to go, it perks up as it turns to the user, nods, gives a small bounce and turns back
  (EAGER), in the stop where the request arrives;
- in the stop before the move to the view pose it winds up, back at rest when the move starts;
- once it has arrived, a follow-through settles, then it holds still.

Each pause is a time-warp hold inside a stop (director.stages), so `check` can hold the track to
the boundary tier. The eager beat's parts play one after another: each is paced to the arm's own
envelope on its own, so overlapping a nod (the wrist) with a hop or a turn would exceed it. The
wind-up and the follow-through are bigger than the vocabulary's defaults, so they show.
"""

import numpy as np

from .. import behaviors as B
from .. import director, layer, runs, scene
from ..timeline import BOUNDARY, DT, Hold, Label, constant_track, hold_ramp_s, output_time, warp
from . import PairClip

# the window on the replay's clock: the request comes at 1.0, the arm is at the view pose at 4.74.
# Its 50 Hz grid skips the run's 10 ms 'locate' phase (PRECISION), which would split the stop.
WINDOW_S = (0.7, 5.1)
SEED = 0
EAGER_TURN_RAD = np.radians(11.0)  # the turn to the user
EAGER_LIFT_M = B.PERK_LIFT_M  # perked up: the head rises this much as it turns
SETTLE_S = 0.05  # each part of the eager beat starts this long after the one before has arrived
NOD_RAD = np.radians(5.5)
BOUNCE_HOPS = 2
BOUNCE_M = 0.014
REACT_S = 0.05  # the eager beat starts this long after the request
REQUEST_CHIP_S = 0.45  # the 'TASK REQUESTED' chip shows this long, then EAGER
WIND_SHIFT_M = 1.7 * B.WIND_UP_SHIFT_M  # the wind-up draws back further than the default, to show
WIND_TURN_RAD = 1.7 * B.WIND_UP_TURN_RAD
FOLLOW_TURN_RAD = 2.0 * B.FOLLOW_TURN_RAD  # the follow-through too (it fits itself to the caps: fit_scale)
FOLLOW_SHIFT_M = 2.0 * B.FOLLOW_SHIFT_M
FOLLOW_S = 1.0
STILL_END_S = 0.2  # still at the view pose once the follow-through has settled
MEASURE_S = 30.0
MOVE_CHIP = "MOVE (as recorded)"  # over the move to the view pose: Puppy Mode leaves it untouched


def eager(ctx, t_on):
    """EAGER from t_on: perks up as it turns to the user (one move), nods, gives a small bounce and
    turns back. Returns ([contributions], when it is back)."""
    turn_s = B.user_turn_s(ctx, t_on, max_turn_rad=EAGER_TURN_RAD, lift_m=EAGER_LIFT_M)
    nod = B.nod(ctx, t_on + turn_s + SETTLE_S, dips=1, depth_rad=NOD_RAD)
    hops = B.bounce(ctx, nod.span()[1] + SETTLE_S, hops=BOUNCE_HOPS, height_m=BOUNCE_M)
    t_back = hops.span()[1]
    look = B.look_at_user(ctx, t_on, t_back, max_turn_rad=EAGER_TURN_RAD, lift_m=EAGER_LIFT_M)
    return [look, nod, hops], t_back + turn_s


def eager_s(func, t, sc, kin):
    """How long the eager beat takes at the functional posture at time t (on a still copy)."""
    i = int(round(t / DT))
    still = constant_track(func.q[i], func.grip[i], MEASURE_S)
    _, t_end = eager(B.Context.of(still, sc, kin, SEED), 1.0)
    return t_end - 1.0


def _stage_at(stages, t):
    return next(s for s in stages if s.t0 <= t <= s.t1)


def _hold(func, at, hold_s):
    at = round(at / DT) * DT
    return Hold(at, max(hold_s, 0.0), hold_ramp_s(func, at, layer.FREE_JOINT_ACCEL, director.HOLD_RAMP_MIN_S))


def parts(func, sc, kin):
    """(warped, contributions): the plain track paused inside its stops, and the beats in them,
    fenced to the stops, with breathing and micro-life wherever the arm rests."""
    margin = director.STILL_MARGIN_S
    stages, events = director.find_events(func, kin)
    t_req = next(e.t for e in events if e.name == "requested")
    t_go = next(e.t for e in events if e.name == "transit_start")
    t_arrive = next(e.t for e in events if e.name == "arrived")
    ask, before, there = _stage_at(stages, t_req), _stage_at(stages, t_go - DT), _stage_at(stages, t_arrive)
    edge = director.HOLD_EDGE * DT
    eager_need = REACT_S + eager_s(func, t_req, sc, kin)
    wind = dict(shift_m=WIND_SHIFT_M, turn_rad=WIND_TURN_RAD)
    wind_need = B.wind_up_s(B.Context.of(func, sc, kin, SEED), t_go, t_go, **wind)
    holds = [
        _hold(func, t_req, eager_need - (ask.t1 - margin - t_req)),
        _hold(func, before.t0 + edge, wind_need + 2 * margin - (before.t1 - before.t0)),
        _hold(func, there.t0 + edge, margin + FOLLOW_S + STILL_END_S - (there.t1 - there.t0)),
    ]
    warped = warp(func, holds)
    ctx = B.Context.of(warped, sc, kin, SEED)

    t_asked = output_time(warped, t_req)
    eager_parts, t_eager_end = eager(ctx, t_asked + REACT_S)
    go = output_time(warped, before.t1) - margin  # the wind-up is back at rest here
    wind_s = B.wind_up_s(ctx, go, go, **wind)
    t_still = output_time(warped, there.t0) + margin
    beats = [
        *eager_parts,
        B.wind_up(ctx, go - wind_s, go, **wind),
        B.follow_through(ctx, t_still, FOLLOW_TURN_RAD, FOLLOW_SHIFT_M, FOLLOW_S),
    ]
    still = director.stage_mask(warped)
    between = np.minimum(B.precision_fence(warped), B.fence(warped, ~still, director.BOUNDARY_FADE_S))
    for c in beats:
        director.fenced(c, between, "faded out where the arm moves or before PRECISION")
    quiet = (warped.t >= go - wind_s) & (warped.t < go)
    background = director.background(B.Context.of(warped, sc, kin, SEED + director.BACKGROUND_SEED), quiet | ~still)
    labels = [
        Label("TASK REQUESTED", t_asked, t_asked + REQUEST_CHIP_S),
        Label("EAGER", t_asked + REQUEST_CHIP_S, t_eager_end),
        Label("WIND UP", go - wind_s, go),
        Label(MOVE_CHIP, output_time(warped, t_go), output_time(warped, t_arrive)),
        Label("FOLLOW THROUGH", t_still, t_still + FOLLOW_S),
    ]
    return warped, [*beats, *background, ctx.empty("labels", labels)]


def puppy_track(func, sc, kin):
    """The Puppy Mode track over the plain one: it promises the boundary tier (`check` holds it)."""
    warped, contributions = parts(func, sc, kin)
    puppy = layer.compose(warped, contributions, kin)
    puppy.tier = BOUNDARY
    return puppy


def build():
    sc = scene.build()
    kin = sc.kin()
    run = runs.load_run(scene.REFERENCE_RUN)
    start = runs.replay_window(run)[0]
    plain = runs.functional_track(run, kin, (start + WINDOW_S[0], start + WINDOW_S[1]))
    return plain, puppy_track(plain, sc, kin)


CLIP = PairClip(
    name="eager_start",
    title="Eager to start: plain vs Puppy Mode",
    subtitle=(
        "the start of a real run (grab container): asked to go, it perks up, nods and bounces, "
        "winds up, then moves as recorded"
    ),
    camera="close_follow",
    build=build,
    mood="CALM",
)
