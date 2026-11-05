"""Dozing off, then waking for a task: resting at home with nothing to do, the arm's breathing
slows and deepens and its head grows heavy. It nods off and catches itself, then curls into a low
rest and sleeps. A task request arrives: it startles awake, stretches, turns to the user and nods
('on it'), ready to go.

All at home, in the close camera. The doze and the wake are one sequence of poses, each move paced
to the caps of the arm's own envelope; the sag drifts away from the user as it goes down
(behaviors.sag_shift), so the folded elbow never closes in on them. The breathing is its own layer,
its period and depth growing as it falls asleep; it stops with the startle.
"""

import numpy as np

from .. import behaviors as B
from .. import curves, layer, runs, scene
from ..kin import UP
from ..timeline import DT, Label, concat, constant_track
from . import REST_GRIP, Clip

# the breathing: awake, then slower and deeper as it falls asleep
BREATH_PERIOD_S = (3.6, 6.0)
BREATH_RISE_M = (0.0035, 0.006)
BREATH_CHIN_RAD = np.radians(1.5)  # the chin lifts this much on the in-breath
BREATH_STOP_S = 0.5  # the breathing has stopped by the time the request arrives (still, then the startle)
CALM_S = 0.8  # awake and calm before it gets sleepy

# the poses: (sag m, look down rad, roll rad, lift m) from resting awake
DROWSY = (0.012, np.radians(7.0), 0.0, 0.0)  # eyelids heavy
NODDING = (0.025, np.radians(17.0), np.radians(3.0), 0.0)  # nodding off...
CAUGHT = (0.006, np.radians(3.0), 0.0, 0.0)  # ...and catching itself
CURLED = (B.DOZE_SAG_M, B.DOZE_LOOK_DOWN_RAD, B.DOZE_ROLL_RAD, 0.0)  # asleep, curled low
DEEPER = (B.DOZE_SAG_M + 0.006, B.DOZE_LOOK_DOWN_RAD + B.DOZE_DIP_RAD, B.DOZE_ROLL_RAD, 0.0)  # a sleepy dip
# startled: the head pops up a little, small enough to be quick (the stretch unfolds the rest)
JOLT = (DEEPER[0] - 0.01, DEEPER[1] - np.radians(6.0), DEEPER[2] - np.radians(2.0), B.WAKE_JOLT_M)
STRETCH = (0.0, -B.WAKE_STRETCH_LOOK_UP_RAD, 0.0, B.WAKE_STRETCH_M)  # up and chin up
AWAKE = (0.0, 0.0, 0.0, 0.0)

# (pose, design seconds to reach it (longer where the caps need it), seconds held there)
DOZE = (
    (DROWSY, 1.4, 0.0),
    (NODDING, 1.0, 0.0),
    (CAUGHT, 0.45, 0.25),
    (CURLED, 1.5, 0.0),
    (DEEPER, 1.0, 0.7),  # sinks a little deeper: asleep
)
REACT_S = 0.15  # the startle follows the request this quickly
WAKE = (
    (JOLT, 0.3, 0.1),
    (STRETCH, 0.9, 0.25),
    (AWAKE, 0.9, 0.0),
)
LOOK_AT = 0.5  # the turn to the user starts this share of the way through the stretch's relax
LOOK_MAX_RAD = np.radians(15.0)  # the turn to the user
NOD_DIPS = 2
NOD_RAD = np.radians(5.5)
NOD_SETTLE_S = 0.1  # the nod starts this long after the turn has arrived
NOD_HOLD_S = 0.15  # still looking at the user this long after the nod
TAIL_S = 0.3
REQUEST_CHIP_S = 0.6  # the 'TASK REQUESTED' chip shows this long, then AWAKE
PROBE_S = 60.0


def pose(ctx, t, sag_m=0.0, down_rad=0.0, roll_rad=0.0, lift_m=0.0):
    """(6,) head offset at the posture at t: a sag (down, drifting away from the user), a lift, a
    look down (a gravity tilt; negative: up) and a roll about the gaze."""
    shift = B.sag_shift(ctx, t, sag_m) + UP * lift_m
    return np.concatenate([shift, ctx.kin.gravity_turn(ctx.q(t), 0.0, down_rad, roll_rad)])


def keyframes(ctx, t_on, start, moves):
    """([(time, offset)], [arrival time per move]): easing from offset `start` at t_on through the
    poses of `moves` ((pose, design s, hold s) each), every move paced to the caps at the posture
    at t_on. The last key is when it ends."""
    keys, arrivals, t_k, here = [(t_on, start)], [], t_on, start
    for spec, design_s, hold_s in moves:
        there = pose(ctx, t_on, *spec)
        t_k += B.paced_move(ctx, t_on, there - here, design_s)
        arrivals.append(t_k)
        keys += [(t_k, there), (t_k + hold_s, there)]
        t_k, here = t_k + hold_s, there
    return keys, arrivals


def eased(t, keys):
    """(N, 6) offsets eased (min-jerk) from key to key; every key is a stop, so any sequence is
    smooth. Held at the first key before it and at the last after it."""
    times = np.array([k for k, _ in keys])
    offsets = np.array([o for _, o in keys])
    seg = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(keys) - 2)
    a, b = times[seg], times[seg + 1]
    u = np.divide(t - a, b - a, out=np.ones_like(t), where=b > a)
    out = offsets[seg] + curves.min_jerk(u)[:, None] * (offsets[seg + 1] - offsets[seg])
    out[t < times[0]] = offsets[0]
    out[t >= times[-1]] = offsets[-1]
    return out


def sleepy_breath(ctx, t_slow, t_asleep, t_stop):
    """Breathing from before the clip until t_stop, slowing and deepening from t_slow until
    t_asleep: the head rises and the chin lifts on each in-breath. It fades out over the
    BREATH_STOP_S before t_stop."""
    t = ctx.track.t
    sleepy = curves.ramp(t, t_slow, t_asleep)
    period = BREATH_PERIOD_S[0] + (BREATH_PERIOD_S[1] - BREATH_PERIOD_S[0]) * sleepy
    rise = BREATH_RISE_M[0] + (BREATH_RISE_M[1] - BREATH_RISE_M[0]) * sleepy
    wave = np.sin(2.0 * np.pi * np.cumsum(DT / period)) * (1.0 - curves.ramp(t, t_stop - BREATH_STOP_S, t_stop))
    c = ctx.empty("breathing")
    for i in np.nonzero(wave)[0]:
        c.head[i, :3] = UP * rise[i] * wave[i]
        c.head[i, 3:] = ctx.kin.gravity_turn(ctx.track.q[i], 0.0, -BREATH_CHIN_RAD * wave[i])
    return c


def parts(ctx):
    """(contributions, when the request arrives, when it is done) on the still track in ctx."""
    doze, dozed = keyframes(ctx, CALM_S, np.zeros(6), DOZE)
    t_request = doze[-1][0]
    wake, woke = keyframes(ctx, t_request + REACT_S, doze[-1][1], WAKE)
    t = ctx.track.t
    sequence = ctx.empty("doze and wake")
    sequence.head[:] = eased(t, doze + wake)
    t_relax = woke[-2] + WAKE[-2][2]  # the stretch starts to relax
    t_look = t_relax + LOOK_AT * (woke[-1] - t_relax)
    turn_s = B.user_turn_s(ctx, t_look, max_turn_rad=LOOK_MAX_RAD)
    t_nod = t_look + turn_s + NOD_SETTLE_S
    nod = B.nod(ctx, t_nod, dips=NOD_DIPS, depth_rad=NOD_RAD)
    t_back = nod.span()[1] + NOD_HOLD_S
    look = B.look_at_user(ctx, t_look, t_back, max_turn_rad=LOOK_MAX_RAD)
    t_done = max(t_back + turn_s, woke[-1])
    t_asleep = dozed[[spec for spec, _, _ in DOZE].index(CURLED)]  # curled up: fully asleep
    labels = [
        Label("SLEEPY", CALM_S, t_request),
        Label("TASK REQUESTED", t_request, t_request + REQUEST_CHIP_S),
        Label("AWAKE", t_request + REQUEST_CHIP_S, t_look),
        Label("EAGER (on it)", t_look, ctx.track.duration + DT),  # ready to go, to the end
    ]
    breathing = sleepy_breath(ctx, CALM_S, t_asleep, t_request)
    return [breathing, sequence, look, nod, ctx.empty("labels", labels)], t_request, t_done


def build():
    sc = scene.build()
    kin = sc.kin()
    home = np.array(runs.HOME_JOINTS)
    probe = B.Context.of(constant_track(home, REST_GRIP, PROBE_S), sc, kin)
    _, t_request, t_done = parts(probe)
    rest = concat(
        [
            constant_track(home, REST_GRIP, t_request, name="rest at home"),
            constant_track(home, REST_GRIP, t_done + TAIL_S - t_request, name="task requested"),
        ],
        name="doze, then a task",
    )
    runs.mark_near_user(rest, kin)
    contributions, _, _ = parts(B.Context.of(rest, sc, kin))
    return layer.compose(rest, contributions, kin)


CLIP = Clip(
    name="doze_wake",
    title="Dozing off, then waking for a task",
    subtitle="between tasks, nothing to do: it dozes off; a task request wakes it: startle, stretch, a nod to the user",
    camera="close",
    build=build,
    mood="CALM",
)
