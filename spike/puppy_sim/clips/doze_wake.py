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
from puppy_motion import behaviors as B

from .. import lib, runs
from ..timeline import DT, Label, concat, constant_track
from . import REST_GRIP, Clip

CALM_S = 0.8  # awake and calm before it gets sleepy
DOZE_S = 5.8  # from CALM_S: drifting off, nodding off and catching itself, curled asleep (library doze_and_wake)
NOD_DIPS = 1
NOD_RAD = np.radians(5.5)
TAIL_S = 0.3
REQUEST_CHIP_S = 0.6  # the 'TASK REQUESTED' chip shows this long, then AWAKE
PROBE_S = 60.0


def parts(ctx):
    """(contributions, when the request arrives, when it is done) on the still track in ctx."""
    sequence = B.doze_and_wake(ctx, CALM_S, DOZE_S, "SLEEPY", "WAKE")
    t_request = CALM_S + DOZE_S
    wake = next(lb for lb in sequence.labels if lb.name == "WAKE")
    turn_s = B.user_turn_s(ctx, wake.t0, max_turn_rad=B.GAZE_MAX_TURN_RAD)  # the look at the user ends the wake
    t_nod = wake.t1 - 1.4 + turn_s / 2.0  # while it holds the look
    nod = B.nod(ctx, t_nod, dips=NOD_DIPS, depth_rad=NOD_RAD)
    t_look = wake.t1 - 1.5 - turn_s / 2.0
    sequence.labels = []
    labels = [
        Label("SLEEPY", CALM_S, t_request),
        Label("TASK REQUESTED", t_request, t_request + REQUEST_CHIP_S),
        Label("AWAKE", t_request + REQUEST_CHIP_S, t_look),
        Label("EAGER (on it)", t_look, (ctx.track.n - 1) * DT + DT),  # ready to go, to the end
    ]
    return [sequence, nod, ctx.empty("labels", labels)], t_request, wake.t1 + turn_s / 2.0


def build():
    e = lib.env()
    home = np.array(runs.HOME_JOINTS)
    probe = lib.context(constant_track(home, REST_GRIP, PROBE_S), e)
    _, t_request, t_done = parts(probe)
    rest = concat(
        [
            constant_track(home, REST_GRIP, t_request, name="rest at home"),
            constant_track(home, REST_GRIP, t_done + TAIL_S - t_request, name="task requested"),
        ],
        name="doze, then a task",
    )
    runs.mark_near_user(rest, e.kin)
    contributions, _, _ = parts(lib.context(rest, e))
    return lib.compose(rest, contributions, e)


CLIP = Clip(
    name="doze_wake",
    title="Dozing off, then waking for a task",
    subtitle="between tasks, nothing to do: it dozes off; a task request wakes it: startle, stretch, a nod to the user",
    camera="close",
    build=build,
    mood="CALM",
)
