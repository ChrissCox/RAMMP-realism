"""Alive at rest: the arm waits at home and breathes, glances at its user (perking up a little as
it turns: hello), tilts its head while looking, turns to look at the container, and settles back
to looking forward.

At home the head sits 0.38 m from the user and may come no closer than 0.345 m (min of its
functional clearance and 0.35 m, less 5 mm), so the glance at the user is a small turn: the neck
turns about the wrist centre, which swings the face toward the user as it turns. From home the face
looks at the counter, away from the user, so the close camera (in front of the arm, on the user's
side) shows the face best. Each move starts once the one before has settled, so none stacks on
another. The breathing has been going since before the clip starts (it opens on an in-breath) and
fades out at its end, so the clip ends at rest and loops. The chips follow the moves one after
another: HELLO, CURIOUS, INTERESTED.
"""

import numpy as np

from puppy_motion import behaviors as B

from .. import lib, runs
from ..timeline import Label, constant_track
from . import REST_GRIP, Clip

IDLE_S = 9.6
GLANCE_ON_S = 0.9  # resting and breathing before it glances at the user
GLANCE_MAX_RAD = np.radians(18.0)  # what the 0.345 m floor allows from home
GLANCE_LIFT_M = 0.01  # it perks up this much as it turns to the user: hello
SETTLE_S = 0.1  # a move starts this long after the one before has arrived
TILT_HOLD_S = 0.45
LOOK_HOLD_S = 1.0


def build():
    e = lib.env()
    rest = constant_track(runs.HOME_JOINTS, REST_GRIP, IDLE_S, name="rest at home")
    runs.mark_near_user(rest, e.kin)
    ctx = lib.context(rest, e)
    glance = B.plan_gaze(ctx, ctx.person, GLANCE_ON_S, lean_m=0.0, max_turn_rad=GLANCE_MAX_RAD, lift_m=GLANCE_LIFT_M)
    look = B.plan_gaze(ctx, ctx.targets.container, GLANCE_ON_S)
    turn_s = max(glance[2], look[2])  # one turn time for both, so the glance blends into the look
    t_tilt = GLANCE_ON_S + turn_s + SETTLE_S
    tilt_s = B.paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, B.TILT_RAD, 0.0, 0.0], B.TILT_TURN_S, B.TILT_OVERSHOOT)
    t_untilt = t_tilt + tilt_s + TILT_HOLD_S
    t_look = t_untilt + tilt_s + SETTLE_S  # the glance holds until the tilt is back
    t_back = t_look + turn_s + LOOK_HOLD_S
    labels = [
        Label("HELLO", GLANCE_ON_S, t_tilt),
        Label("CURIOUS", t_tilt, t_look),
        Label("INTERESTED", t_look, t_back + turn_s / 2),
    ]
    parts = [
        B.breathe(ctx, -B.BREATH_PERIOD_S),  # already breathing: the clip opens on an in-breath
        B.gaze_to(
            ctx,
            ctx.person,
            GLANCE_ON_S,
            t_look,
            turn_s=turn_s,
            lean_m=0.0,
            max_turn_rad=GLANCE_MAX_RAD,
            lift_m=GLANCE_LIFT_M,
            name="glance at the user",
        ),
        B.head_tilt(ctx, t_tilt, t_untilt),
        B.gaze_to(ctx, ctx.targets.container, t_look, t_back, turn_s=turn_s, name="look at the container"),
        ctx.empty("labels", labels),
    ]
    return lib.compose(rest, parts, e)


CLIP = Clip(
    name="idle_alive",
    title="Alive at rest",
    subtitle="waiting for a task: it breathes, says hello to the user, tilts its head, eyes the container",
    camera="close",
    build=build,
    mood="CALM",
)
