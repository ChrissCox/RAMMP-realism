"""Alive at rest: the arm waits at home and breathes, glances at its user, tilts its head while
looking, turns to look at the container, and settles back to looking forward.

At home the head sits 0.38 m from the user and may come no closer than 0.345 m (min of its
functional clearance and 0.35 m, less 5 mm), so the glance at the user is a small turn: the neck
turns about the wrist centre, which swings the face toward the user as it turns. From home the face
looks at the counter, away from the user, so the close camera (in front of the arm, on the user's
side) shows the face best. Each move starts once the one before has settled, so none stacks on
another.
"""

import numpy as np

from .. import behaviors as B
from .. import layer, runs, scene
from ..timeline import constant_track
from . import REST_GRIP, Clip

IDLE_S = 9.0
GLANCE_ON_S = 0.5
GLANCE_MAX_RAD = np.radians(18.0)  # what the 0.345 m floor allows from home
SETTLE_S = 0.1  # a move starts this long after the one before has arrived
TILT_HOLD_S = 0.4
LOOK_HOLD_S = 0.8


def build():
    sc = scene.build()
    kin = sc.kin()
    rest = constant_track(runs.HOME_JOINTS, REST_GRIP, IDLE_S, name="rest at home")
    runs.mark_near_user(rest, kin)
    ctx = B.Context.of(rest, sc, kin)
    glance = B.plan_gaze(ctx, ctx.person, GLANCE_ON_S, lean_m=0.0, max_turn_rad=GLANCE_MAX_RAD)
    look = B.plan_gaze(ctx, ctx.container, GLANCE_ON_S)
    turn_s = max(glance[2], look[2])  # one turn time for both, so the glance blends into the look
    t_tilt = GLANCE_ON_S + turn_s + SETTLE_S
    tilt_s = B.paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, B.TILT_RAD, 0.0, 0.0], B.TILT_TURN_S, B.TILT_OVERSHOOT)
    t_untilt = t_tilt + tilt_s + TILT_HOLD_S
    t_look = t_untilt + tilt_s + SETTLE_S  # the glance holds until the tilt is back
    parts = [
        B.breathe(ctx),
        B.gaze_to(
            ctx,
            ctx.person,
            GLANCE_ON_S,
            t_look,
            beat="HELLO",
            turn_s=turn_s,
            lean_m=0.0,
            max_turn_rad=GLANCE_MAX_RAD,
            name="glance at the user",
        ),
        B.head_tilt(ctx, t_tilt, t_untilt, beat="CURIOUS"),
        B.gaze_to(
            ctx,
            ctx.container,
            t_look,
            t_look + turn_s + LOOK_HOLD_S,
            beat="INTERESTED",
            turn_s=turn_s,
            name="look at the container",
        ),
    ]
    return layer.compose(rest, parts, kin)


CLIP = Clip(
    name="idle_alive",
    title="Alive at rest",
    subtitle="breathing, a glance at the user, a curious tilt, a look at the container",
    camera="close",
    build=build,
    mood="CALM",
)
