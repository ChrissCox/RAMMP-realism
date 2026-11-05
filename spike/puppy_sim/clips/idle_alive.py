"""Alive at rest: the arm waits at home and breathes, glances at its user, tilts its head,
looks at the container, and settles back to looking forward.

At home the head sits 0.38 m from the user and may come no closer than 0.345 m (min of its
functional clearance and 0.35 m, less 5 mm), so the glance at the user is a small turn: the
neck turns about the wrist centre, which swings the face toward the user as it turns.
"""

import numpy as np

from .. import behaviors, layer, runs, scene
from ..timeline import constant_track
from . import Clip

IDLE_S = 9.8
REST_GRIP = 0.79  # the reference run's gripper while it waited at home
TURN_S = 2.0  # both gazes, so the glance blends straight into the look at the container
GLANCE_ON_S, GLANCE_OFF_S = 0.6, 4.0
GLANCE_MAX_RAD = np.radians(18.0)  # what the 0.345 m floor allows from home
TILT_ON_S, TILT_OFF_S = 2.0, 3.6
LOOK_ON_S, LOOK_OFF_S = 4.0, 6.8


def build():
    sc = scene.build()
    kin = sc.kin()
    rest = constant_track(runs.HOME_JOINTS, REST_GRIP, IDLE_S, name="rest at home")
    runs.mark_near_user(rest, kin)
    ctx = behaviors.Context.of(rest, sc, kin)
    parts = [
        behaviors.breathe(ctx),
        behaviors.gaze_to(
            ctx,
            ctx.person,
            GLANCE_ON_S,
            GLANCE_OFF_S,
            beat="HELLO",
            turn_s=TURN_S,
            lean_m=0.0,
            max_turn_rad=GLANCE_MAX_RAD,
            name="glance at the user",
        ),
        behaviors.head_tilt(ctx, TILT_ON_S, TILT_OFF_S, beat="CURIOUS"),
        behaviors.gaze_to(
            ctx, ctx.container, LOOK_ON_S, LOOK_OFF_S, beat="INTERESTED", turn_s=TURN_S, name="look at the container"
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
