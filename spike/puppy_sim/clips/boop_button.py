"""Boop the button, plain and with Puppy Mode side by side: a scripted button press from home
(scene.press_targets: the closed fingertips along the panel's normal, the face upright).

The task: from home the arm closes its gripper on the way to a staging pose 8 cm out from the button
and then to the pre-press pose 2 cm out (FREE), pauses, presses the button with the closed fingertips and backs out 6 cm (all PRECISION: contact),
and stops. Puppy Mode adds, while the arm is stopped between those moves: a lock-on before the
press (in the stop at the staging pose: the move that follows ends at a precise pose, so the
Director leans in and stares at the button), and after it a happy reaction (the
proud look at the user rising as it turns, then a wag with chomps while it looks). Every move plays
exactly as scripted, and the pauses stay within the 'demo' budget's 6 s.

The camera is a reverse angle from beyond the counter, right of the panel (as 'counter' is for the
container): the face looks toward it while it stares and presses (the button itself is on the
panel's far side), the user is behind, and the happy look at the user turns the face to us.
"""

import numpy as np
from puppy_motion.behaviors import Targets

from .. import lib, render, replay, runs
from ..timeline import DT, FREE, PRECISION, concat, constant_track, joint_move, scripted_track
from . import REST_GRIP, PairClip
from ..scene import PRESS_STANDOFF_M
from .success_proud import min_jerk_grip

# the scripted task
PRESS_GRIP = 1.0  # closed: the fingertips press
HOME_REST_S = 0.5
TRANSIT_S = 3.4  # home -> staging pose (shoulder and wrist about 2 rad each: under 1.2 rad/s)
CLOSE_S = 0.8  # the gripper closes during the first part of the transit
STAGE_M = 0.08  # the staging pose: this far out from the button along its normal
STAGE_REST_S = 0.4
APPROACH_S = 1.2  # staging -> pre-press
PRE_REST_S = 0.4
PRESS_S = 0.5  # 2 cm in to the button
BOOP_S = 0.25  # the fingertips on it
BACK_OUT_M = 0.06  # back out this far from the button
BACK_OUT_S = 0.7
END_REST_S = 0.4
BIGGER = {"director.LOCK_LEAN_M": 2.0}  # the lock-on leans in further than the director's, to show
EDGE_S = 3 * DT  # an event arrives this far into a stop
PRESS_SEED = (-0.602, 0.215, -2.967, -2.162, -0.138, 1.308, 1.786)  # an elbow-up reach like the run's

CAMERA = "boop"
render.CAMERAS.setdefault(  # a reverse angle from beyond the counter, right of the panel
    CAMERA, render.Camera(lookat=(0.58, 0.05, 0.37), distance=1.1, azimuth=160.0, elevation=-14.0, fovy=40.0)
)


def functional(sc, kin):
    """The plain press (see the module doc): home, transit, pre-press stop, press, boop, back out, stop."""
    home = np.array(runs.HOME_JOINTS)
    (p_pre, R), (p_press, _) = sc.press_targets(kin)
    q_pre, _, _ = kin.ik(np.array(PRESS_SEED), p_pre, R)
    q_stage, _, _ = kin.ik(q_pre, p_pre + (STAGE_M - PRESS_STANDOFF_M) * sc.button_normal, R)
    transit = joint_move(home, q_stage, TRANSIT_S, PRESS_GRIP, FREE, "to the staging pose")
    closing = min_jerk_grip(REST_GRIP, PRESS_GRIP, CLOSE_S)
    transit.grip[: len(closing)] = closing
    q_in, _ = kin.cartesian_path(q_pre, p_press, R, PRESS_S)
    q_out, _ = kin.cartesian_path(q_in[-1], p_press + BACK_OUT_M * sc.button_normal, R, BACK_OUT_S)
    func = concat(
        [
            constant_track(home, REST_GRIP, HOME_REST_S, name="home"),
            transit,
            constant_track(q_stage, PRESS_GRIP, STAGE_REST_S, name="staging"),
            joint_move(q_stage, q_pre, APPROACH_S, PRESS_GRIP, FREE, "to the pre-press pose"),
            constant_track(q_pre, PRESS_GRIP, PRE_REST_S, name="pre-press"),
            scripted_track(q_in, PRESS_GRIP, PRECISION, "press"),
            constant_track(q_in[-1], PRESS_GRIP, BOOP_S, PRECISION, "boop"),
            scripted_track(q_out, PRESS_GRIP, PRECISION, "back out"),
            constant_track(q_out[-1], PRESS_GRIP, END_REST_S, name="done"),
        ],
        name="press the button",
    )
    runs.mark_near_user(func, kin)
    return func


def build():
    """Plain, and Puppy Mode through puppy_motion's Director as the button_press capability would drive
    it: asked to go, it reports the move to the pre-press pose (which ends at a precise pose: the
    director's lock-on stare before it, aimed at the button), the press and the back-out (PRECISION: no
    beat), and the success."""
    from puppy_motion import director as pm

    e = lib.env()
    func = functional(e.sc, e.kin)
    (_, _), (p_press, _) = e.sc.press_targets(e.kin)
    targets = Targets(person_head=e.person_head, container=e.container, button=p_press)
    home, stage, pre, done = replay._stage_runs(func)  # the four stops: (i0, i1, before, after)
    t_done = done[0] * DT + EDGE_S
    events = replay.sorted_events(
        [
            (0.1, pm.Phase("stage", "")),
            (home[1] * DT - EDGE_S, pm.BeforeMove(False)),
            (stage[0] * DT, pm.AfterMove()),
            (stage[0] * DT + EDGE_S, pm.Phase("approach", "")),
            (stage[1] * DT - EDGE_S, pm.BeforeMove(True)),  # the move to the pre-press pose ends at a precise pose
            (pre[0] * DT, pm.AfterMove()),
            (pre[0] * DT + EDGE_S, pm.Phase("press", "")),
            (t_done, pm.Phase("done", "done")),
            (t_done, pm.Succeeded()),
        ]
    )
    with replay.exaggerated(BIGGER):
        puppy = replay.direct_scripted(func, events, (("button_press", 0.0),), targets=targets, e=e)
    return func, replay.display_names(puppy)


CLIP = PairClip(
    name="boop_button",
    title="Boop the button",
    subtitle="a scripted press from home: Puppy Mode locks on before it and turns to you happy after it, "
    "only while the arm waits",
    camera=CAMERA,
    build=build,
)
