"""Boop the button, plain and with Puppy Mode side by side: a scripted button press from home
(scene.press_targets: the closed fingertips along the panel's normal, the face upright).

The task: from home the arm closes its gripper on the way to the pre-press pose 2 cm out (FREE),
pauses, presses the button with the closed fingertips and backs out 6 cm (all PRECISION: contact),
and stops. Puppy Mode adds, while the arm is stopped between those moves: a lock-on before the
press (at a 2 cm standoff a lean toward the button would cross the panel's clearance floor, so the
focus is a small crouch, a lean straight down, and a stare), and after it a happy reaction (the
proud look at the user rising as it turns, then a wag with chomps while it looks). Every move plays
exactly as scripted, and the pauses stay within the 'demo' budget's 6 s.

The camera is a reverse angle from beyond the counter, right of the panel (as 'counter' is for the
container): the face looks toward it while it stares and presses (the button itself is on the
panel's far side), the user is behind, and the happy look at the user turns the face to us.
"""

import numpy as np

from .. import behaviors as B
from .. import director, render, runs, scene
from ..timeline import FREE, PRECISION, Label, concat, constant_track, joint_move, scripted_track
from . import REST_GRIP, PairClip
from .success_proud import min_jerk_grip, proud_flourish, staged, stop_at

# the scripted task
PRESS_GRIP = 1.0  # closed: the fingertips press
HOME_REST_S = 0.5
TRANSIT_S = 3.3  # home -> pre-press (shoulder and wrist about 2 rad each: under 1.2 rad/s)
CLOSE_S = 0.8  # the gripper closes during the first part of the transit
PRE_REST_S = 0.4
PRESS_S = 0.5  # 2 cm in to the button
BOOP_S = 0.25  # the fingertips on it
BACK_OUT_M = 0.06  # back out this far from the button
BACK_OUT_S = 0.7
END_REST_S = 0.4
PRESS_SEED = (-0.602, 0.215, -2.967, -2.162, -0.138, 1.308, 1.786)  # an elbow-up reach like the run's

# the beats (within the 'demo' budget: at most 6 s added)
CROUCH_M = 0.03  # the lock-on's lean: straight down, staring at the button
STARE_S = 0.4
LOCK_CHIP = "LOCK ON"
LOOK_RAD = np.radians(20.0)  # the happy look at the user (behind the arm, 162 deg from the button)
HAPPY_CYCLES = 2.0  # wag swings
HAPPY_CHOMPS = 2
HAPPY_CHIP = "HAPPY"

CAMERA = "boop"
render.CAMERAS.setdefault(  # a reverse angle from beyond the counter, right of the panel
    CAMERA, render.Camera(lookat=(0.58, 0.05, 0.37), distance=1.1, azimuth=160.0, elevation=-14.0, fovy=40.0)
)


def functional(sc, kin):
    """The plain press (see the module doc): home, transit, pre-press stop, press, boop, back out, stop."""
    home = np.array(runs.HOME_JOINTS)
    (p_pre, R), (p_press, _) = sc.press_targets(kin)
    q_pre, _, _ = kin.ik(np.array(PRESS_SEED), p_pre, R)
    transit = joint_move(home, q_pre, TRANSIT_S, PRESS_GRIP, FREE, "to the button")
    closing = min_jerk_grip(REST_GRIP, PRESS_GRIP, CLOSE_S)
    transit.grip[: len(closing)] = closing
    q_in, _ = kin.cartesian_path(q_pre, p_press, R, PRESS_S)
    q_out, _ = kin.cartesian_path(q_in[-1], p_press + BACK_OUT_M * sc.button_normal, R, BACK_OUT_S)
    func = concat(
        [
            constant_track(home, REST_GRIP, HOME_REST_S, name="home"),
            transit,
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


def lock_on(ctx, t0):
    """The focus before the press: a small crouch (a lean straight down) held in a stare, back
    before the press."""
    below = ctx.kin.fk(ctx.q(t0), "tool")[0] - np.array([0.0, 0.0, 1.0])
    t1 = t0 + B.lock_on_s(ctx, below, t0, STARE_S, CROUCH_M)
    stare = B.lock_on(ctx, below, t0, t1, lean_m=CROUCH_M)
    return [stare, ctx.empty(LOCK_CHIP, [Label(LOCK_CHIP, t0, t1)])]


def build():
    sc = scene.build()
    kin = sc.kin()
    func = functional(sc, kin)
    happy = proud_flourish(LOOK_RAD, HAPPY_CHIP, tilt=False, cycles=HAPPY_CYCLES, chomps=HAPPY_CHOMPS)
    puppy, _ = staged(func, {stop_at(func, "pre-press"): lock_on, stop_at(func, "done"): happy}, sc, kin)
    puppy.budget = director.BUDGETS[director.DEFAULT_BUDGET]  # check holds the added time to it
    return func, puppy


CLIP = PairClip(
    name="boop_button",
    title="Boop the button",
    subtitle="a scripted press from home: Puppy Mode locks on before it and turns to you happy after it, "
    "only while the arm waits",
    camera=CAMERA,
    build=build,
)
