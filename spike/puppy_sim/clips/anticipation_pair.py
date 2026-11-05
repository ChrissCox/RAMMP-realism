"""Anticipation, plain vs Puppy Mode, on one free move: from the real run's view pose over the
counter, the arm turns to look up and to the right (a glance away from the container). Before the
move Puppy Mode winds up (a small draw-back against the coming move, at rest again before the move
starts) and once the arm has arrived and stopped it follows through (the head carries on a little
the way it was going, then settles with a small wobble). Both play only while the arm is stopped
(the boundary tier, placed by the director, whose other beats are left out), so the move itself
runs as scripted.

A scripted move rather than the real home -> view: at home the draw-back would point at the user,
and the clearance floor there cuts it to a few pixels; here, 0.9 m from the user, it has room.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .. import director, runs, scene
from ..kin import UP, across_gaze
from ..timeline import DT, Label, concat, constant_track, joint_move
from . import PairClip

VIEW_T = 4.8  # the real run's view pose: just arrived, before detect
SHIFT_M = (0.0, -0.10, 0.05)  # the move: the tool goes right and up (still 0.9 m from the user)
TURN_RAD = (np.radians(-22.0), np.radians(-12.0))  # (pan right, tilt up): it turns its face toward the camera
REST_S = 1.6  # still before the move is asked for
PLAN_S = 0.8  # the move's phase starts this long before the arm moves (the real run's approach: 0.77 s planning)
MOVE_S = 2.2
ARRIVE_S = 3.2  # still after it: the follow-through, then calm
ONLY = ("wind_up", "follow_through")


def _side_pose(kin, q):
    """Joints with the tool shifted SHIFT_M and the gaze panned and tilted TURN_RAD (gravity turns)."""
    p, R = kin.fk(q, "tool")
    pan, tilt = TURN_RAD
    turn = Rotation.from_rotvec(pan * UP) * Rotation.from_rotvec(tilt * across_gaze(R))
    q_side, _, _ = kin.ik(q, p + np.array(SHIFT_M), turn.as_matrix() @ R)
    return q_side


def functional(kin):
    """Rest at the real view pose, the scripted move (its phase starting PLAN_S before the arm
    moves, as a driver call does while it plans), rest where it arrives."""
    real = runs.functional_track(runs.load_run(scene.REFERENCE_RUN), kin)
    i = int(round(VIEW_T / DT))
    q_view, grip = real.q[i], float(real.grip[i])
    side = _side_pose(kin, q_view)
    func = concat(
        [
            constant_track(q_view, grip, REST_S, name="at the view pose"),
            constant_track(q_view, grip, PLAN_S, name="look up and right"),
            joint_move(q_view, side, MOVE_S, grip, name="look up and right"),
            constant_track(side, grip, ARRIVE_S, name="there"),
        ],
        name="a free move",
    )
    move = func.phases[1]
    func.phases[1:3] = [Label(move.name, move.t0, func.phases[2].t1)]  # planning and moving: one phase
    runs.mark_near_user(func, kin)
    return func


def build():
    sc = scene.build()
    kin = sc.kin()
    func = functional(kin)
    skip = tuple(b for b in director.BEATS if b not in ONLY)
    puppy, _ = director.direct_track(func, "grab_container", sc, kin, skip=skip)
    return func, puppy


CLIP = PairClip(
    name="anticipation_pair",
    title="Anticipation: a wind-up before a move, a follow-through after it",
    subtitle="while stopped before the move it draws back; the move runs as planned; it lands, carries on, settles",
    camera="close_follow",
    build=build,
    mood="CALM",
)
