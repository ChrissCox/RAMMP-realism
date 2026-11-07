"""Motion craft on short scripted moves at home, the camera following the head: a wind-up in the
stop before a turn to the side (back at rest when the move starts), a lead gaze riding on the move
(labelled 'later: in motion': Puppy Mode would have to run the move itself), a follow-through once
the arm is still at the side, the same move back plain; then a lock-on stare in the stop before a
short approach whose last few centimetres are PRECISION, which then runs exactly as scripted.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from puppy_motion import behaviors as B

from .. import lib, replay, runs
from ..kin import UP, across_gaze
from ..timeline import DT, PRECISION, Label, concat, constant_track, joint_move, scripted_track
from . import REST_GRIP, Clip

SIDE_SHIFT_M = (0.0, 0.10, -0.03)  # the turn to the side: the tool moves left and a little down
SIDE_TURN_RAD = (np.radians(25.0), np.radians(10.0))  # (pan left, tilt down)
MOVE_S = 2.2
START_REST_S = 1.6  # at home before the turn: the wind-up plays here
SIDE_REST_S = 2.0  # at the side: the follow-through settles
GAP_S = 0.6
APPROACH_M = 0.12  # forward, along the gaze
APPROACH_S = 2.4
APPROACH_PRECISION_M = 0.04  # the last part of the approach is PRECISION
LOCK_REST_S = 2.0  # the stop before the approach: the lock-on stares here
LOCK_TARGET_AHEAD_M = 0.3  # the stare is at a point straight ahead
PRECISE_REST_S = 1.0
RETREAT_S = 2.2
END_REST_S = 1.2
STILL_MARGIN_S = 0.1  # beats start this long after the arm stops and end this long before it moves
LEAD_CHIP = "LEAD GAZE (later: in motion)"


def _side_pose(kin, home):
    """Joints with the tool shifted SIDE_SHIFT_M and the gaze panned and tilted SIDE_TURN_RAD
    (a gravity turn: tilt down about the axis across the gaze, then pan about world z)."""
    p, R = kin.fk(home, "tool")
    pan, tilt = SIDE_TURN_RAD
    turn = Rotation.from_rotvec(pan * UP) * Rotation.from_rotvec(tilt * across_gaze(R))
    q, _, _ = kin.ik(home, p + np.array(SIDE_SHIFT_M), turn.as_matrix() @ R)
    return q


def _approach(kin, home):
    """A straight tool move forward along the gaze, its last APPROACH_PRECISION_M PRECISION; a
    precise rest at its end; the retreat home."""
    p, R = kin.fk(home, "tool")
    end = p + APPROACH_M * R[:, 2]
    q, _ = kin.cartesian_path(home, end, R, APPROACH_S)
    move = scripted_track(q, REST_GRIP, name="approach")
    tool = kin.fk_track(move.q, "tool")[0]
    move.zone[np.linalg.norm(tool - end, axis=1) < APPROACH_PRECISION_M] = PRECISION
    rest = constant_track(q[-1], REST_GRIP, PRECISE_REST_S, zone=PRECISION, name="at it")
    back = joint_move(q[-1], home, RETREAT_S, REST_GRIP, name="back")
    return move, rest, back


def _functional(kin, home):
    side = _side_pose(kin, home)
    move, precise, back = _approach(kin, home)
    func = concat(
        [
            constant_track(home, REST_GRIP, START_REST_S, name="rest"),
            joint_move(home, side, MOVE_S, REST_GRIP, name="turn"),
            constant_track(side, REST_GRIP, SIDE_REST_S, name="at the side"),
            joint_move(side, home, MOVE_S, REST_GRIP, name="back (plain)"),
            constant_track(home, REST_GRIP, LOCK_REST_S, name="stop"),
            move,
            precise,
            back,
            constant_track(home, REST_GRIP, END_REST_S, name="rest"),
        ],
        name="motion craft",
    )
    runs.mark_near_user(func, kin)
    return func


def build():
    e = lib.env()
    kin = e.kin
    func = _functional(kin, np.array(runs.HOME_JOINTS))
    ctx = lib.context(func, e)
    (h0, h1, _, _), (s0, s1, _, _), (l0, l1, _, _) = replay._stage_runs(func)[:3]  # the stops at home, at the side, before the approach
    margin = STILL_MARGIN_S
    go, arrive = h1 * DT, s0 * DT
    wind_s = B.wind_up_s(ctx, go - margin, go)
    stare = (l0 * DT + margin, l1 * DT - margin)
    eye, R = kin.fk(func.q[l0], "eye")
    precise = np.nonzero(func.zone == PRECISION)[0]
    labels = [
        Label("WIND UP", go - margin - wind_s, go - margin),
        Label(LEAD_CHIP, go - B.LEAD_S, arrive),
        Label("FOLLOW THROUGH", arrive + margin, arrive + margin + B.FOLLOW_SETTLE_S),
        Label("PLAIN MOVE", s1 * DT, l0 * DT),
        Label("LOCK ON", *stare),
        Label("FOCUS", precise[0] * DT, (precise[-1] + 1) * DT),
    ]
    parts = [
        B.wind_up(ctx, go - margin - wind_s, go - margin),
        B.lead_gaze(ctx, go - B.LEAD_S - B.LEAD_FADE_S, arrive),
        B.follow_through(ctx, arrive + margin),
        B.lock_on(ctx, eye + LOCK_TARGET_AHEAD_M * R[:, 2], *stare),
        ctx.empty("labels", labels),
    ]
    return lib.compose(func, parts, e)


CLIP = Clip(
    name="reel_craft",
    title="Puppy Mode motion craft",
    subtitle="wind-up, follow-through and lock-on play between moves; the lead gaze rides on one (later)",
    camera="close_follow",
    build=build,
)
