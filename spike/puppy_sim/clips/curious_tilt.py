"""Curious head tilt: resting at home, the arm notices the container on the counter. It looks at
it (perking up a little as it turns), sniffs toward it, gives the classic puppy double head tilt
(one side, hold, the other side, hold) and turns back to rest, letting out a breath.

From home the face looks along the counter and the container is about 31 deg away, down and
toward the user's side, where the close camera stands: the look turns the face toward the camera,
so the tilt rolls the face almost in the picture. Within the arm's envelope the look takes about
1.4 s (joint 5's speed cap paces it) and a roll about the gaze is joint 7's alone, so the tilt
pivots a little below the gaze axis, as a head tilts on its neck: the head swings to the side it
tilts to, which doubles the eye's travel and spreads the move over the shoulder and elbow. To keep
the clip short the parts overlap where they load different joints: the sniff leans in while the
look settles, the tilt starts while the sniff leans back, and the head comes level as it turns
back. Breathing pauses while it concentrates; once the head is back it lets out a breath.
"""

import numpy as np
from puppy_motion import behaviors as B
from puppy_motion import curves
from puppy_motion.compose import Contribution  # noqa: F401  (the type parts() returns)

from .. import lib, runs
from ..kin import UP
from ..timeline import Label, constant_track
from . import REST_GRIP, Clip

CALM_S = 0.5  # resting before the container catches its eye
NOTICE_LIFT_M = 0.012  # the head perks up this much as it turns to look
SNIFF_AT = 0.85  # the sniff starts this share of the way through the look (as it settles)
SNIFF_PECKS = 2
SNIFF_LEAN_M = 0.02
TILT_OVERLAP_S = 0.4  # the tilt starts this long before the sniff has leaned back
TILT_RAD = np.radians(14.0)  # each side of the double tilt
TILT_HOLD_S = 0.35
SWING_OVERSHOOT = 0.0  # the swing from one side to the other is the long move: no overshoot
BREATH_LEAD_S = 2.0  # the breathing before has been going this long when the clip starts
SIGH_AT_S = 0.4  # the breath starts this long before the head is back
SIGH_RISE_M = 0.008  # the breath: the head rises this much and the chin lifts
SIGH_LIFT_RAD = np.radians(2.5)
SIGH_IN_S = 0.8
SIGH_OUT_S = 1.1
TAIL_S = 0.3  # still once the breath is out
PROBE_S = 60.0  # the timing is measured on a still track this long first


def tilt_roll(ctx, t_on, angle_rad=TILT_RAD, hold_s=TILT_HOLD_S, side=1, level_s=0.0):
    """The classic puppy double tilt's roll about the gaze from t_on: toward `side` (+1 its
    right), hold, over to the other side, hold, and back level (over at least level_s, e.g. with a
    turn it should ride on), each move paced to the caps, the first and the last with a little
    overshoot. Returns (roll (N,), when it starts to come back level, when it is level)."""
    t = ctx.track.t
    roll = np.zeros(ctx.track.n)
    moves = (
        (side * angle_rad, B.TILT_OVERSHOOT, hold_s, 0.0),
        (-side * angle_rad, SWING_OVERSHOOT, hold_s, 0.0),
        (0.0, B.TILT_OVERSHOOT, 0.0, level_s),
    )
    level, t_k = 0.0, t_on
    for target, overshoot, hold, at_least in moves:
        t_start, step = t_k, target - level
        move_s = max(at_least, B.paced_move(ctx, t_k, [0.0, 0.0, 0.0, step, 0.0, 0.0], B.TILT_TURN_S, overshoot))
        roll += step * curves.overshoot_ease((t - t_k) / move_s, overshoot)
        level, t_k = target, t_k + move_s + hold
    return roll, t_start, t_k


def double_tilt(ctx, roll):
    """The tilt as a contribution: `roll` (N,) about the gaze."""
    c = ctx.empty("double tilt")
    c.head[:, 3] = roll
    return c


def sigh(ctx, t_on, rise_m=SIGH_RISE_M, lift_rad=SIGH_LIFT_RAD, in_s=SIGH_IN_S, out_s=SIGH_OUT_S):
    """One slow breath from t_on: the head rises and the chin lifts over in_s, then settles over
    out_s."""
    t = ctx.track.t
    env = curves.ramp(t, t_on, t_on + in_s) - curves.ramp(t, t_on + in_s, t_on + in_s + out_s)
    c = ctx.empty("breath")
    c.head[:, :3] = UP * rise_m * env[:, None]
    B._add_gravity(ctx, c, tilt=-lift_rad * env)
    return c


def parts(ctx):
    """(contributions, when it is done) on the still track in ctx."""
    container = ctx.targets.container
    t_notice = CALM_S
    _, _, turn_s, _ = B.plan_gaze(ctx, container, t_notice, lift_m=NOTICE_LIFT_M)
    sniff = B.sniff(ctx, container, t_notice + SNIFF_AT * turn_s, pecks=SNIFF_PECKS, lean_m=SNIFF_LEAN_M)
    t_tilt = sniff.span()[1] - TILT_OVERLAP_S
    roll, t_off, t_back = tilt_roll(ctx, t_tilt, level_s=turn_s)
    look = B.gaze_to(ctx, container, t_notice, t_off, lift_m=NOTICE_LIFT_M, name="look at the container")
    look.labels = [Label("CURIOUS", t_notice, t_off + turn_s / 2)]
    breath = sigh(ctx, t_back - SIGH_AT_S)
    t = ctx.track.t
    before = B.breathe(ctx, -BREATH_LEAD_S, t_back, avoid=t >= t_notice)  # holds its breath once it notices
    return [before, look, sniff, double_tilt(ctx, roll), breath], breath.span()[1]


def rest_and_parts(e=None):
    """(the still track at home, its contributions): timed on a long still track first, then
    played on one just long enough."""
    e = e or lib.env()
    home = np.array(runs.HOME_JOINTS)
    _, t_done = parts(lib.context(constant_track(home, REST_GRIP, PROBE_S), e))
    rest = constant_track(home, REST_GRIP, t_done + TAIL_S, name="rest at home")
    runs.mark_near_user(rest, e.kin)
    return rest, parts(lib.context(rest, e))[0]


def build():
    e = lib.env()
    rest, contributions = rest_and_parts(e)
    return lib.compose(rest, contributions, e)


CLIP = Clip(
    name="curious_tilt",
    title="Curious head tilt",
    subtitle="at rest, waiting for a task: the container catches its eye; a look, a sniff, a double head tilt",
    camera="close",
    build=build,
    mood="CALM",
)
