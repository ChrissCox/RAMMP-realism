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

from .. import behaviors as B
from .. import curves, layer, runs, scene
from ..kin import UP, head_rotation
from ..timeline import Label, constant_track
from . import REST_GRIP, Clip

CALM_S = 0.5  # resting before the container catches its eye
NOTICE_LIFT_M = 0.012  # the head perks up this much as it turns to look
SNIFF_AT = 0.85  # the sniff starts this share of the way through the look (as it settles)
SNIFF_PECKS = 2
SNIFF_LEAN_M = 0.02
TILT_OVERLAP_S = 0.4  # the tilt starts this long before the sniff has leaned back
TILT_RAD = np.radians(14.0)  # each side of the double tilt
TILT_PIVOT_M = 0.06  # the tilt pivots up to this far below the gaze axis (on its neck)
PIVOT_BLEND = 1.0 / 3.0  # the pivot depth blends from one side's to the other's within this share of the tilt
PIVOT_HEADROOM = 1.2  # the pivots are sized for a tilt this much bigger (its overshoot, the overlapping parts)
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


def _pivot_shift(R, roll, depth_m):
    """World shift (3,) that makes a roll about the gaze pivot depth_m below the gaze axis, in the
    head frame R."""
    return R @ np.array([-np.sin(roll), np.cos(roll) - 1.0, 0.0]) * depth_m


def safe_pivots(ctx, t, look, roll_rad, pivot_m=TILT_PIVOT_M):
    """(depth for +roll_rad, depth for -roll_rad): the deepest pivot (at most pivot_m) each side of
    the tilt may use at the look's pose at t, keeping every arm part SAFE_MARGIN_M above its
    clearance floors (beside the user the head may not swing toward them)."""
    i = ctx.index(t)
    q, grip, held = ctx.track.q[i], ctx.track.grip[i], bool(ctx.track.held[i])
    func = ctx.kin.clearances(q, grip, held)
    floor = np.minimum(layer.clearance_floor(func[None])[0] + B.SAFE_MARGIN_M, func)
    _, R = ctx.kin.fk(q, "ee")
    gazed = R @ head_rotation(look.head[i])

    def clear(roll, depth):
        offset = look.head[i] + np.concatenate([_pivot_shift(gazed, roll, depth), [roll, 0.0, 0.0]])
        return bool(np.all(ctx.kin.clearances(q + ctx.kin.head_dq(q, offset), grip, held) >= floor))

    depths = []
    for roll in (roll_rad, -roll_rad):
        lo, hi = 0.0, pivot_m
        if clear(roll, hi):
            lo = hi
        for _ in range(B.SAFE_SCALE_STEPS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if clear(roll, mid) else (lo, mid)
        depths.append(lo)
    return tuple(depths)


def double_tilt(ctx, roll, look, depths, angle_rad=TILT_RAD):
    """The tilt as a contribution: `roll` (N,) about the gaze, in the head frame the look (a
    contribution) gives at each sample, pivoting below the gaze axis by depths[0] on the +roll side
    and depths[1] on the other, blended smoothly through level, so the head swings to the side it
    tilts to as far as the clearance floors allow."""
    mid, half = (depths[0] + depths[1]) / 2.0, (depths[0] - depths[1]) / 2.0
    c = ctx.empty("double tilt")
    c.head[:, 3] = roll
    for i in np.nonzero(roll)[0]:
        _, R = ctx.kin.fk(ctx.track.q[i], "ee")
        depth = mid + half * np.tanh(roll[i] / (angle_rad * PIVOT_BLEND))
        c.head[i, :3] = _pivot_shift(R @ head_rotation(look.head[i]), roll[i], depth)
    return c


def sigh(ctx, t_on, rise_m=SIGH_RISE_M, lift_rad=SIGH_LIFT_RAD, in_s=SIGH_IN_S, out_s=SIGH_OUT_S):
    """One slow breath from t_on: the head rises and the chin lifts over in_s, then settles over
    out_s."""
    t = ctx.track.t
    env = curves.ramp(t, t_on, t_on + in_s) - curves.ramp(t, t_on + in_s, t_on + in_s + out_s)
    c = ctx.empty("breath")
    for i in np.nonzero(env)[0]:
        c.head[i, :3] = UP * rise_m * env[i]
        c.head[i, 3:] = ctx.kin.gravity_turn(ctx.track.q[i], 0.0, -lift_rad * env[i])
    return c


def parts(ctx):
    """(contributions, when it is done) on the still track in ctx."""
    t_notice = CALM_S
    _, _, turn_s, _ = B.plan_gaze(ctx, ctx.container, t_notice, lift_m=NOTICE_LIFT_M)
    sniff = B.sniff(ctx, ctx.container, t_notice + SNIFF_AT * turn_s, pecks=SNIFF_PECKS, lean_m=SNIFF_LEAN_M)
    t_tilt = sniff.span()[1] - TILT_OVERLAP_S
    roll, t_off, t_back = tilt_roll(ctx, t_tilt, level_s=turn_s)
    look = B.gaze_to(ctx, ctx.container, t_notice, t_off, lift_m=NOTICE_LIFT_M, name="look at the container")
    look.beats = [Label("CURIOUS", t_notice, t_off + turn_s / 2)]
    depths = safe_pivots(ctx, t_tilt, look, TILT_RAD * PIVOT_HEADROOM)
    breath = sigh(ctx, t_back - SIGH_AT_S)
    t = ctx.track.t
    before = B.breathe(ctx, -BREATH_LEAD_S, t_back, avoid=t >= t_notice)  # holds its breath once it notices
    return [before, look, sniff, double_tilt(ctx, roll, look, depths), breath], breath.span()[1]


def rest_and_parts(sc, kin):
    """(the still track at home, its contributions): timed on a long still track first, then
    played on one just long enough."""
    home = np.array(runs.HOME_JOINTS)
    _, t_done = parts(B.Context.of(constant_track(home, REST_GRIP, PROBE_S), sc, kin))
    rest = constant_track(home, REST_GRIP, t_done + TAIL_S, name="rest at home")
    runs.mark_near_user(rest, kin)
    return rest, parts(B.Context.of(rest, sc, kin))[0]


def build():
    sc = scene.build()
    kin = sc.kin()
    rest, contributions = rest_and_parts(sc, kin)
    return layer.compose(rest, contributions, kin)


CLIP = Clip(
    name="curious_tilt",
    title="Curious head tilt",
    subtitle="at rest, waiting for a task: the container catches its eye; a look, a sniff, a double head tilt",
    camera="close",
    build=build,
    mood="CALM",
)
