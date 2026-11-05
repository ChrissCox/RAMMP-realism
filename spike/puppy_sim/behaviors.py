"""The Puppy Mode vocabulary: small motions that make the arm look alive.

Every behavior describes motion only. It plays on the clock of the (already time-warped)
functional track in its Context and returns one layer.Contribution (compound behaviors combine
their parts into one, so compose() scales them as a whole):

    def behavior(ctx, [target,] t_on, ..., beat=None) -> layer.Contribution

compose() applies the zone gain, keeps the arm clear of the person and the obstacles, and shrinks
a behavior that would still exceed a cap (logged). The defaults below fit the caps at the task's
resting postures: moves time themselves to the caps (paced_move, paced_bump) and oscillations
limit their own amplitude to them (fit_scale; noted in the log). RAMMP's limits make a 30 deg head
turn take ~2.5 s, and they keep anything fast small: at 2 Hz the 3 rad/s^2 cap allows a roll of
about a degree.

Two kinds of turn (see kin): face turns roll about the gaze wherever the face is (a tilt, a
wiggle, a shake); gravity turns tip the gaze toward or away from the floor or pan it about the
vertical (a nod, a perk, a droop, a look around). At the task's view and grasp postures the face
is rolled 90 deg, and there a head-frame nod would read as a 'no'.

Gating that is the behavior's own job (the layer's zone gains come on top): breathe and
micro_life fade out before PRECISION; wiggle and clack never act while the gripper holds something
or near the user (clack only in FREE); and while it holds a container nothing tips it: looks only
pan (level=True), the muted proud does not tilt, breathing and micro_life keep the chin level.
Big static poses stop short of the clearance floors (safe_scale) instead of being shrunk.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from . import curves
from .kin import Kin, across_gaze
from .layer import (
    EXPR_GRIP_SPEED,
    EXPR_HEAD_SPEED,
    EXPR_JOINT_ACCEL,
    EXPR_JOINT_SPEED,
    GAIN_GUARD_SAMPLES,
    GAIN_RAMP_S,
    Contribution,
    clearance_floor,
    ramped_floor,
)
from .timeline import DT, FREE, NEAR_USER, PRECISION, Label, Track

PACE = 0.8  # behaviors use at most this share of the expressive caps
SAFE_MARGIN_M = 0.002  # big static poses stop this much above an arm part's clearance floors
PRECISION_FADE_S = 0.6  # background motion (breathe, micro_life) is gone this long before PRECISION
GATE_FADE_S = 0.3  # wiggle and clack fade out this fast where they must not act
SAFE_SCALE_STEPS = 8  # bisection steps when shortening a pose to keep it clear
PATH_SAMPLES = np.linspace(0.0, 1.15, 11)  # where pacing samples a move's progress (past 1: overshoot)
SAG_AWAY_SHARE = 0.8  # a sag also drifts this much (per unit down) away from the user, horizontally

# breathe: a slow rise and fall of the head with the chin lifting on the in-breath, never two alike
BREATH_PERIOD_S = 4.0
BREATH_RISE_M = 0.0035  # amplitude (a 7 mm bob)
BREATH_NOD_RAD = np.radians(1.25)
BREATH_FADE_S = 1.0
BREATH_WOBBLE = 0.3  # breath-to-breath amplitude variation
BREATH_WOBBLE_HZ = 0.25
BREATH_SEED = 7

# micro_life: barely visible drift; unexplained constant motion reads as inattentive (ELEGNT)
MICRO_SHIFT_M = 0.002 / np.sqrt(3.0)  # per axis: at most 2 mm in all
MICRO_TURN_RAD = np.radians(0.5) / np.sqrt(3.0)  # per axis: at most 0.5 deg in all
MICRO_HZ = 0.35
MICRO_FADE_S = 1.0
MICRO_SEED = 11

# gaze_to: a quick eased turn that overshoots a little; the lean follows a moment later
GAZE_TURN_S = 1.2  # at least; longer where the caps need it
GAZE_OVERSHOOT = 0.08
GAZE_LEAN_M = 0.015
GAZE_LEAN_LAG_S = 0.1
GAZE_MAX_TURN_RAD = np.radians(40.0)
LEVEL_LOOK_MAX_RAD = np.radians(20.0)  # holding a container: it swings at the end of a 0.38 m lever

# head_tilt: the curious puppy tilt, rolled about wherever the head is looking
TILT_RAD = np.radians(20.0)
TILT_TURN_S = 0.9
TILT_OVERSHOOT = 0.12

# perk: ears up, a quick lift and look up that springs past and settles
PERK_LIFT_M = 0.02
PERK_LOOK_UP_RAD = np.radians(8.0)
PERK_RISE_S = 0.6
PERK_OVERSHOOT = 0.15
PERK_HOLD_S = 0.8
PERK_RELAX_S = 1.0

# droop: sad or sheepish, a slow sag and a look down
DROOP_SAG_M = 0.05
DROOP_LOOK_DOWN_RAD = np.radians(20.0)
DROOP_S = 1.6
DROOP_RECOVER_S = 1.4

# puppy_eyes: from under a droop, a small lift and a look up at the user, then down again
PUPPY_LIFT_M = 0.01
PUPPY_TURN_S = 0.9
PUPPY_OVERSHOOT = 0.05
PUPPY_MAX_LIFT_RAD = np.radians(20.0)  # the look up through the droop (a gravity tilt)
PUPPY_MAX_PAN_RAD = np.radians(12.0)  # and a little turn toward the user

# wiggle: a happy wag, the head swaying and rolling side to side, as big as the caps allow. The
# sway is what shows: it spreads over the shoulder, elbow and wrist joints (each with its own
# acceleration budget), while the roll loads joint 7 alone (about 1 deg at 2 Hz)
WIGGLE_HZ = 2.0
WIGGLE_CYCLES = 3
WIGGLE_MAX_ROLL_RAD = np.radians(2.0)
WIGGLE_MAX_SWAY_M = 0.012

# shake_off: a dog shaking off water, a decaying lateral shake with a roll (balanced like wiggle)
SHAKE_HZ = 2.0
SHAKE_CYCLES = 2.5
SHAKE_DECAY_S = 0.6
SHAKE_MAX_ROLL_RAD = np.radians(2.5)
SHAKE_MAX_SWAY_M = 0.015

# startle: a quick small retreat away from the user (or an event) with a look up, a freeze, a relax
STARTLE_RETREAT_M = 0.015
STARTLE_LOOK_UP_RAD = np.radians(4.0)
STARTLE_RISE_S = 0.35
STARTLE_FREEZE_S = 0.6
STARTLE_RELAX_S = 1.2

# sniff: lean in toward a target, then tiny quick pecks in and back
SNIFF_LEAN_M = 0.015
SNIFF_LEAN_S = 0.6
SNIFF_STANDOFF_M = 0.10  # the tool point never comes closer than this to the target
SNIFF_PECKS = 4
SNIFF_PECK_S = 0.45
SNIFF_MAX_DEPTH_M = 0.008

# wind_up: anticipation, a small draw-back against the coming motion, released into it
WIND_UP_SHIFT_M = 0.012
WIND_UP_TURN_RAD = np.radians(4.0)
WIND_UP_S = 0.45
WIND_UP_RELEASE_S = 0.5
WIND_UP_RELEASE_LEAD_S = 0.1  # the release starts this long before the motion does
PROBE_M = 0.02  # a motion's direction: where the eye is once it has moved this far
PROBE_RAD = np.radians(5.0)  # (or the head has turned this far)
PROBE_MAX_S = 2.5  # looking at most this far along the track
MIN_PROBE_M = 1e-3  # head moves below this (or turns below MIN_PROBE_RAD) give no direction
MIN_PROBE_RAD = np.radians(0.2)

# lead_gaze: look where you're going, the gaze runs ahead of the functional path
LEAD_S = 0.25
LEAD_FADE_S = 0.5
LEAD_SMOOTH_S = 0.12
LEAD_SMOOTH_PAD = 3.0  # smoothing reaches this many sigmas past the window
WRIST_SINGULAR_RAD = np.radians(6.0)  # |joint 6| below this: joints 5 and 7 line up (singular)
WRIST_CLEAR_RAD = np.radians(20.0)  # above this a head turn is cheap wrist motion again

# follow_through: at the end of a transit the head carries on a little, then settles
FOLLOW_TURN_RAD = np.radians(3.5)
FOLLOW_SHIFT_M = 0.008
FOLLOW_HZ = 0.8
FOLLOW_DAMPING = 0.4
FOLLOW_SETTLE_S = 1.8
FOLLOW_ONSET_S = 0.25  # it starts this long before the stop, while the arm decelerates

# lock_on: the focus stare before PRECISION, a lean in and stillness
LOCK_LEAN_M = 0.01
LOCK_LEAN_S = 0.5
LOCK_STILL_S = 0.45  # the lean is back this long before the precise motion resumes (> the gain ramp)
LOCK_LEAD_S = 0.2  # the lean starts this long before the stop, as the arm slows (overlapping action)

# double_take: look at a, glance to b, pause, snap back to a with a lean in
DOUBLE_LOOK_S = 0.9
DOUBLE_GLANCE_S = 0.6
DOUBLE_PAUSE_S = 0.5
DOUBLE_SNAP_S = 0.4
DOUBLE_HOLD_S = 0.8
DOUBLE_LEAN_M = 0.012
DOUBLE_MAX_TURN_RAD = np.radians(30.0)

# nod: acknowledge, small dips of the gaze toward the floor
NOD_RAD = np.radians(5.0)
NOD_DIP_S = 0.6  # at least; the accel cap stretches it
NOD_DIPS = 2

# bounce: excited, small vertical hops
BOUNCE_M = 0.008
BOUNCE_HOP_S = 0.4  # at least
BOUNCE_HOPS = 3

# clack: the gripper as a mouth, quick partial chomps
CLACK_SIZE = 0.25  # share of the gripper's stroke
CLACK_CHOMP_S = 0.4  # at least; the gripper's speed stretches it
CLACK_CHOMPS = 3

# doze_and_wake: curl up low, breathe slowly, nod off; a startled wake, a stretch, a look at the user
DOZE_SAG_M = 0.03
DOZE_LOOK_DOWN_RAD = np.radians(16.0)
DOZE_ROLL_RAD = np.radians(7.0)
DOZE_SETTLE_S = 2.2
DOZE_BREATH_M = 0.004
DOZE_BREATH_PERIOD_S = 5.5
DOZE_DIP_RAD = np.radians(5.0)
DOZE_DIP_SINK_S = 1.0
DOZE_DIP_CATCH_S = 0.45
DOZE_DIPS_AT = (0.38, 0.62)  # the nodding-off dips start at these shares of the doze
WAKE_RELEASE_S = 1.6
WAKE_JOLT_M = 0.01
WAKE_JOLT_LOOK_UP_RAD = np.radians(5.0)
WAKE_JOLT_S = 0.4  # about the perk's rise
WAKE_JOLT_HOLD_S = 0.3
WAKE_STRETCH_M = 0.02
WAKE_STRETCH_LOOK_UP_RAD = np.radians(12.0)
WAKE_STRETCH_S = 1.4
WAKE_STRETCH_HOLD_S = 0.5
WAKE_LOOK_HOLD_S = 1.2

# proud: success, perk + look at the user (+ a small tilt and a clack or wiggle when empty-handed)
PROUD_TILT_RAD = np.radians(12.0)
PROUD_LOOK_AT_S = 0.3  # after t_on; the tilt follows once the look has arrived
PROUD_FLOURISH_S = 1.6  # time the tilt holds for the flourish (three chomps or wags)
PROUD_BOB_M = 0.005
PROUD_BOBS = 2

# search: a puzzled look around the counter after a failed detection
SEARCH_SPOTS = ((0.0, 0.14), (0.08, -0.16), (-0.06, 0.03))  # (forward, left) m from the gaze's spot
SEARCH_LOOK_S = 1.0
SEARCH_DWELL_S = 0.5
SEARCH_TILT_RAD = np.radians(14.0)
SEARCH_MAX_TURN_RAD = np.radians(25.0)


@dataclass(frozen=True)
class Context:
    """What a behavior sees: the (warped) functional track it plays on, a Kin, the scene's
    targets (the user's head, the container's brim centre, the button, the counter height) and a
    seed for its noise."""

    track: Track
    kin: Kin
    person: np.ndarray
    container: np.ndarray
    button: np.ndarray
    table_z: float
    seed: int = 0

    @classmethod
    def of(cls, track, scene, kin=None, seed=0):
        """A context on a scene.Scene."""
        kin = kin or scene.kin()
        pl = scene.placement
        return cls(track, kin, scene.person_head, pl.center, scene.button_pos, pl.table_top_z, seed)

    def index(self, t):
        """The sample at time t (clipped to the track)."""
        return int(np.clip(round(t / DT), 0, self.track.n - 1))

    def q(self, t):
        return self.track.q[self.index(t)]

    def empty(self, name, beats=()):
        return Contribution.empty(name, self.track.n, beats)


# ----------------------------------------------------------------------------------- helpers
def _beat(name, t0, t1):
    return [Label(name, t0, t1)] if name else []


def turn_angle(offset):
    """How far a head offset turns the gaze (rad)."""
    return float(np.arccos(np.clip(np.cos(offset[4]) * np.cos(offset[5]), -1.0, 1.0)))


def _path_rates(ctx, t, offset):
    """(joint, travel): the most a joint (rad) and a head point (m) move per unit of a move's
    progress toward head offset `offset` at the posture at t, sampled along the way (a big turn
    is far from linear) and a little past it (overshoot)."""
    q = ctx.q(t)
    offset = np.asarray(offset, dtype=float)
    dq = np.array([ctx.kin.head_dq(q, s * offset) for s in PATH_SAMPLES])
    step = np.diff(PATH_SAMPLES)
    joint = float((np.abs(np.diff(dq, axis=0)).max(axis=1) / step).max())
    travel = max(ctx.kin.head_travel(q + a, b - a) / ds for a, b, ds in zip(dq[:-1], dq[1:], step))
    return joint, travel


def _paced(ctx, t, offset, design_s, slope, curve):
    """design_s, or longer if a move of `offset` (at the posture at t) whose shape peaks at
    slope * A / T (speed) and curve * A / T^2 (acceleration) would use more than PACE of a cap:
    joint speed and acceleration of the joint offsets that realise it, head speed at eye and tool."""
    joint, travel = _path_rates(ctx, t, offset)
    need = max(
        slope * joint / (PACE * EXPR_JOINT_SPEED),
        slope * travel / (PACE * EXPR_HEAD_SPEED),
        float(np.sqrt(curve * joint / (PACE * EXPR_JOINT_ACCEL))),
    )
    return max(design_s, need)


def paced_move(ctx, t, offset, design_s, overshoot=0.0):
    """Duration for easing to head offset `offset` (overshoot_ease) within PACE of the caps."""
    return _paced(ctx, t, offset, design_s, *curves.ease_peaks(overshoot))


def paced_bump(ctx, t, offset, design_s):
    """Duration for one bump out to head offset `offset` and back within PACE of the caps."""
    return _paced(ctx, t, offset, design_s, curves.BUMP_PEAK_SLOPE, curves.BUMP_PEAK_CURVE)


def fit_scale(ctx, t, offset, wave):
    """The largest scale (at most 1) at which `offset` (6,) times the unit waveform `wave` (N,)
    stays within PACE of the caps at the posture at t."""
    joint, travel = _path_rates(ctx, t, offset)
    rate = float(np.abs(np.gradient(wave, DT)).max())
    curve = float(np.abs(np.gradient(np.gradient(wave, DT), DT)).max())
    pairs = (
        (PACE * EXPR_JOINT_SPEED, joint * rate),
        (PACE * EXPR_JOINT_ACCEL, joint * curve),
        (PACE * EXPR_HEAD_SPEED, travel * rate),
    )
    return min([1.0] + [cap / used for cap, used in pairs if used > 0.0])


def _outside(track, forbidden, fade_s):
    """(N,) 1 away from the forbidden samples, 0 on them, easing (raised cosine over fade_s) to 0
    before each forbidden sample and back up after it."""
    return ramped_floor((~np.asarray(forbidden, dtype=bool)).astype(float), GAIN_GUARD_SAMPLES, fade_s)


def wrist_clear(track):
    """(N,) 0 where the functional wrist is near its singularity (|joint 6| < WRIST_SINGULAR_RAD:
    a head turn there needs the shoulder, or fast big wrist motion), 1 beyond WRIST_CLEAR_RAD,
    changing no faster than a LEAD_FADE_S raised cosine."""
    raw = curves.ramp(np.abs(track.q[:, 5]), WRIST_SINGULAR_RAD, WRIST_CLEAR_RAD)
    return ramped_floor(raw, 0, LEAD_FADE_S)


def precision_fence(track):
    """(N,) 0 on PRECISION samples and 1 away from them, easing over the gain ramp (a sample
    before and after): multiplying a contribution by it keeps it out of PRECISION."""
    return _outside(track, track.zone == PRECISION, GAIN_RAMP_S)


def _add_gravity(ctx, c, pan=0.0, tilt=0.0, roll=0.0):
    """Add gravity turns (N,) (or scalars) to c.head's rotation, at each sample's functional
    posture: pan about world z, tilt down about the axis across the gaze, roll about the gaze."""
    n = ctx.track.n
    pan, tilt, roll = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in (pan, tilt, roll))
    for i in np.nonzero((pan != 0.0) | (tilt != 0.0) | (roll != 0.0))[0]:
        c.head[i, 3:] += ctx.kin.gravity_turn(ctx.track.q[i], pan[i], tilt[i], roll[i])


def _gravity_offset(ctx, t, shift=(0.0, 0.0, 0.0), pan=0.0, tilt=0.0, roll=0.0):
    """(6,) head offset of a world shift plus a gravity turn, at the posture at t."""
    return np.concatenate([shift, ctx.kin.gravity_turn(ctx.q(t), pan, tilt, roll)])


def safe_scale(ctx, t, offset):
    """Largest s in [0, 1] for which s * offset at the posture at t keeps every arm part
    SAFE_MARGIN_M above its clearance floors (layer.clearance_floor: to the user and to the
    obstacles): a look at the user or a droop near the counter stops short instead of being
    shrunk by compose()."""
    i = ctx.index(t)
    q, grip, held = ctx.track.q[i], ctx.track.grip[i], bool(ctx.track.held[i])
    func = ctx.kin.clearances(q, grip, held)
    floor = np.minimum(clearance_floor(func[None])[0] + SAFE_MARGIN_M, func)
    offset = np.asarray(offset, dtype=float)

    def clear(s):
        return bool(np.all(ctx.kin.clearances(q + ctx.kin.head_dq(q, s * offset), grip, held) >= floor))

    if clear(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(SAFE_SCALE_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if clear(mid) else (lo, mid)
    return lo


def _limited(scale, what):
    """A note for the log when a behavior limited its own amplitude to the caps."""
    return [f"amplitude limited to x{scale:.2f} of its design by the caps ({what})"] if scale < 1.0 else []


def _keyframed(t, keys):
    """(N, 6) offsets eased (min-jerk) from key to key: keys = [(time, offset (6,))], held exactly
    before the first and after the last. Every key is a stop (zero velocity), so any sequence is
    smooth."""
    times = np.array([k for k, _ in keys], dtype=float)
    offsets = np.array([o for _, o in keys], dtype=float)
    seg = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(keys) - 2)
    a, b = times[seg], times[seg + 1]
    u = np.divide(t - a, b - a, out=np.ones_like(t, dtype=float), where=b > a)
    out = offsets[seg] + curves.min_jerk(u)[:, None] * (offsets[seg + 1] - offsets[seg])
    out[t < times[0]] = offsets[0]
    out[t >= times[-1]] = offsets[-1]
    return out


def gaze_spot(ctx, t):
    """Where the gaze (from the eye along ee z) meets the counter top at the posture at t; the
    container when the gaze does not point down at it."""
    eye, R = ctx.kin.fk(ctx.q(t), "eye")
    g = R[:, 2]
    if g[2] >= -1e-3:
        return ctx.container.copy()
    return eye + (ctx.table_z - eye[2]) / g[2] * g


# --------------------------------------------------------------------------- alive at rest
def breathe(ctx, t0=0.0, t1=None, rise_m=BREATH_RISE_M, period_s=BREATH_PERIOD_S, avoid=None, beat=None):
    """Breathing over [t0, t1] (default: the whole track): the head rises and the chin lifts on
    the in-breath, never two breaths alike. Fades in and out over BREATH_FADE_S, and out before
    PRECISION and wherever `avoid` (N,) is set (moving, a stare). While the gripper holds
    something the chin stays level (the container must not tip): only the rise remains."""
    track = ctx.track
    t1 = track.duration if t1 is None else t1
    t = track.t
    forbidden = (track.zone == PRECISION) | (False if avoid is None else avoid)
    env = curves.envelope(t, t0, t1 - BREATH_FADE_S, BREATH_FADE_S, BREATH_FADE_S)
    env = env * _outside(track, forbidden, PRECISION_FADE_S)
    wobble = 1.0 + BREATH_WOBBLE * curves.smooth_noise(track.n, DT, BREATH_WOBBLE_HZ, ctx.seed + BREATH_SEED)
    wave = curves.oscillate(t - t0, period_s) * env * wobble
    c = ctx.empty("breathe", _beat(beat, t0, t1))
    c.head[:, 2] = rise_m * wave
    _add_gravity(ctx, c, tilt=-BREATH_NOD_RAD * wave * _level(track))  # chin up on the in-breath
    return c


def micro_life(ctx, t0=0.0, t1=None, avoid=None, beat=None):
    """Barely visible smooth drift of the head over [t0, t1]: at most 2 mm and 0.5 deg (no turn
    while the gripper holds something). Never in PRECISION (it fades out before), nor wherever
    `avoid` (N,) is set."""
    track = ctx.track
    t1 = track.duration if t1 is None else t1
    forbidden = (track.zone == PRECISION) | (False if avoid is None else avoid)
    env = curves.envelope(track.t, t0, t1 - MICRO_FADE_S, MICRO_FADE_S, MICRO_FADE_S)
    env = env * _outside(track, forbidden, PRECISION_FADE_S)
    c = ctx.empty("micro_life", _beat(beat, t0, t1))
    seed = ctx.seed + MICRO_SEED
    for k in range(3):
        c.head[:, k] = MICRO_SHIFT_M * env * curves.smooth_noise(track.n, DT, MICRO_HZ, seed + k)
        turn = curves.smooth_noise(track.n, DT, MICRO_HZ, seed + 3 + k)
        c.head[:, 3 + k] = MICRO_TURN_RAD * env * turn * _level(track)
    return c


def _level(track):
    """(N,) 0 while the gripper holds something, easing to 1 away from it (GATE_FADE_S): the
    weight of anything that would tip a held container."""
    return _outside(track, track.held, GATE_FADE_S)


# ---------------------------------------------------------------------------------- looking
def gaze_to(
    ctx,
    target,
    t_on,
    t_off,
    beat=None,
    turn_s=GAZE_TURN_S,
    lean_m=GAZE_LEAN_M,
    max_turn_rad=GAZE_MAX_TURN_RAD,
    mode="shortest",
    name="gaze",
):
    """Look at world point `target` from t_on until t_off, then turn back to the functional gaze.

    A saccade-like eased turn with a GAZE_OVERSHOOT overshoot, paced to the caps; the lean toward
    the target follows GAZE_LEAN_LAG_S later (overlapping action). Look-at by IK at each
    functional posture (the eyes stay on target while the arm moves); `mode` is kin.look_at_offset's
    ('pan' keeps a held container level). The turn stops short where it would bring an arm part
    near a clearance floor (plan_gaze). It holds at least until it has arrived (t_off is pushed
    back to t_on + the paced turn), so a turn back never stacks on the turn out. Two gazes where
    one's t_off is the other's t_on (same turn_s) blend straight from one target to the next. The
    beat ends halfway through the turn back."""
    target = np.asarray(target, dtype=float)
    lean_m, max_turn_rad, turn_s = plan_gaze(ctx, target, t_on, turn_s, lean_m, max_turn_rad, mode)
    t_off = max(t_off, t_on + turn_s)
    t = ctx.track.t
    turn = curves.overshoot_envelope(t, t_on, t_off, turn_s, GAZE_OVERSHOOT)
    lean = curves.envelope(t, t_on + GAZE_LEAN_LAG_S, t_off + GAZE_LEAN_LAG_S, turn_s, turn_s)
    c = ctx.empty(name, _beat(beat, t_on, t_off + turn_s / 2))
    for i in np.nonzero((turn != 0.0) | (lean != 0.0))[0]:
        look = ctx.kin.look_at_offset(ctx.track.q[i], target, lean_m, max_turn_rad, mode)
        c.head[i, :3] = look[:3] * lean[i]
        c.head[i, 3:] = look[3:] * turn[i]
    return c


def plan_gaze(
    ctx, target, t_on, turn_s=GAZE_TURN_S, lean_m=GAZE_LEAN_M, max_turn_rad=GAZE_MAX_TURN_RAD, mode="shortest"
):
    """(lean_m, max_turn_rad, turn_s) a gaze at `target` from t_on will use: shortened where it
    would come near a clearance floor (safe_scale, at the posture at t_on) and paced to the caps."""
    first = ctx.kin.look_at_offset(ctx.q(t_on), target, lean_m, max_turn_rad, mode)
    safe = safe_scale(ctx, t_on, first * (1.0 + GAZE_OVERSHOOT))
    if safe < 1.0:
        max_turn_rad = turn_angle(first) * safe
        lean_m *= safe
        first = ctx.kin.look_at_offset(ctx.q(t_on), target, lean_m, max_turn_rad, mode)
    return lean_m, max_turn_rad, paced_move(ctx, t_on, first, turn_s, GAZE_OVERSHOOT)


def head_tilt(ctx, t_on, t_off, side=1, angle_rad=TILT_RAD, beat=None, turn_s=TILT_TURN_S):
    """The curious tilt: roll the head about its gaze by angle_rad toward `side` (+1 its right,
    -1 its left) from t_on until t_off (at least until it is up), with a little overshoot each way,
    paced to the caps. Its beat ends halfway through the tilt back."""
    roll = side * angle_rad
    turn_s = paced_move(ctx, t_on, [0, 0, 0, roll, 0, 0], turn_s, TILT_OVERSHOOT)
    t_off = max(t_off, t_on + turn_s)
    c = ctx.empty("head_tilt", _beat(beat, t_on, t_off + turn_s / 2))
    c.head[:, 3] = roll * curves.overshoot_envelope(ctx.track.t, t_on, t_off, turn_s, TILT_OVERSHOOT)
    return c


def double_take(ctx, a, b, t_on, lean_m=DOUBLE_LEAN_M, max_turn_rad=DOUBLE_MAX_TURN_RAD, beat=None):
    """A comedic beat at a still posture: look at a, glance away to b, pause, snap back to a with a
    small lean in, hold, and turn back. Each move is as quick as the caps allow."""
    q = ctx.q(t_on)
    look_a = ctx.kin.look_at_offset(q, np.asarray(a, dtype=float), 0.0, max_turn_rad, "pan_tilt")
    look_b = ctx.kin.look_at_offset(q, np.asarray(b, dtype=float), 0.0, max_turn_rad, "pan_tilt")
    look_b = look_b * safe_scale(ctx, t_on, look_b)
    d = np.asarray(a, dtype=float) - ctx.kin.fk(q, "eye")[0]
    snap = look_a + np.concatenate([lean_m * d / np.linalg.norm(d), np.zeros(3)])
    keys, t_k = [(t_on, np.zeros(6))], t_on
    for target, design_s, hold_s in (
        (look_a, DOUBLE_LOOK_S, DOUBLE_PAUSE_S),
        (look_b, DOUBLE_GLANCE_S, DOUBLE_PAUSE_S),
        (snap, DOUBLE_SNAP_S, DOUBLE_HOLD_S),
        (np.zeros(6), DOUBLE_LOOK_S, 0.0),
    ):
        t_k += paced_move(ctx, t_on, target - keys[-1][1], design_s)
        keys.append((t_k, target))
        t_k += hold_s
        keys.append((t_k, target))
    c = ctx.empty("double_take", _beat(beat, t_on, t_k))
    c.head[:] = _keyframed(ctx.track.t, keys)
    return c


def search(ctx, t_on, around=None, spots=SEARCH_SPOTS, beat=None):
    """A puzzled look around after a failed detection, at a still posture: glances at spots on the
    counter around `around` (default: where the gaze meets it), each with a tilt to alternating
    sides (each kept clear: safe_scale), then back to the functional gaze."""
    q = ctx.q(t_on)
    around = gaze_spot(ctx, t_on) if around is None else np.asarray(around, dtype=float)
    _, R = ctx.kin.fk(q, "ee")
    forward = np.array([R[0, 2], R[1, 2], 0.0])
    forward = forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 1e-6 else np.array([1.0, 0.0, 0.0])
    left = np.cross([0.0, 0.0, 1.0], forward)
    keys, t_k = [(t_on, np.zeros(6))], t_on
    for k, (fwd, lft) in enumerate(spots):
        look = ctx.kin.look_at_offset(q, around + fwd * forward + lft * left, 0.0, SEARCH_MAX_TURN_RAD)
        look[3] += (-1) ** k * SEARCH_TILT_RAD
        look *= safe_scale(ctx, t_on, look)
        t_k += paced_move(ctx, t_on, look - keys[-1][1], SEARCH_LOOK_S)
        keys += [(t_k, look), (t_k + SEARCH_DWELL_S, look)]
        t_k += SEARCH_DWELL_S
    t_k += paced_move(ctx, t_on, keys[-1][1], SEARCH_LOOK_S)
    keys.append((t_k, np.zeros(6)))
    c = ctx.empty("search", _beat(beat, t_on, t_k))
    c.head[:] = _keyframed(ctx.track.t, keys)
    return c


def sniff(ctx, target, t_on, pecks=SNIFF_PECKS, lean_m=SNIFF_LEAN_M, beat=None):
    """Lean in toward `target` (the tool point stays SNIFF_STANDOFF_M away), then `pecks` tiny
    quick pecks in and back, as deep as the caps allow (up to SNIFF_MAX_DEPTH_M), then back."""
    q = ctx.q(t_on)
    d = np.asarray(target, dtype=float) - ctx.kin.fk(q, "tool")[0]
    room = np.linalg.norm(d) - SNIFF_STANDOFF_M
    if room <= 0.0:
        c = ctx.empty("sniff", _beat(beat, t_on, t_on))
        c.notes = ["no room: the tool is already within the standoff"]
        return c
    u = d / np.linalg.norm(d)
    lean = min(lean_m, 0.5 * room)
    depth = min(SNIFF_MAX_DEPTH_M, room - lean)
    lean_s = paced_move(ctx, t_on, np.concatenate([lean * u, np.zeros(3)]), SNIFF_LEAN_S)
    peck_s = paced_bump(ctx, t_on, np.concatenate([depth * u, np.zeros(3)]), SNIFF_PECK_S)
    t_pecks = t_on + lean_s
    t_back = t_pecks + pecks * peck_s
    t = ctx.track.t
    wave = curves.bumps(t, t_pecks, pecks, peck_s)
    scale = fit_scale(ctx, t_on, np.concatenate([depth * u, np.zeros(3)]), wave)
    c = ctx.empty("sniff", _beat(beat, t_on, t_back + lean_s))
    c.head[:, :3] = u * (lean * curves.envelope(t, t_on, t_back, lean_s, lean_s) + scale * depth * wave)[:, None]
    c.notes = _limited(scale, f"pecks {scale * depth * 1000:.1f} mm")
    return c


# ------------------------------------------------------------------------------------ moods
def perk(ctx, t_on, t_off=None, lift_m=PERK_LIFT_M, look_up_rad=PERK_LOOK_UP_RAD, level=False, beat=None):
    """Ears up: a quick lift and a look up that springs past and settles, held until t_off
    (default: PERK_HOLD_S after the rise), then a slower relax. level=True (holding a container):
    the lift only."""
    look_up = 0.0 if level else look_up_rad
    peak = _gravity_offset(ctx, t_on, (0.0, 0.0, lift_m), tilt=-look_up)
    rise = paced_move(ctx, t_on, peak, PERK_RISE_S, PERK_OVERSHOOT)
    relax = paced_move(ctx, t_on, peak, PERK_RELAX_S)
    t_relax = max(t_on + rise, t_on + rise + PERK_HOLD_S if t_off is None else t_off)
    t = ctx.track.t
    env = curves.overshoot_ease((t - t_on) / rise, PERK_OVERSHOOT) - curves.ramp(t, t_relax, t_relax + relax)
    c = ctx.empty("perk", _beat(beat, t_on, t_relax + relax / 2))
    c.head[:, 2] = lift_m * env
    _add_gravity(ctx, c, tilt=-look_up * env)
    return c


def sag_shift(ctx, t, sag_m):
    """World shift of a sag of sag_m: down, drifting away from the user (horizontally) as it goes,
    so the folded elbow and wrist do not close in on them."""
    away = ctx.kin.fk(ctx.q(t), "eye")[0] - ctx.person
    away[2] = 0.0
    away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.zeros(3)
    d = np.array([0.0, 0.0, -1.0]) + SAG_AWAY_SHARE * away
    return sag_m * d / np.linalg.norm(d)


def droop(ctx, t_on, t_off, sag_m=DROOP_SAG_M, look_down_rad=DROOP_LOOK_DOWN_RAD, beat=None):
    """Sad or sheepish: a slow sag (sag_shift) and a look down from t_on, held until t_off, then a
    slow recovery. It sags only as far as the clearance floors allow (safe_scale)."""
    shift = sag_shift(ctx, t_on, sag_m)
    peak = _gravity_offset(ctx, t_on, shift, tilt=look_down_rad)
    down_s = paced_move(ctx, t_on, peak, DROOP_S)
    up_s = paced_move(ctx, t_on, peak, DROOP_RECOVER_S)
    safe = safe_scale(ctx, t_on, peak)
    env = safe * curves.envelope(ctx.track.t, t_on, t_off, down_s, up_s)
    c = ctx.empty("droop", _beat(beat, t_on, t_off + up_s / 2))
    c.notes = _limited(safe, "kept clear of the counter and the user")
    c.head[:, :3] = shift * env[:, None]
    _add_gravity(ctx, c, tilt=look_down_rad * env)
    return c


def puppy_eyes(ctx, t_on, t_off, target=None, under_tilt=DROOP_LOOK_DOWN_RAD, beat=None):
    """While drooped (by a gravity look-down of under_tilt): a small lift and a look up toward the
    user (or `target`) through the droop from t_on, held until t_off, then the gaze drops back
    down: the sheepish glance. Mostly a look up (to the user's height, at most
    PUPPY_MAX_LIFT_RAD) with a little turn toward them (PUPPY_MAX_PAN_RAD): a big pan of a face
    turned on its side would be a slow roll."""
    target = ctx.person if target is None else np.asarray(target, dtype=float)
    pan, tilt, _ = ctx.kin.aim(ctx.q(t_on), target)
    pan = float(np.clip(pan, -PUPPY_MAX_PAN_RAD, PUPPY_MAX_PAN_RAD))
    tilt = -min(max(under_tilt - tilt, 0.0), PUPPY_MAX_LIFT_RAD)  # up from the drooped gaze
    peak = _gravity_offset(ctx, t_on, (0.0, 0.0, PUPPY_LIFT_M), pan=pan, tilt=tilt)
    safe = safe_scale(ctx, t_on, peak)
    turn_s = paced_move(ctx, t_on, peak, PUPPY_TURN_S, PUPPY_OVERSHOOT)
    t_off = max(t_off, t_on + turn_s)
    t = ctx.track.t
    env = safe * curves.overshoot_envelope(t, t_on, t_off, turn_s, PUPPY_OVERSHOOT)
    c = ctx.empty("puppy_eyes", _beat(beat, t_on, t_off + turn_s / 2))
    c.head[:, 2] = PUPPY_LIFT_M * env
    _add_gravity(ctx, c, pan=pan * env, tilt=tilt * env)
    return c


def startle(ctx, t_on, source=None, retreat_m=STARTLE_RETREAT_M, look_up_rad=STARTLE_LOOK_UP_RAD, beat=None):
    """A small quick retreat away from `source` (default: the user) with a look up, as quick as
    the head-speed cap allows; a freeze; a slow relax."""
    source = ctx.person if source is None else np.asarray(source, dtype=float)
    away = ctx.kin.fk(ctx.q(t_on), "eye")[0] - source
    away = retreat_m * away / np.linalg.norm(away)
    peak = _gravity_offset(ctx, t_on, away, tilt=-look_up_rad)
    rise = paced_move(ctx, t_on, peak, STARTLE_RISE_S)
    relax = paced_move(ctx, t_on, peak, STARTLE_RELAX_S)
    t_relax = t_on + rise + STARTLE_FREEZE_S
    env = curves.ramp(ctx.track.t, t_on, t_on + rise) - curves.ramp(ctx.track.t, t_relax, t_relax + relax)
    c = ctx.empty("startle", _beat(beat, t_on, t_relax + relax / 2))
    c.head[:, :3] = away * env[:, None]
    _add_gravity(ctx, c, tilt=-look_up_rad * env)
    return c


def nod(ctx, t_on, dips=NOD_DIPS, depth_rad=NOD_RAD, beat=None):
    """Acknowledge: `dips` small dips of the gaze toward the floor and back, paced to the caps."""
    dip_s = paced_bump(ctx, t_on, _gravity_offset(ctx, t_on, tilt=depth_rad), NOD_DIP_S)
    c = ctx.empty("nod", _beat(beat, t_on, t_on + dips * dip_s))
    _add_gravity(ctx, c, tilt=depth_rad * curves.bumps(ctx.track.t, t_on, dips, dip_s))
    return c


def bounce(ctx, t_on, hops=BOUNCE_HOPS, height_m=BOUNCE_M, beat=None):
    """Excited: `hops` small vertical hops, paced to the caps."""
    hop_s = paced_bump(ctx, t_on, [0.0, 0.0, height_m, 0.0, 0.0, 0.0], BOUNCE_HOP_S)
    c = ctx.empty("bounce", _beat(beat, t_on, t_on + hops * hop_s))
    c.head[:, 2] = height_m * curves.bumps(ctx.track.t, t_on, hops, hop_s)
    return c


def _side_to_side(ctx, t_on, wave, roll_rad, sway_m):
    """(offset (6,), scale) of a roll about the gaze with a horizontal sway across it toward the
    side the head tips to, fitted to the caps for the waveform."""
    _, R = ctx.kin.fk(ctx.q(t_on), "ee")
    across = across_gaze(R)
    toward_right = -1.0 if across @ R[:, 0] > 0.0 else 1.0  # +roll tips the head to its right (-ee x)
    offset = np.concatenate([toward_right * sway_m * across, [roll_rad, 0.0, 0.0]])
    return offset, fit_scale(ctx, t_on, offset, wave)


def _sway_roll(scale, sway_m, roll_rad):
    return f"sway {scale * sway_m * 1000:.1f} mm, roll {np.degrees(scale * roll_rad):.1f} deg"


def _no_play(track):
    """Where wiggle and clack must not act: holding something, NEAR_USER or PRECISION."""
    return track.held | (track.zone == NEAR_USER) | (track.zone == PRECISION)


def wiggle(ctx, t_on, cycles=WIGGLE_CYCLES, freq_hz=WIGGLE_HZ, beat=None):
    """Happy wag: the head rolls side to side (swaying with it) `cycles` times at freq_hz, as big
    as the caps allow (up to WIGGLE_MAX_*). Never while the gripper holds something or near the
    user: zero there, fading out before."""
    t = ctx.track.t
    wave = curves.windowed_sine(t, t_on, cycles, freq_hz)
    offset, scale = _side_to_side(ctx, t_on, wave, WIGGLE_MAX_ROLL_RAD, WIGGLE_MAX_SWAY_M)
    gate = _outside(ctx.track, _no_play(ctx.track), GATE_FADE_S)
    c = ctx.empty("wiggle", _beat(beat, t_on, t_on + cycles / freq_hz))
    c.head[:] = scale * offset * (wave * gate)[:, None]
    c.notes = _limited(scale, _sway_roll(scale, WIGGLE_MAX_SWAY_M, WIGGLE_MAX_ROLL_RAD))
    return c


def shake_off(ctx, t_on, cycles=SHAKE_CYCLES, freq_hz=SHAKE_HZ, beat=None):
    """A dog shaking off water after a mishap: a decaying roll with a lateral shake, as big as the
    caps allow (up to SHAKE_MAX_*)."""
    t = ctx.track.t
    wave = curves.windowed_sine(t, t_on, cycles, freq_hz, SHAKE_DECAY_S)
    wave = wave / np.abs(wave).max()
    offset, scale = _side_to_side(ctx, t_on, wave, SHAKE_MAX_ROLL_RAD, SHAKE_MAX_SWAY_M)
    c = ctx.empty("shake_off", _beat(beat, t_on, t_on + cycles / freq_hz))
    c.head[:] = scale * offset * wave[:, None]
    c.notes = _limited(scale, _sway_roll(scale, SHAKE_MAX_SWAY_M, SHAKE_MAX_ROLL_RAD))
    return c


def clack(ctx, t_on, chomps=CLACK_CHOMPS, size=CLACK_SIZE, beat=None):
    """The gripper as a mouth: quick partial chomps (opening when it is mostly closed, closing when
    mostly open) within the 2F-85's speed. Only when it holds nothing, in FREE and not near the
    user: zero elsewhere, fading out before."""
    chomp_s = max(CLACK_CHOMP_S, curves.BUMP_PEAK_SLOPE * size / EXPR_GRIP_SPEED)
    sign = -1.0 if ctx.track.grip[ctx.index(t_on)] > 0.5 else 1.0
    gate = _outside(ctx.track, _no_play(ctx.track) | (ctx.track.zone != FREE), GATE_FADE_S)
    c = ctx.empty("clack", _beat(beat, t_on, t_on + chomps * chomp_s))
    c.grip[:] = sign * size * curves.bumps(ctx.track.t, t_on, chomps, chomp_s) * gate
    return c


# ------------------------------------------------------------------------- motion craft
def _probe(ctx, t, step):
    """The time (after t for step +1, before it for -1) at which the head has moved PROBE_M or
    turned PROBE_RAD from where it is at t, looking at most PROBE_MAX_S along the track."""
    i = ctx.index(t)
    end = ctx.index(t + step * PROBE_MAX_S)
    idx = np.arange(i, end + step, step)
    p, R = ctx.kin.fk_track(ctx.track.q[idx], "eye")
    moved = np.linalg.norm(p - p[0], axis=1)
    turned = Rotation.from_matrix(np.einsum("ji,njk->nik", R[0], R)).magnitude()
    far = np.nonzero((moved >= PROBE_M) | (turned >= PROBE_RAD))[0]
    return idx[far[0] if far.size else -1] * DT


def _coming(ctx, t0, t1):
    """(world shift unit (3,), head-frame turn unit (3,) as [roll, pitch, yaw]) of the head's
    motion on the functional track from t0 to t1 (zero vectors where it does not move)."""
    p0, R0 = ctx.kin.fk(ctx.q(t0), "eye")
    p1, R1 = ctx.kin.fk(ctx.q(t1), "eye")
    shift = p1 - p0
    rv = Rotation.from_matrix(R0.T @ R1).as_rotvec()  # head frame (R0): x pitch, y yaw, z roll
    turn = np.array([rv[2], rv[0], rv[1]])
    unit_shift = shift / np.linalg.norm(shift) if np.linalg.norm(shift) > MIN_PROBE_M else np.zeros(3)
    unit_turn = turn / np.linalg.norm(turn) if np.linalg.norm(turn) > MIN_PROBE_RAD else np.zeros(3)
    return unit_shift, unit_turn


def wind_up(ctx, t_on, t_release, shift_m=WIND_UP_SHIFT_M, turn_rad=WIND_UP_TURN_RAD, beat=None):
    """Anticipation (Takayama's forethought): from t_on the head draws back against the motion
    that starts at t_release (the direction of its first PROBE_M or PROBE_RAD on the functional
    track) and holds, then springs back into it as the motion starts. The counter-turn (wrist)
    and the draw-back shift (shoulder and elbow) each go as far as the clearance floors allow."""
    unit_shift, unit_turn = _coming(ctx, t_release, _probe(ctx, t_release, 1))
    turn = np.concatenate([np.zeros(3), -turn_rad * unit_turn])
    shift = np.concatenate([-shift_m * unit_shift, np.zeros(3)])
    back = safe_scale(ctx, t_on, turn) * turn + safe_scale(ctx, t_on, shift) * shift
    back = back * safe_scale(ctx, t_on, back)
    draw_s = paced_move(ctx, t_on, back, WIND_UP_S)
    release_s = paced_move(ctx, t_on, back, WIND_UP_RELEASE_S)
    t_go = t_release - WIND_UP_RELEASE_LEAD_S
    env = curves.envelope(ctx.track.t, t_on, t_go, draw_s, release_s)
    c = ctx.empty("wind_up", _beat(beat, t_on, t_go + release_s))
    c.head[:] = back * env[:, None]
    return c


def lead_gaze(ctx, t_on, t_off, lead_s=LEAD_S, beat=None):
    """Look where you're going: over [t_on, t_off] the gaze turns to where the functional motion
    will be looking lead_s later. Smoothed over LEAD_SMOOTH_S (the real trajectory's jerk would
    otherwise reach the expressive acceleration cap), eased in and out over LEAD_FADE_S, and
    eased out where the transit passes the wrist singularity (wrist_clear)."""
    track = ctx.track
    env = curves.envelope(track.t, t_on, t_off - LEAD_FADE_S, LEAD_FADE_S, LEAD_FADE_S) * wrist_clear(track)
    active = np.nonzero(env > 0.0)[0]
    c = ctx.empty("lead_gaze", _beat(beat, t_on, t_off))
    if not active.size:
        return c
    pad = int(round(LEAD_SMOOTH_PAD * LEAD_SMOOTH_S / DT))
    idx = np.arange(max(0, active[0] - pad), min(track.n, active[-1] + pad + 1))
    ahead = np.minimum(idx + int(round(lead_s / DT)), track.n - 1)
    _, R_now = ctx.kin.fk_track(track.q[idx], "ee")
    _, R_ahead = ctx.kin.fk_track(track.q[ahead], "ee")
    yaw_pitch = Rotation.from_matrix(np.einsum("nji,njk->nik", R_now, R_ahead)).as_euler("YXZ")[:, :2]
    yaw_pitch = gaussian_filter1d(yaw_pitch, LEAD_SMOOTH_S / DT, axis=0, mode="nearest")
    c.head[idx, 5] = yaw_pitch[:, 0] * env[idx]
    c.head[idx, 4] = yaw_pitch[:, 1] * env[idx]
    return c


def follow_through(ctx, t_end, turn_rad=FOLLOW_TURN_RAD, shift_m=FOLLOW_SHIFT_M, beat=None):
    """At the end of a transit (the functional motion stops at t_end) the head carries on a little
    in the direction it was going (over its last PROBE_M or PROBE_RAD), then settles back with a
    damped wobble."""
    unit_shift, unit_turn = _coming(ctx, _probe(ctx, t_end, -1), t_end)
    carry = np.concatenate([shift_m * unit_shift, turn_rad * unit_turn])
    t0 = t_end - FOLLOW_ONSET_S
    wave = curves.wobble(ctx.track.t, t0, FOLLOW_HZ, FOLLOW_DAMPING, FOLLOW_SETTLE_S)
    scale = fit_scale(ctx, t_end, carry, wave) if np.any(carry) else 1.0
    c = ctx.empty("follow_through", _beat(beat, t0, t0 + FOLLOW_SETTLE_S))
    c.head[:] = scale * carry * wave[:, None]
    c.notes = _limited(scale, f"{np.degrees(scale * turn_rad):.1f} deg")
    return c


def lock_on(ctx, target, t_stop, t_resume, lean_m=LOCK_LEAN_M, beat=None):
    """The focus stare, played in the hold before PRECISION (the arm stops at t_stop and the
    precise motion resumes at t_resume): a lean of lean_m toward `target` from LOCK_LEAD_S before
    the stop, held, and eased back LOCK_STILL_S before t_resume, so the head is on the functional
    pose before the gain closes; then stillness. Inside the zone nothing but timing (the
    director's slow-in)."""
    t_on = t_stop - LOCK_LEAD_S
    tool = ctx.kin.fk(ctx.q(t_stop), "tool")[0]
    d = np.asarray(target, dtype=float) - tool
    lean = np.concatenate([lean_m * d / np.linalg.norm(d), np.zeros(3)])
    lean_s = paced_move(ctx, t_on, lean, LOCK_LEAN_S)
    t_back = t_resume - LOCK_STILL_S - lean_s
    c = ctx.empty("lock_on", _beat(beat, t_on, t_resume))
    if t_back > t_on + lean_s:
        c.head[:] = lean * curves.envelope(ctx.track.t, t_on, t_back, lean_s, lean_s)[:, None]
    return c


# -------------------------------------------------------------------------------- compounds
def proud(ctx, t_on, t_off, look_until=None, flourish="clack", beat=None):
    """Success over [t_on, t_off]: perk and look at the user; holding nothing, a small tilt once
    the look has arrived, with a flourish ('clack' or 'wiggle') while it is held, if both fit
    before t_off (else noted). Holding something it is muted, so the container stays level: a
    lift-only perk, a pan-only look and a small bob; no tilt, wiggle or clack. The look holds until
    look_until (default t_off), e.g. to run on into an attentive look at the user."""
    holding = bool(ctx.track.held[ctx.index(t_on)])
    t_look = t_on + PROUD_LOOK_AT_S
    look = look_at_user(ctx, t_look, t_off if look_until is None else look_until, level=holding)
    parts, notes = [perk(ctx, t_on, t_off, level=holding), look], []
    t_arrived = t_look + user_turn_s(ctx, t_look, holding)
    if holding:
        parts.append(bounce(ctx, t_arrived, hops=PROUD_BOBS, height_m=PROUD_BOB_M))
    else:
        t_tilt = t_arrived
        tilt_s = paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, PROUD_TILT_RAD, 0.0, 0.0], TILT_TURN_S, TILT_OVERSHOOT)
        t_back = t_off - tilt_s  # the tilt is back before the look turns back
        if t_back >= t_tilt + tilt_s + PROUD_FLOURISH_S:
            parts.append(head_tilt(ctx, t_tilt, t_back, angle_rad=PROUD_TILT_RAD))
            if flourish:
                parts.append({"clack": clack, "wiggle": wiggle}[flourish](ctx, t_tilt + tilt_s))
        else:
            notes.append("no time for the tilt and the flourish before the look turns back")
    c = Contribution.combine("proud", parts, beats=_beat(beat, t_on, t_off))
    c.notes += notes
    return c


def look_at_user(ctx, t_on, t_off, level=False, max_turn_rad=GAZE_MAX_TURN_RAD, beat=None):
    """Look at the user (a gaze without a lean, a gravity pan and tilt), as far as their clearance
    floor allows; level (holding a container): pan only, at most LEVEL_LOOK_MAX_RAD. Two of these
    where one's t_off is the other's t_on (same arguments) run on as one look."""
    mode = "pan" if level else "pan_tilt"
    max_turn_rad = min(max_turn_rad, LEVEL_LOOK_MAX_RAD) if level else max_turn_rad
    return gaze_to(
        ctx, ctx.person, t_on, t_off, beat, lean_m=0.0, max_turn_rad=max_turn_rad, mode=mode, name="look at the user"
    )


def user_turn_s(ctx, t_on, level=False, max_turn_rad=GAZE_MAX_TURN_RAD):
    """How long look_at_user from t_on takes to turn (paced at the posture at t_on)."""
    max_turn_rad = min(max_turn_rad, LEVEL_LOOK_MAX_RAD) if level else max_turn_rad
    mode = "pan" if level else "pan_tilt"
    return plan_gaze(ctx, ctx.person, t_on, lean_m=0.0, max_turn_rad=max_turn_rad, mode=mode)[2]


def doze_and_wake(ctx, t_on, doze_s, beat_doze=None, beat_wake=None):
    """Drift into a curled low rest (a sag, a look down, a slight roll) with slow deep breathing
    and nodding-off dips for doze_s; then a startled wake (a quick jolt up as the rest lets go), a
    slow stretch, and a look at the user. Returns one contribution with both beats."""
    t = ctx.track.t
    t_wake = t_on + doze_s
    sag = sag_shift(ctx, t_on, DOZE_SAG_M)
    curl_peak = _gravity_offset(ctx, t_on, sag, tilt=DOZE_LOOK_DOWN_RAD, roll=DOZE_ROLL_RAD)
    deepest = curl_peak + _gravity_offset(ctx, t_on, (0.0, 0.0, -DOZE_BREATH_M), tilt=DOZE_DIP_RAD)
    settle_s = paced_move(ctx, t_on, curl_peak, DOZE_SETTLE_S)
    release_s = paced_move(ctx, t_on, curl_peak, WAKE_RELEASE_S)
    t_release = t_wake + WAKE_JOLT_S  # the rest lets go once the jolt is up
    curl = curves.envelope(t, t_on, t_release, settle_s, release_s)
    catch_s = paced_move(ctx, t_on, _gravity_offset(ctx, t_on, tilt=DOZE_DIP_RAD), DOZE_DIP_CATCH_S)
    dips = sum(_sink_and_catch(t, t_on + f * doze_s, DOZE_DIP_SINK_S, catch_s) for f in DOZE_DIPS_AT)
    breath = curves.oscillate(t - t_on, DOZE_BREATH_PERIOD_S) * curl
    rest = ctx.empty("doze")
    rest.head[:, :3] = sag * curl[:, None]
    rest.head[:, 2] += DOZE_BREATH_M * breath
    _add_gravity(ctx, rest, tilt=DOZE_LOOK_DOWN_RAD * curl + DOZE_DIP_RAD * dips, roll=DOZE_ROLL_RAD * curl)
    rest.scale = safe_scale(ctx, t_on, deepest)
    jolt = perk(ctx, t_wake, t_wake + WAKE_JOLT_S + WAKE_JOLT_HOLD_S, WAKE_JOLT_M, WAKE_JOLT_LOOK_UP_RAD)
    t_stretch = t_release + release_s
    stretch_peak = _gravity_offset(ctx, t_stretch, (0.0, 0.0, WAKE_STRETCH_M), tilt=-WAKE_STRETCH_LOOK_UP_RAD)
    stretch_s = paced_move(ctx, t_stretch, stretch_peak, WAKE_STRETCH_S)
    t_relax = t_stretch + stretch_s + WAKE_STRETCH_HOLD_S
    reach = curves.ramp(t, t_stretch, t_stretch + stretch_s) - curves.ramp(t, t_relax, t_relax + stretch_s)
    stretch = ctx.empty("stretch")
    stretch.head[:, 2] = WAKE_STRETCH_M * reach
    _add_gravity(ctx, stretch, tilt=-WAKE_STRETCH_LOOK_UP_RAD * reach)
    t_look = t_relax + stretch_s / 2
    look = look_at_user(ctx, t_look, t_look + WAKE_LOOK_HOLD_S + GAZE_TURN_S, beat="look")
    beats = _beat(beat_doze, t_on, t_wake) + _beat(beat_wake, t_wake, look.beats[0].t1)
    return Contribution.combine("doze_and_wake", [rest, jolt, stretch, look], beats=beats)


def _sink_and_catch(t, t0, sink_s, catch_s):
    """0 -> 1 slowly over sink_s from t0, then back to 0 quickly over catch_s: nodding off."""
    return curves.ramp(t, t0, t0 + sink_s) - curves.ramp(t, t0 + sink_s, t0 + sink_s + catch_s)
