"""The Puppy Mode vocabulary: small motions that make the arm look alive.

Every behavior describes motion only. It plays on the clock of the (already time-warped)
functional track in its Context and returns one compose.Contribution (compound behaviors combine
their parts into one, so compose() scales them as a whole):

    def behavior(ctx, [target,] t_on, ..., beat=None) -> Contribution

compose() applies the zone gain, keeps the arm clear of the person and the obstacles, and shrinks
a behavior that would still exceed a cap (logged). The defaults below fit the caps at the task's
resting postures: moves time themselves to the caps of their zone (paced_move, paced_bump) and
oscillations limit their own amplitude to them (fit_scale; noted in the log). In FREE a 25 deg head
turn takes about a second; anything quick stays small, because a joint may accelerate at only
3 rad/s^2 (the task's own peak): at 1.3 Hz that allows a roll of about 2 deg, so a happy wag reads
as a sway of the whole head rather than a waggle of the wrist.

The motion-craft beats are self-contained so they can play between arm calls: a wind-up comes back
to rest before its move starts, a follow-through starts once the arm is still, a lock-on leans and
comes back within its stare. Only lead_gaze rides on a moving functional trajectory.

Two kinds of turn (see gaze): face turns roll about the gaze wherever the face is (a tilt, a
wiggle, a shake); gravity turns tip the gaze toward or away from the floor or pan it about the
vertical (a nod, a perk, a droop, a look around). At the task's view and grasp postures the face
is rolled 90 deg, and there a head-frame nod would read as a 'no'.

Gating that is the behavior's own job (the layer's zone gains come on top): breathe and
micro_life fade out before PRECISION; wiggle and clack never act while the gripper holds something
or near the user (clack only in FREE); and while it holds a container nothing tips it: looks only
pan (level=True), the muted proud does not tilt, breathing and micro_life keep the chin level.
Big static poses stop short of the clearance floors (safe_scale) instead of being shrunk.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import curves, gaze
from .caps import DEFAULT_CAPS, GAIN_GUARD_SAMPLES, GAIN_RAMP_S, CapSet, ramped_floor, sample_caps
from .compose import (
    EXPR_GRIP_SPEED,
    HEAD_SPEED_SITES,
    Contribution,
    clearance_floor,
    clearances_of,
    head_dq,
)
from .gaze import UP, across_gaze
from .track import DT, Label, Track, Zone

FREE, NEAR_USER, PRECISION = Zone.FREE, Zone.NEAR_USER, Zone.PRECISION
WRIST_JOINTS = slice(4, 7)  # joints 5-7

PACE = 0.9  # behaviors use at most this share of the expressive caps
SAFE_MARGIN_M = 0.002  # big static poses stop this much above an arm part's clearance floors
PRECISION_FADE_S = 0.6  # background motion (breathe, micro_life) is gone this long before PRECISION
GATE_FADE_S = 0.3  # wiggle and clack fade out this fast where they must not act
SAFE_SCALE_STEPS = 8  # bisection steps when shortening a pose to keep it clear
PATH_SAMPLES = np.linspace(0.0, 1.15, 11)  # where pacing samples a move's progress (past 1: overshoot)
GAZE_AIM_M = gaze.FALLBACK_AIM_M  # with no target, aim this far along the gaze (0.35 m from the eye)
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
GAZE_TURN_S = 0.7  # at least; longer where the caps need it (about 1 s for 25 deg in FREE)
GAZE_OVERSHOOT = 0.06
GAZE_LEAN_M = 0.015
GAZE_LEAN_LAG_S = 0.1
GAZE_MAX_TURN_RAD = np.radians(40.0)
LEVEL_LOOK_MAX_RAD = np.radians(20.0)  # holding a container: it swings at the end of a 0.38 m lever

# head_tilt: the curious puppy tilt, rolled about wherever the head is looking
TILT_RAD = np.radians(20.0)
TILT_TURN_S = 0.6
TILT_OVERSHOOT = 0.08

# perk: ears up, a quick lift and look up that springs past and settles
PERK_LIFT_M = 0.025
PERK_LOOK_UP_RAD = np.radians(10.0)
PERK_RISE_S = 0.45
PERK_OVERSHOOT = 0.15
PERK_HOLD_S = 0.5
PERK_RELAX_S = 0.7

# droop: sad or sheepish, a slow sag and a look down
DROOP_SAG_M = 0.05
DROOP_LOOK_DOWN_RAD = np.radians(20.0)
DROOP_S = 1.1
DROOP_RECOVER_S = 0.9

# puppy_eyes: from under a droop, a small lift and a look up at the user, then down again
PUPPY_LIFT_M = 0.01
PUPPY_TURN_S = 0.6
PUPPY_OVERSHOOT = 0.05
PUPPY_MAX_LIFT_RAD = np.radians(20.0)  # the look up through the droop (a gravity tilt)
PUPPY_MAX_PAN_RAD = np.radians(12.0)  # and a little turn toward the user

# wiggle: a happy sway, the whole head swinging side to side with a roll toward the side it swings
# to, as big as the caps allow (fit_scale). The sway is what shows: it spreads over the shoulder,
# elbow and wrist (each joint with its own acceleration budget); the roll loads joint 7 alone
WIGGLE_HZ = 1.25
WIGGLE_CYCLES = 2.5
WIGGLE_MAX_SWAY_M = 0.035  # eye travel of the sway, peak to peak
WIGGLE_MAX_TURN_RAD = np.radians(6.0)  # per wrist joint

# shake_off: a dog shaking off water, the same sway and wrist wag as a quicker shiver dying away
SHAKE_HZ = 1.5
SHAKE_CYCLES = 3.0
SHAKE_DECAY_S = 1.0
SHAKE_MAX_SWAY_M = 0.035
SHAKE_MAX_TURN_RAD = np.radians(6.0)

# startle: a quick small retreat away from the user (or an event) with a look up, a freeze, a relax
STARTLE_RETREAT_M = 0.02
STARTLE_LOOK_UP_RAD = np.radians(6.0)
STARTLE_RISE_S = 0.3
STARTLE_FREEZE_S = 0.5
STARTLE_RELAX_S = 0.8

# sniff: lean in toward a target, then quick pecks in and back
SNIFF_LEAN_M = 0.025
SNIFF_LEAN_S = 0.45
SNIFF_STANDOFF_M = 0.10  # the tool point never comes closer than this to the target
SNIFF_PECKS = 3
SNIFF_PECK_S = 0.4
SNIFF_MAX_DEPTH_M = 0.015

# wind_up: anticipation before a move, a small draw-back against it and back, at rest when it starts
WIND_UP_SHIFT_M = 0.015
WIND_UP_TURN_RAD = np.radians(5.0)
WIND_UP_S = 0.4
WIND_UP_HOLD_S = 0.1  # drawn back this long
WIND_UP_RELEASE_S = 0.35
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

# follow_through: once the arm is still after a move, the head carries on a little, then settles
FOLLOW_TURN_RAD = np.radians(4.0)
FOLLOW_SHIFT_M = 0.015
FOLLOW_HZ = 0.85
FOLLOW_DAMPING = 0.45
FOLLOW_SETTLE_S = 1.3

# lock_on: the focus stare before PRECISION, a lean toward the target, held, and stillness
LOCK_LEAN_M = 0.015
LOCK_LEAN_S = 0.35
LOCK_STILL_S = 0.2  # the lean is back this long before the stare ends (and the next move may start)

# double_take: look at a, glance to b, pause, snap back to a with a lean in
DOUBLE_LOOK_S = 0.6
DOUBLE_GLANCE_S = 0.4
DOUBLE_PAUSE_S = 0.35
DOUBLE_SNAP_S = 0.3
DOUBLE_HOLD_S = 0.6
DOUBLE_LEAN_M = 0.015
DOUBLE_MAX_TURN_RAD = np.radians(30.0)

# nod: acknowledge, dips of the gaze toward the floor
NOD_RAD = np.radians(10.0)
NOD_DIP_S = 0.45  # at least; the accel cap stretches it
NOD_DIPS = 2

# bounce: excited, vertical hops
BOUNCE_M = 0.02
BOUNCE_HOP_S = 0.35  # at least
BOUNCE_HOPS = 3

# clack: the gripper as a mouth, quick partial chomps
CLACK_SIZE = 0.35  # share of the gripper's stroke
CLACK_CHOMP_S = 0.4  # at least; the gripper's speed stretches it
CLACK_CHOMPS = 3

# doze_and_wake: curl up low, breathe slowly, nod off; a startled wake, a stretch, a look at the user
DOZE_SAG_M = 0.03
DOZE_LOOK_DOWN_RAD = np.radians(16.0)
DOZE_ROLL_RAD = np.radians(7.0)
DOZE_SETTLE_S = 2.0
DOZE_BREATH_M = 0.004
DOZE_BREATH_PERIOD_S = 5.0
DOZE_DIP_RAD = np.radians(5.0)
DOZE_DIP_SINK_S = 1.0
DOZE_DIP_CATCH_S = 0.45
DOZE_DIPS_AT = (0.38, 0.62)  # the nodding-off dips start at these shares of the doze
WAKE_RELEASE_S = 1.0
WAKE_JOLT_M = 0.015
WAKE_JOLT_LOOK_UP_RAD = np.radians(6.0)
WAKE_STRETCH_M = 0.025
WAKE_STRETCH_LOOK_UP_RAD = np.radians(12.0)
WAKE_STRETCH_S = 0.9
WAKE_STRETCH_HOLD_S = 0.4
WAKE_LOOK_HOLD_S = 0.8

# proud: success, perk + look at the user (+ a small tilt and a clack or wiggle when empty-handed)
PROUD_TILT_RAD = np.radians(12.0)
PROUD_LOOK_AT_S = 0.1  # after t_on; the tilt follows once the look has arrived
PROUD_LIFT_M = 0.025  # the head rises this much as it looks at the user
PROUD_FLOURISH_S = 1.2  # time the tilt holds for the flourish (three chomps or a sway)
PROUD_BOB_M = 0.008
PROUD_BOBS = 2

# search: a puzzled look around the counter after a failed detection
SEARCH_SPOTS = ((0.0, 0.14), (0.08, -0.16), (-0.06, 0.03))  # (forward, left) m from the gaze's spot
SEARCH_LOOK_S = 0.6
SEARCH_DWELL_S = 0.35
SEARCH_TILT_RAD = np.radians(14.0)
SEARCH_MAX_TURN_RAD = np.radians(25.0)


@dataclass(frozen=True, eq=False)
class Targets:
    """World points behaviors aim at. The robot has no perception of its own: only the person's
    head comes from the scene; an object target left None means "where the camera is looking"
    (Context.aim_point: GAZE_AIM_M along the current gaze)."""

    person_head: np.ndarray
    container: Optional[np.ndarray] = None
    button: Optional[np.ndarray] = None

    @classmethod
    def from_scene(cls, scene):
        return cls(person_head=scene.person_head())


@dataclass(frozen=True, eq=False)
class Context:
    """What a behavior sees: the (warped) functional track it plays on, the kinematic Chain and the
    geometry Scene, the Targets, a seed for its noise and the cap preset it paces itself to
    (caps.CAP_PRESETS; compose the result with the same one)."""

    track: Track
    chain: object
    scene: object
    targets: Targets
    seed: int = 0
    caps: str = DEFAULT_CAPS
    limits: object = field(init=False, repr=False, default=None)  # caps.SampleCaps along the track

    def __post_init__(self):
        object.__setattr__(self, "limits", sample_caps(self.track.zone, self.caps))

    @classmethod
    def of(cls, track, chain, scene, targets=None, seed=0, caps=DEFAULT_CAPS):
        """A context on a Chain and a Scene (targets default to Targets.from_scene)."""
        return cls(track, chain, scene, Targets.from_scene(scene) if targets is None else targets, seed, caps)

    @property
    def person(self):
        return self.targets.person_head

    def aim_point(self, target, t):
        """`target` as a world point, or when it is None the point GAZE_AIM_M from the eye along the
        gaze at the posture at t (at the view pose the wrist camera is already looking at the object)."""
        if target is not None:
            return np.asarray(target, dtype=float)
        return gaze.gaze_point(self.chain, self.q(t), GAZE_AIM_M)

    def index(self, t):
        """The sample at time t (clipped to the track)."""
        return int(np.clip(round(t / DT), 0, self.track.n - 1))

    def q(self, t):
        return self.track.q[self.index(t)]

    def caps_at(self, t):
        """The expressive caps (caps.CapSet) at time t: its zone's tier in this context's preset."""
        i, lim = self.index(t), self.limits
        return CapSet(lim.joint_speed[i], lim.joint_accel[i], float(lim.head_speed[i]))

    def empty(self, name, labels=()):
        return Contribution.empty(name, self.track.n, labels)


# ----------------------------------------------------------------------------------- helpers
def _beat(name, t0, t1):
    return [Label(name, t0, t1)] if name else []


def turn_angle(offset):
    """How far a head offset turns the gaze (rad)."""
    return float(np.arccos(np.clip(np.cos(offset[4]) * np.cos(offset[5]), -1.0, 1.0)))


def _path_rates(ctx, t, offset):
    """(joint (7,), travel): the most each joint (rad) and a head point (m) move per unit of a
    move's progress toward head offset `offset` at the posture at t, sampled along the way (a big
    turn is far from linear) and a little past it (overshoot)."""
    q = ctx.q(t)
    offset = np.asarray(offset, dtype=float)
    dq = np.array([head_dq(ctx.chain, q, s * offset) for s in PATH_SAMPLES])
    step = np.diff(PATH_SAMPLES)
    joint = (np.abs(np.diff(dq, axis=0)) / step[:, None]).max(axis=0)
    travel = max(gaze.head_travel(ctx.chain, q + a, b - a) / ds for a, b, ds in zip(dq[:-1], dq[1:], step))
    return joint, travel


def _cap_shares(ctx, t, joint, travel):
    """(speed, accel, head): how much of the caps at time t one unit of rate uses (the worst joint)."""
    cap = ctx.caps_at(t)
    return (
        float(np.max(joint / np.asarray(cap.joint_speed))),
        float(np.max(joint / np.asarray(cap.joint_accel))),
        travel / cap.head_speed,
    )


def _paced(ctx, t, offset, design_s, slope, curve):
    """design_s, or longer if a move of `offset` (at the posture at t) whose shape peaks at
    slope * A / T (speed) and curve * A / T^2 (acceleration) would use more than PACE of a cap at t:
    joint speed and acceleration of the joint offsets that realise it, head speed at eye and tool."""
    speed, accel, head = _cap_shares(ctx, t, *_path_rates(ctx, t, offset))
    need = max(slope * speed / PACE, slope * head / PACE, float(np.sqrt(curve * accel / PACE)))
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
    speed, accel, head = _cap_shares(ctx, t, *_path_rates(ctx, t, offset))
    rate = float(np.abs(np.gradient(wave, DT)).max())
    curve = float(np.abs(np.gradient(np.gradient(wave, DT), DT)).max())
    used = (speed * rate, accel * curve, head * rate)
    return min([1.0] + [PACE / u for u in used if u > 0.0])


def fence(track, forbidden, fade_s):
    """(N,) 1 away from the forbidden samples, 0 on them, easing (raised cosine over fade_s) to 0
    before each forbidden sample and back up after it: multiplying a contribution by it keeps it
    off them."""
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
    return fence(track, track.zone == PRECISION, GAIN_RAMP_S)


def _add_gravity(ctx, c, pan=0.0, tilt=0.0, roll=0.0):
    """Add gravity turns (N,) (or scalars) to c.head's rotation, at each sample's functional
    posture: pan about world z, tilt down about the axis across the gaze, roll about the gaze."""
    n = ctx.track.n
    pan, tilt, roll = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in (pan, tilt, roll))
    for i in np.nonzero((pan != 0.0) | (tilt != 0.0) | (roll != 0.0))[0]:
        c.head[i, 3:] += gaze.gravity_turn(ctx.chain, ctx.track.q[i], pan[i], tilt[i], roll[i])


def _gravity_offset(ctx, t, shift=(0.0, 0.0, 0.0), pan=0.0, tilt=0.0, roll=0.0):
    """(6,) head offset of a world shift plus a gravity turn, at the posture at t."""
    return np.concatenate([shift, gaze.gravity_turn(ctx.chain, ctx.q(t), pan, tilt, roll)])


def safe_scale(ctx, t, offset):
    """Largest s in [0, 1] for which s * offset (a head offset) at the posture at t keeps every arm
    part SAFE_MARGIN_M above its clearance floors (compose.clearance_floor: to the user and to the
    obstacles): a look at the user or a droop near the counter stops short instead of being
    shrunk by compose()."""
    q, offset = ctx.q(t), np.asarray(offset, dtype=float)
    return _safe_fraction(ctx, t, lambda s: head_dq(ctx.chain, q, s * offset))


def safe_joint_scale(ctx, t, dq):
    """safe_scale for joint offsets dq (7,)."""
    return _safe_fraction(ctx, t, lambda s: s * np.asarray(dq, dtype=float))


def _safe_fraction(ctx, t, dq_at):
    """Largest s in [0, 1] whose joint offset dq_at(s) at the posture at t keeps every arm part
    SAFE_MARGIN_M above its clearance floors (bisection)."""
    i = ctx.index(t)
    q, held = ctx.track.q[i], bool(ctx.track.held[i])
    func = clearances_of(ctx.chain, ctx.scene, q, held)
    floor = np.minimum(clearance_floor(func) + SAFE_MARGIN_M, func)

    def clear(s):
        return bool(np.all(clearances_of(ctx.chain, ctx.scene, q + dq_at(s), held) >= floor))

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
    """Where the gaze (from the eye along ee z) is looking at the posture at t: the container if the
    targets have one, else the point GAZE_AIM_M along the gaze (there is no counter model)."""
    return ctx.aim_point(ctx.targets.container, t)


# --------------------------------------------------------------------------- alive at rest
def breathe(ctx, t0=0.0, t1=None, rise_m=BREATH_RISE_M, period_s=BREATH_PERIOD_S, avoid=None, beat=None):
    """Breathing over [t0, t1] (default: the whole track): the head rises and the chin lifts on
    the in-breath, never two breaths alike. Fades in and out over BREATH_FADE_S, and out before
    PRECISION and wherever `avoid` (N,) is set (moving, a stare). While the gripper holds
    something the chin stays level (the container must not tip): only the rise remains."""
    track = ctx.track
    t1 = (track.n - 1) * DT if t1 is None else t1
    t = track.t
    forbidden = (track.zone == PRECISION) | (False if avoid is None else avoid)
    env = curves.envelope(t, t0, t1 - BREATH_FADE_S, BREATH_FADE_S, BREATH_FADE_S)
    env = env * fence(track, forbidden, PRECISION_FADE_S)
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
    t1 = (track.n - 1) * DT if t1 is None else t1
    forbidden = (track.zone == PRECISION) | (False if avoid is None else avoid)
    env = curves.envelope(track.t, t0, t1 - MICRO_FADE_S, MICRO_FADE_S, MICRO_FADE_S)
    env = env * fence(track, forbidden, PRECISION_FADE_S)
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
    return fence(track, track.held, GATE_FADE_S)


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
    lift_m=0.0,
    name="gaze",
):
    """Look at world point `target` from t_on until t_off, then turn back to the functional gaze.

    A saccade-like eased turn with a GAZE_OVERSHOOT overshoot, paced to the caps, the head rising
    lift_m as it turns (one move: a proud or eager look); the lean toward the target follows
    GAZE_LEAN_LAG_S later (overlapping action). Look-at by IK at each
    functional posture (the eyes stay on target while the arm moves); `mode` is gaze.look_at_offset's
    ('pan' keeps a held container level). The turn stops short where it would bring an arm part
    near a clearance floor (plan_gaze). It holds at least until it has arrived (t_off is pushed
    back to t_on + the paced turn), so a turn back never stacks on the turn out. Two gazes where
    one's t_off is the other's t_on (same turn_s) blend straight from one target to the next. The
    beat ends halfway through the turn back."""
    target = ctx.aim_point(target, t_on)
    lean_m, max_turn_rad, turn_s, lift_m = plan_gaze(ctx, target, t_on, turn_s, lean_m, max_turn_rad, mode, lift_m)
    t_off = max(t_off, t_on + turn_s)
    t = ctx.track.t
    turn = curves.overshoot_envelope(t, t_on, t_off, turn_s, GAZE_OVERSHOOT)
    lean = curves.envelope(t, t_on + GAZE_LEAN_LAG_S, t_off + GAZE_LEAN_LAG_S, turn_s, turn_s)
    c = ctx.empty(name, _beat(beat, t_on, t_off + turn_s / 2))
    for i in np.nonzero((turn != 0.0) | (lean != 0.0))[0]:
        look = gaze.look_at_offset(ctx.chain, ctx.track.q[i], target, lean_m, max_turn_rad, mode)
        c.head[i, :3] = look[:3] * lean[i] + UP * lift_m * turn[i]
        c.head[i, 3:] = look[3:] * turn[i]
    return c


def plan_gaze(
    ctx,
    target,
    t_on,
    turn_s=GAZE_TURN_S,
    lean_m=GAZE_LEAN_M,
    max_turn_rad=GAZE_MAX_TURN_RAD,
    mode="shortest",
    lift_m=0.0,
):
    """(lean_m, max_turn_rad, turn_s, lift_m) a gaze at `target` from t_on will use: shortened where
    it would come near a clearance floor (safe_scale, at the posture at t_on) and paced to the caps."""
    lift = UP * lift_m
    first = gaze.look_at_offset(ctx.chain, ctx.q(t_on), target, lean_m, max_turn_rad, mode)
    first[:3] += lift
    safe = safe_scale(ctx, t_on, first * (1.0 + GAZE_OVERSHOOT))
    if safe < 1.0:
        max_turn_rad = turn_angle(first) * safe
        lean_m, lift_m = lean_m * safe, lift_m * safe
        first = gaze.look_at_offset(ctx.chain, ctx.q(t_on), target, lean_m, max_turn_rad, mode)
        first[:3] += UP * lift_m
    return lean_m, max_turn_rad, paced_move(ctx, t_on, first, turn_s, GAZE_OVERSHOOT), lift_m


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
    a, b = ctx.aim_point(a, t_on), ctx.aim_point(b, t_on)
    look_a = gaze.look_at_offset(ctx.chain, q, a, 0.0, max_turn_rad, "pan_tilt")
    look_b = gaze.look_at_offset(ctx.chain, q, b, 0.0, max_turn_rad, "pan_tilt")
    look_a = look_a * safe_scale(ctx, t_on, look_a)  # every pose stops short of the floors (the capsule
    look_b = look_b * safe_scale(ctx, t_on, look_b)  # model reads a little less room than the spike's mesh)
    d = a - ctx.chain.fk(q, "eye")[0]
    snap = look_a + np.concatenate([lean_m * d / np.linalg.norm(d), np.zeros(3)])
    snap = snap * safe_scale(ctx, t_on, snap)
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
    counter around `around` (default: gaze_spot, where the gaze is looking), each with a tilt to alternating
    sides (each kept clear: safe_scale), then back to the functional gaze."""
    q = ctx.q(t_on)
    around = gaze_spot(ctx, t_on) if around is None else np.asarray(around, dtype=float)
    _, R = ctx.chain.fk(q, "ee")
    forward = np.array([R[0, 2], R[1, 2], 0.0])
    forward = forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 1e-6 else np.array([1.0, 0.0, 0.0])
    left = np.cross([0.0, 0.0, 1.0], forward)
    keys, t_k = [(t_on, np.zeros(6))], t_on
    for k, (fwd, lft) in enumerate(spots):
        look = gaze.look_at_offset(ctx.chain, q, around + fwd * forward + lft * left, 0.0, SEARCH_MAX_TURN_RAD)
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
    d = ctx.aim_point(target, t_on) - ctx.chain.fk(q, "tool")[0]
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
def paced_perk_s(ctx, t, level=False, lift_m=PERK_LIFT_M, look_up_rad=PERK_LOOK_UP_RAD):
    """How long a perk's rise at the posture at t takes (level: the lift only)."""
    peak = _gravity_offset(ctx, t, (0.0, 0.0, lift_m), tilt=0.0 if level else -look_up_rad)
    return paced_move(ctx, t, peak, PERK_RISE_S, PERK_OVERSHOOT)


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
    away = ctx.chain.fk(ctx.q(t), "eye")[0] - ctx.person
    away[2] = 0.0
    away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.zeros(3)
    d = np.array([0.0, 0.0, -1.0]) + SAG_AWAY_SHARE * away
    return sag_m * d / np.linalg.norm(d)


def droop(ctx, t_on, t_off, sag_m=DROOP_SAG_M, look_down_rad=DROOP_LOOK_DOWN_RAD, quick=1.0, beat=None):
    """Sad or sheepish: a slow sag (sag_shift) and a look down from t_on, held until t_off, then a
    slow recovery (both quick times their design length at least). It sags only as far as the
    clearance floors allow (safe_scale)."""
    shift = sag_shift(ctx, t_on, sag_m)
    peak = _gravity_offset(ctx, t_on, shift, tilt=look_down_rad)
    down_s = paced_move(ctx, t_on, peak, DROOP_S * quick)
    up_s = paced_move(ctx, t_on, peak, DROOP_RECOVER_S * quick)
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
    pan, tilt, _ = gaze.aim(ctx.chain, ctx.q(t_on), target)
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
    away = ctx.chain.fk(ctx.q(t_on), "eye")[0] - source
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


def _amplitudes(ctx, t, wave):
    """(7,) the largest amplitude each joint may give the unit waveform `wave` within PACE of the
    caps at t (its speed and acceleration)."""
    cap = ctx.caps_at(t)
    rate = float(np.abs(np.gradient(wave, DT)).max())
    curve = float(np.abs(np.gradient(np.gradient(wave, DT), DT)).max())
    return PACE * np.minimum(np.asarray(cap.joint_accel) / curve, np.asarray(cap.joint_speed) / rate)


def _wag(ctx, t_on, env, phase, max_sway_m, max_turn_rad):
    """(joint offsets (N, 7), note) of a wag: the carrier sin(phase) under the envelope env (both
    (N,)). Every joint goes as far as its own caps allow, signed so all of them push the eye the same
    way across the gaze, each in proportion to how much of its motion is sideways. The wrist (joints
    5-7: a pan and a roll about the neck, at most max_turn_rad each) swings both ways. The arm (joints
    1-4: the sway, at most max_sway_m of eye travel) swings out to the side the clearance floors
    leave more room on and back, as far peak to peak as a swing both ways: beside the user it only
    swings away from them. The head speed is kept within PACE of its cap."""
    q = ctx.q(t_on)
    wave_both = env * np.sin(phase)
    _, R = ctx.chain.fk(q, "ee")
    J = ctx.chain.jacobian(q, "eye")[:3]
    lateral = across_gaze(R) @ J
    sideways = lateral / np.maximum(np.linalg.norm(J, axis=0), 1e-9)  # signed share of each joint's motion
    wrist = np.zeros(7)
    wrist[WRIST_JOINTS] = np.clip(
        (sideways * _amplitudes(ctx, t_on, wave_both))[WRIST_JOINTS], -max_turn_rad, max_turn_rad
    )
    wrist *= min(safe_joint_scale(ctx, t_on, wrist), safe_joint_scale(ctx, t_on, -wrist))
    wave_out = env * (1.0 + np.sin(phase)) / 2.0  # 0..env: out and back (the same shape the other way)
    arm = sideways * _amplitudes(ctx, t_on, wave_out)
    arm[WRIST_JOINTS] = 0.0
    room = (safe_joint_scale(ctx, t_on, arm), safe_joint_scale(ctx, t_on, -arm))
    side = 1.0 if room[0] >= room[1] else -1.0
    arm = side * arm * max(room)
    sway = float(np.linalg.norm(J @ arm))
    arm *= min(1.0, max_sway_m / sway) if sway > 0.0 else 1.0
    head = _head_speed_share(ctx, t_on, wrist, arm, wave_both, wave_out)
    wrist, arm = wrist * head, arm * head
    out = env * (1.0 + side * np.sin(phase)) / 2.0
    joints = wrist * wave_both[:, None] + arm * out[:, None]
    note = (
        f"sway {np.linalg.norm(J @ arm) * 1000.0:.0f} mm out {'and back' if side > 0 else 'the other way and back'}, "
        f"wrist {np.degrees(np.abs(wrist).max()):.1f} deg (caps '{ctx.caps}')"
    )
    return joints, note


def _head_speed_share(ctx, t, wrist, arm, wave_both, wave_out):
    """The scale (at most 1) that keeps a wag's eye and tool points within PACE of the head speed
    cap at t (first order)."""
    q = ctx.q(t)
    rate_both = float(np.abs(np.gradient(wave_both, DT)).max())
    rate_out = float(np.abs(np.gradient(wave_out, DT)).max())
    worst = 0.0
    for site in HEAD_SPEED_SITES:
        J = ctx.chain.jacobian(q, site)[:3]
        worst = max(worst, np.linalg.norm(J @ wrist) * rate_both + np.linalg.norm(J @ arm) * rate_out)
    cap = PACE * ctx.caps_at(t).head_speed
    return min(1.0, cap / worst) if worst > 0.0 else 1.0


def _no_play(track):
    """Where wiggle and clack must not act: holding something, NEAR_USER or PRECISION."""
    return track.held | (track.zone == NEAR_USER) | (track.zone == PRECISION)


def _wag_envelope(t, t_on, cycles, freq_hz, decay_s=np.inf):
    """(env, phase): a windowed sine's envelope, peak 1 (curves.sine_envelope), and its phase."""
    env = curves.sine_envelope(t, t_on, cycles, freq_hz, decay_s)
    return env / env.max(), 2.0 * np.pi * freq_hz * (np.asarray(t, dtype=float) - t_on)


def wiggle(ctx, t_on, cycles=WIGGLE_CYCLES, freq_hz=WIGGLE_HZ, beat=None):
    """A happy sway: the head swings side to side `cycles` times at freq_hz, the arm swaying it
    across the gaze while the wrist pans and rolls it toward the same side, as big as the caps and
    the clearance floors allow (_wag; the size it reached is noted). Never while the gripper holds
    something or near the user: zero there, fading out before."""
    env, phase = _wag_envelope(ctx.track.t, t_on, cycles, freq_hz)
    joints, note = _wag(ctx, t_on, env, phase, WIGGLE_MAX_SWAY_M, WIGGLE_MAX_TURN_RAD)
    gate = fence(ctx.track, _no_play(ctx.track), GATE_FADE_S)
    c = ctx.empty("wiggle", _beat(beat, t_on, t_on + cycles / freq_hz))
    c.joints[:] = joints * gate[:, None]
    c.notes = [note]
    return c


def shake_off(ctx, t_on, cycles=SHAKE_CYCLES, freq_hz=SHAKE_HZ, beat=None):
    """A dog shaking off water after a mishap: the wiggle's sway and wrist wag as a quicker shiver
    dying away, as big as the caps allow (the size it reached is noted)."""
    env, phase = _wag_envelope(ctx.track.t, t_on, cycles, freq_hz, SHAKE_DECAY_S)
    joints, note = _wag(ctx, t_on, env, phase, SHAKE_MAX_SWAY_M, SHAKE_MAX_TURN_RAD)
    c = ctx.empty("shake_off", _beat(beat, t_on, t_on + cycles / freq_hz))
    c.joints[:] = joints
    c.notes = [note]
    return c


def clack(ctx, t_on, chomps=CLACK_CHOMPS, size=CLACK_SIZE, beat=None):
    """The gripper as a mouth: quick partial chomps (opening when it is mostly closed, closing when
    mostly open) within the 2F-85's speed. Only when it holds nothing, in FREE and not near the
    user: zero elsewhere, fading out before."""
    chomp_s = max(CLACK_CHOMP_S, curves.BUMP_PEAK_SLOPE * size / EXPR_GRIP_SPEED)
    sign = -1.0 if ctx.track.grip[ctx.index(t_on)] > 0.5 else 1.0
    gate = fence(ctx.track, _no_play(ctx.track) | (ctx.track.zone != FREE), GATE_FADE_S)
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
    poses = [ctx.chain.fk(q, "eye") for q in ctx.track.q[idx]]
    p, R = np.array([x[0] for x in poses]), np.array([x[1] for x in poses])
    moved = np.linalg.norm(p - p[0], axis=1)
    turned = np.array([gaze.rotation_angle(R[0].T @ Rn) for Rn in R])
    far = np.nonzero((moved >= PROBE_M) | (turned >= PROBE_RAD))[0]
    return idx[far[0] if far.size else -1] * DT


def _coming(ctx, t0, t1):
    """(world shift unit (3,), head-frame turn unit (3,) as [roll, pitch, yaw]) of the head's
    motion on the functional track from t0 to t1 (zero vectors where it does not move)."""
    p0, R0 = ctx.chain.fk(ctx.q(t0), "eye")
    p1, R1 = ctx.chain.fk(ctx.q(t1), "eye")
    shift = p1 - p0
    rv = gaze.rotation_vector(R0.T @ R1)  # head frame (R0): x pitch, y yaw, z roll
    turn = np.array([rv[2], rv[0], rv[1]])
    unit_shift = shift / np.linalg.norm(shift) if np.linalg.norm(shift) > MIN_PROBE_M else np.zeros(3)
    unit_turn = turn / np.linalg.norm(turn) if np.linalg.norm(turn) > MIN_PROBE_RAD else np.zeros(3)
    return unit_shift, unit_turn


def wind_up_s(
    ctx, t, t_release, shift_m=WIND_UP_SHIFT_M, turn_rad=WIND_UP_TURN_RAD, draw_s=WIND_UP_S, release_s=WIND_UP_RELEASE_S
):
    """How long a wind-up at the posture at t, before the move that starts at t_release, takes."""
    back = _draw_back(ctx, t, t_release, shift_m, turn_rad)
    return paced_move(ctx, t, back, draw_s) + WIND_UP_HOLD_S + paced_move(ctx, t, back, release_s)


def _draw_back(ctx, t, t_release, shift_m, turn_rad):
    """(6,) the wind-up's offset at the posture at t: against the direction of the move that starts at
    t_release (its first PROBE_M or PROBE_RAD on the functional track: the call's target), each of
    the counter-turn (wrist) and the draw-back shift (shoulder and elbow) as far as the clearance
    floors allow."""
    unit_shift, unit_turn = _coming(ctx, t_release, _probe(ctx, t_release, 1))
    turn = np.concatenate([np.zeros(3), -turn_rad * unit_turn])
    shift = np.concatenate([-shift_m * unit_shift, np.zeros(3)])
    back = safe_scale(ctx, t, turn) * turn + safe_scale(ctx, t, shift) * shift
    return back * safe_scale(ctx, t, back)


def wind_up(
    ctx,
    t_on,
    t_release,
    shift_m=WIND_UP_SHIFT_M,
    turn_rad=WIND_UP_TURN_RAD,
    draw_s=WIND_UP_S,
    release_s=WIND_UP_RELEASE_S,
    beat=None,
):
    """Anticipation (Takayama's forethought), self-contained before a move: the head draws back
    against the motion that starts at t_release, holds a moment and comes back, at rest on the
    functional pose by t_release, so the move then runs as recorded. It starts at t_on, or later
    when the window is longer than it needs (wind_up_s)."""
    back = _draw_back(ctx, t_on, t_release, shift_m, turn_rad)
    draw_s = paced_move(ctx, t_on, back, draw_s)
    release_s = paced_move(ctx, t_on, back, release_s)
    room = t_release - t_on - WIND_UP_HOLD_S
    if room < draw_s + release_s:  # a short window: smaller and quicker (its accel goes as A / T^2)
        f = max(0.0, room / (draw_s + release_s))
        back, draw_s, release_s = back * f**2, draw_s * f, release_s * f
    t_back = t_release - release_s
    t_draw = max(t_on, t_back - WIND_UP_HOLD_S - draw_s)
    env = curves.envelope(ctx.track.t, t_draw, t_back, draw_s, release_s)
    c = ctx.empty("wind_up", _beat(beat, t_draw, t_release))
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
    R_now = [ctx.chain.fk(q, "ee")[1] for q in track.q[idx]]
    R_ahead = [ctx.chain.fk(q, "ee")[1] for q in track.q[ahead]]
    yaw_pitch = np.array([gaze.euler_yxz(a.T @ b)[:2] for a, b in zip(R_now, R_ahead)])
    yaw_pitch = gaze.gaussian_smooth(yaw_pitch, LEAD_SMOOTH_S / DT)
    c.head[idx, 5] = yaw_pitch[:, 0] * env[idx]
    c.head[idx, 4] = yaw_pitch[:, 1] * env[idx]
    return c


def follow_through(ctx, t_still, turn_rad=FOLLOW_TURN_RAD, shift_m=FOLLOW_SHIFT_M, settle_s=FOLLOW_SETTLE_S, beat=None):
    """After a move, once the arm is still at t_still: the head carries on a little in the
    direction it was going (over its last PROBE_M or PROBE_RAD), then settles back onto the
    functional pose with a damped wobble, at rest by t_still + settle_s."""
    unit_shift, unit_turn = _coming(ctx, _probe(ctx, t_still, -1), t_still)
    carry = np.concatenate([shift_m * unit_shift, turn_rad * unit_turn])
    wave = curves.wobble(ctx.track.t, t_still, FOLLOW_HZ, FOLLOW_DAMPING, settle_s)
    scale = fit_scale(ctx, t_still, carry, wave) if np.any(carry) else 1.0
    c = ctx.empty("follow_through", _beat(beat, t_still, t_still + settle_s))
    c.head[:] = scale * carry * wave[:, None]
    c.notes = _limited(scale, f"{np.degrees(scale * turn_rad):.1f} deg")
    return c


def _lean_toward(ctx, target, t, lean_m):
    """(6,) a head shift of lean_m from the tool point at the posture at t toward `target`."""
    d = ctx.aim_point(target, t) - ctx.chain.fk(ctx.q(t), "tool")[0]
    return np.concatenate([lean_m * d / np.linalg.norm(d), np.zeros(3)])


def lock_on_s(ctx, target, t, stare_s, lean_m=LOCK_LEAN_M):
    """How long a lock-on at the posture at t takes with stare_s of stillness once leaned in: the
    lean in, the stare, the lean back and LOCK_STILL_S."""
    return 2.0 * paced_move(ctx, t, _lean_toward(ctx, target, t, lean_m), LOCK_LEAN_S) + stare_s + LOCK_STILL_S


def lock_on(ctx, target, t_on, t_off, lean_m=LOCK_LEAN_M, beat=None):
    """The focus stare at a stop before PRECISION (the arm still from t_on until the next motion at
    t_off): a lean of lean_m toward `target`, held, and eased back LOCK_STILL_S before t_off, so
    the head is on the functional pose when the motion resumes; then stillness. (At the standoff
    the director slows the precise motion that follows; elsewhere nothing changes inside the zone.)"""
    lean = _lean_toward(ctx, target, t_on, lean_m)
    lean_s = paced_move(ctx, t_on, lean, LOCK_LEAN_S)
    t_back = t_off - LOCK_STILL_S - lean_s
    c = ctx.empty("lock_on", _beat(beat, t_on, t_off))
    if t_back > t_on + lean_s:
        c.head[:] = lean * curves.envelope(ctx.track.t, t_on, t_back, lean_s, lean_s)[:, None]
    return c


# -------------------------------------------------------------------------------- compounds
def proud(ctx, t_on, t_off, look_until=None, flourish="clack", max_turn_rad=GAZE_MAX_TURN_RAD, beat=None):
    """Success over [t_on, t_off]: a proud lift as it looks at the user (one move); holding nothing,
    a small tilt once the look has arrived, with a flourish ('clack' or 'wiggle') while it is held,
    if both fit before t_off (else noted). Holding something it is muted, so the container stays
    level: a pan-only look and a small bob; no tilt, wiggle or clack. flourish None: just the look.
    The look holds until look_until (default t_off), e.g. to run on into an attentive look."""
    holding = bool(ctx.track.held[ctx.index(t_on)])
    t_look = t_on + PROUD_LOOK_AT_S
    until = t_off if look_until is None else look_until
    look = look_at_user(ctx, t_look, until, holding, max_turn_rad, lift_m=PROUD_LIFT_M)
    parts, notes = [look], []
    t_arrived = t_look + user_turn_s(ctx, t_look, holding, max_turn_rad, lift_m=PROUD_LIFT_M)
    if not flourish:
        pass
    elif holding:
        parts.append(bounce(ctx, t_arrived, hops=PROUD_BOBS, height_m=PROUD_BOB_M))
    else:
        t_tilt = t_arrived
        tilt_s = paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, PROUD_TILT_RAD, 0.0, 0.0], TILT_TURN_S, TILT_OVERSHOOT)
        t_back = t_off - tilt_s  # the tilt is back before the look turns back
        if t_back >= t_tilt + tilt_s + PROUD_FLOURISH_S:
            parts.append(head_tilt(ctx, t_tilt, t_back, angle_rad=PROUD_TILT_RAD))
            parts.append({"clack": clack, "wiggle": wiggle}[flourish](ctx, t_tilt + tilt_s))
        else:
            notes.append("no time for the tilt and the flourish before the look turns back")
    c = Contribution.combine("proud", parts, labels=_beat(beat, t_on, t_off))
    c.notes += notes
    return c


def look_at_user(ctx, t_on, t_off, level=False, max_turn_rad=GAZE_MAX_TURN_RAD, lift_m=0.0, beat=None):
    """Look at the user (a gaze without a lean, a gravity pan and tilt; rising lift_m as it turns),
    as far as their clearance floor allows; level (holding a container): pan only, at most
    LEVEL_LOOK_MAX_RAD. Two of these where one's t_off is the other's t_on (same arguments) run on as
    one look."""
    mode = "pan" if level else "pan_tilt"
    max_turn_rad = min(max_turn_rad, LEVEL_LOOK_MAX_RAD) if level else max_turn_rad
    return gaze_to(
        ctx,
        ctx.person,
        t_on,
        t_off,
        beat,
        lean_m=0.0,
        max_turn_rad=max_turn_rad,
        mode=mode,
        lift_m=lift_m,
        name="look at the user",
    )


def user_turn_s(ctx, t_on, level=False, max_turn_rad=GAZE_MAX_TURN_RAD, lift_m=0.0):
    """How long look_at_user from t_on takes to turn (paced at the posture at t_on)."""
    max_turn_rad = min(max_turn_rad, LEVEL_LOOK_MAX_RAD) if level else max_turn_rad
    mode = "pan" if level else "pan_tilt"
    return plan_gaze(ctx, ctx.person, t_on, lean_m=0.0, max_turn_rad=max_turn_rad, mode=mode, lift_m=lift_m)[2]


def doze_and_wake(ctx, t_on, doze_s, beat_doze=None, beat_wake=None):
    """Drift into a curled low rest (a sag, a look down, a slight roll) with slow deep breathing
    and nodding-off dips for doze_s; then a startled wake (a quick jolt up, then the rest lets go), a
    slow stretch the jolt hands over to, and a look at the user. Returns one contribution with both beats."""
    t = ctx.track.t
    t_wake = t_on + doze_s
    sag = sag_shift(ctx, t_on, DOZE_SAG_M)
    curl_peak = _gravity_offset(ctx, t_on, sag, tilt=DOZE_LOOK_DOWN_RAD, roll=DOZE_ROLL_RAD)
    deepest = curl_peak + _gravity_offset(ctx, t_on, (0.0, 0.0, -DOZE_BREATH_M), tilt=DOZE_DIP_RAD)
    settle_s = paced_move(ctx, t_on, curl_peak, DOZE_SETTLE_S)
    release_s = paced_move(ctx, t_on, curl_peak, WAKE_RELEASE_S)
    t_release = t_wake + paced_perk_s(ctx, t_wake, False, WAKE_JOLT_M, WAKE_JOLT_LOOK_UP_RAD)  # once the jolt is up
    curl = curves.envelope(t, t_on, t_release, settle_s, release_s)
    catch_s = paced_move(ctx, t_on, _gravity_offset(ctx, t_on, tilt=DOZE_DIP_RAD), DOZE_DIP_CATCH_S)
    dips = sum(_sink_and_catch(t, t_on + f * doze_s, DOZE_DIP_SINK_S, catch_s) for f in DOZE_DIPS_AT)
    breath = curves.oscillate(t - t_on, DOZE_BREATH_PERIOD_S) * curl
    rest = ctx.empty("doze")
    rest.head[:, :3] = sag * curl[:, None]
    rest.head[:, 2] += DOZE_BREATH_M * breath
    _add_gravity(ctx, rest, tilt=DOZE_LOOK_DOWN_RAD * curl + DOZE_DIP_RAD * dips, roll=DOZE_ROLL_RAD * curl)
    rest.scale = safe_scale(ctx, t_on, deepest)
    t_stretch = t_release + release_s
    jolt = perk(ctx, t_wake, t_stretch, WAKE_JOLT_M, WAKE_JOLT_LOOK_UP_RAD)  # held, then handed over to the stretch
    stretch_peak = _gravity_offset(ctx, t_stretch, (0.0, 0.0, WAKE_STRETCH_M), tilt=-WAKE_STRETCH_LOOK_UP_RAD)
    stretch_s = paced_move(ctx, t_stretch, stretch_peak, WAKE_STRETCH_S)
    t_relax = t_stretch + stretch_s + WAKE_STRETCH_HOLD_S
    reach = curves.ramp(t, t_stretch, t_stretch + stretch_s) - curves.ramp(t, t_relax, t_relax + stretch_s)
    stretch = ctx.empty("stretch")
    stretch.head[:, 2] = WAKE_STRETCH_M * reach
    _add_gravity(ctx, stretch, tilt=-WAKE_STRETCH_LOOK_UP_RAD * reach)
    t_look = t_relax + stretch_s / 2
    look = look_at_user(ctx, t_look, t_look + WAKE_LOOK_HOLD_S + GAZE_TURN_S, beat="look")
    beats = _beat(beat_doze, t_on, t_wake) + _beat(beat_wake, t_wake, look.labels[0].t1)
    return Contribution.combine("doze_and_wake", [rest, jolt, stretch, look], labels=beats)


def _sink_and_catch(t, t0, sink_s, catch_s):
    """0 -> 1 slowly over sink_s from t0, then back to 0 quickly over catch_s: nodding off."""
    return curves.ramp(t, t0, t0 + sink_s) - curves.ramp(t, t0 + sink_s, t0 + sink_s + catch_s)
