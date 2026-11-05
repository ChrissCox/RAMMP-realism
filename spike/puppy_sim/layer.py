"""Composition: q = q_func(tau(t)) + gain(zone) * dq_expressive(t).

The functional track comes in already time-warped (timeline.warp). Behaviors give Contributions:
head offsets (turned into joint offsets by DLS at the functional posture), joint offsets (expressive
base motions) and gripper offsets. compose() sums them, applies the zone gain and the near-user
projection, fits the caps, and returns the composed Track (with .func, .gain and .log). The
guarantees, each checked by checks.py:

- PRECISION: the gain is exactly 0 and q is copied from the functional track (bit-identical).
- Gains ramp (raised cosine over GAIN_RAMP_S) and reach a zone's level before its first sample.
- NEAR_USER: gain 0.35, and no offset component that would bring an arm part closer to the person.
- Caps: the expressive part stays under EXPR_* (the gripper offset under the 2F-85's speed); the
  composite under the URDF velocity limits, the joint ranges and RAMMP's guards; and every arm
  part (the fingers as composed) above its clearance floor (person: min of
  its functional clearance and 0.35 m, less 5 mm; obstacles: min of its functional clearance and
  3 cm). A behavior that would break one scales its own amplitude down (logged, the same factor
  for the whole behavior, so the motion stays smooth); the functional motion is never changed.
"""

from dataclasses import dataclass, field

import numpy as np

from . import curves
from .kin import ARM_PARTS, GUARD_LIMITS, VEL_LIMITS
from .runs import NEAR_USER_M
from .timeline import DT, FREE, NEAR_USER, PRECISION, Track

GAINS = {FREE: 1.0, NEAR_USER: 0.35, PRECISION: 0.0}
GRIP_GAINS = {FREE: 1.0, NEAR_USER: 0.0, PRECISION: 0.0}  # and 0 while the gripper holds something
GAIN_RAMP_S = 0.3
GAIN_GUARD_SAMPLES = 1  # a ramp down ends this many samples before the lower zone (never inside it)
EXPR_JOINT_SPEED = 0.85  # rad/s per joint, the expressive part alone (RAMMP arm.JOINT_SPEED)
EXPR_HEAD_SPEED = 0.16  # m/s at the head, the expressive part alone (RAMMP arm.CART_SPEED)
EXPR_JOINT_ACCEL = 3.0  # rad/s^2 per joint, the expressive part alone (assumed)
EXPR_GRIP_SPEED = 0.150 / 0.085  # strokes/s, the gripper offset alone: the 2F-85's 150 mm/s over 85 mm
HEAD_SPEED_SITES = ("eye", "tool")  # the head's speed is the faster of these points
PERSON_SLACK_M = 0.005  # a part may come this much closer to the person than min(functional, 0.35)
OBSTACLE_KEEP_M = 0.03  # near obstacles a part keeps min(functional clearance, this)
OBSTACLE_TOL_M = 0.0005  # numerical tolerance on that rule
PROJECTION_SWEEPS = 2  # passes over the parts' gradients when removing toward-the-user motion
CAP_MARGIN = 0.95  # a behavior over a cap shrinks to this fraction of what would just fit
CAP_WINDOW_S = 0.3  # a behavior active within this of a violation shares the blame
BLAME_SHARE = 0.25  # ... if it is at least this big a share of the biggest one there
MAX_BLAME_VISITS = 40  # violations examined (worst first) per fitting round
HEAD_LEVER_M = 0.1  # head shifts count as turns of shift / HEAD_LEVER_M when sharing blame
FIT_ROUNDS = 8


@dataclass
class Contribution:
    """One behavior's share of the expressive layer on the composed track's clock (N samples):
    head (N, 6) head offsets (kin's convention), joints (N, 7) joint offsets, grip (N,) gripper
    offsets, the beats it shows, the amplitude scale compose() settled on, and notes (e.g. an
    amplitude the behavior limited to the caps itself) that compose() copies into the track log."""

    name: str
    head: np.ndarray
    joints: np.ndarray
    grip: np.ndarray
    beats: list = field(default_factory=list)
    scale: float = 1.0
    notes: list = field(default_factory=list)

    @classmethod
    def empty(cls, name, n, beats=()):
        return cls(name, np.zeros((n, 6)), np.zeros((n, 7)), np.zeros(n), list(beats))

    @classmethod
    def combine(cls, name, parts, beats=None):
        """One contribution that is the sum of `parts` (each at its scale): a compound behavior
        that compose() scales as a whole. Beats and notes are the parts' unless beats is given."""
        n = len(parts[0].grip)
        out = cls.empty(name, n, [b for p in parts for b in p.beats] if beats is None else beats)
        for p in parts:
            out.head += p.scale * p.head
            out.joints += p.scale * p.joints
            out.grip += p.scale * p.grip
            out.notes += p.notes
        return out

    def active(self):
        """(N,) where it does anything."""
        return np.any(self.head != 0, axis=1) | np.any(self.joints != 0, axis=1) | (self.grip != 0)

    def span(self):
        """(first, last) time it does anything, or None."""
        idx = np.nonzero(self.active())[0]
        return (idx[0] * DT, idx[-1] * DT) if idx.size else None


@dataclass(frozen=True)
class Violation:
    """A cap exceeded at a sample: `ratio` > 1 is how far over (1 / the scale that would fit).
    Clearance violations also name the target row (0 person, 1 obstacles) and the arm part."""

    kind: str
    index: int
    ratio: float
    detail: str
    row: int = -1
    part: int = -1


def ramped_floor(level, guard=0, ramp_s=GAIN_RAMP_S):
    """A smooth curve that never exceeds `level` (values in [0, 1]): every sample caps its
    neighbours at level + (1 - level) * raised_cosine(distance), the ramp starting `guard` samples
    away and taking ramp_s."""
    level = np.asarray(level, dtype=float)
    n_ramp = max(1, int(round(ramp_s / DT)))
    reach = guard + n_ramp
    padded = np.pad(level, reach, mode="edge")
    out = level.copy()
    for k in range(1, reach + 1):
        w = curves.raised_cosine((k - guard) / n_ramp) if k > guard else 0.0
        for shifted in (padded[reach - k : reach - k + len(level)], padded[reach + k : reach + k + len(level)]):
            out = np.minimum(out, shifted + (1.0 - shifted) * w)
    return out


def zone_gain(zone, gains=GAINS):
    """Per-sample gain from the zones: each sample's level, with raised-cosine ramps of
    GAIN_RAMP_S that finish GAIN_GUARD_SAMPLES before a lower zone starts and begin as long after
    it ends, so every sample of a zone sees at most its own level."""
    return ramped_floor([gains[int(z)] for z in zone], guard=GAIN_GUARD_SAMPLES)


def near_user_weight(zone):
    """1 on NEAR_USER samples, easing to 0 around them like a gain ramp."""
    return 1.0 - ramped_floor((np.asarray(zone) != NEAR_USER).astype(float), guard=GAIN_GUARD_SAMPLES)


def compose(track, contributions, kin):
    """The Puppy Mode track: the warped functional `track` plus `contributions` (each scaled in
    place when it would exceed a cap or a clearance floor)."""
    gain = zone_gain(track.zone)
    grip_gain = zone_gain(np.where(track.held, PRECISION, track.zone), GRIP_GAINS)
    near = near_user_weight(track.zone)
    func_clear = kin.clearance_track(track)
    log = [f"{c.name}: {note}" for c in contributions for note in c.notes]
    for _ in range(FIT_ROUNDS):
        head, joints, dgrip = _summed(contributions, track.n)
        expr = gain[:, None] * (joints + _head_to_joints(kin, track.q, head, gain))
        expr = away_from_user(kin, track, expr, near)
        q = np.where((gain == 0.0)[:, None], track.q, track.q + expr)
        grip = np.where(grip_gain == 0.0, track.grip, np.clip(track.grip + grip_gain * dgrip, 0.0, 1.0))
        violations = cap_violations(kin, track, q, func_clear, grip)
        if not violations or not _shrink(kin, track, (gain, grip_gain), contributions, violations, log):
            break
    if violations:
        log.append(f"still over a cap after fitting: {len(violations)} samples, worst {violations[0].kind}")
    beats = sorted((b for c in contributions for b in c.beats), key=lambda b: b.t0)
    return Track(
        q=q,
        grip=grip,
        zone=track.zone.copy(),
        held=track.held.copy(),
        tau=track.tau.copy(),
        phases=list(track.phases),
        beats=beats,
        func=track,
        gain=gain,
        log=log,
        name=track.name,
    )


def _summed(contributions, n):
    head, joints, grip = np.zeros((n, 6)), np.zeros((n, 7)), np.zeros(n)
    for c in contributions:
        head += c.scale * c.head
        joints += c.scale * c.joints
        grip += c.scale * c.grip
    return head, joints, grip


def _head_to_joints(kin, q_func, head, gain):
    """Joint offsets realising the head offsets at each functional posture (where gain > 0)."""
    dq = np.zeros((len(q_func), 7))
    for i in np.nonzero((gain > 0.0) & np.any(head != 0.0, axis=1))[0]:
        dq[i] = kin.head_dq(q_func[i], head[i])
    return dq


def away_from_user(kin, track, expr, near):
    """Near the user, remove from each sample's offset every component that would bring an arm
    part closer to the person (first order, at the functional posture), weighted by `near`."""
    expr = expr.copy()
    for i in np.nonzero((near > 0.0) & np.any(expr != 0.0, axis=1))[0]:
        grads = kin.clearance_gradients(track.q[i], track.grip[i], track.held[i])[0]
        for _ in range(PROJECTION_SWEEPS):
            for g in grads:
                toward = g @ expr[i]
                if toward < 0.0 and g @ g > 0.0:
                    expr[i] -= near[i] * toward / (g @ g) * g
    return expr


# ------------------------------------------------------------------------------------- caps
def clearance_floor(func_clear):
    """(N, 2, parts) the least clearance each arm part may have, row 0 to the person (min of the
    functional clearance and NEAR_USER_M, less PERSON_SLACK_M), row 1 to the obstacles (min of
    the functional clearance and OBSTACLE_KEEP_M, less a numerical tolerance)."""
    floor = np.empty_like(func_clear)
    floor[:, 0] = np.minimum(func_clear[:, 0], NEAR_USER_M) - PERSON_SLACK_M
    floor[:, 1] = np.minimum(func_clear[:, 1], OBSTACLE_KEEP_M) - OBSTACLE_TOL_M
    return floor


def cap_violations(kin, track, q, func_clear, grip=None):
    """Every cap the composite q (and gripper, default the functional one) breaks, against the
    functional track: the expressive part's joint speed, acceleration and head speed; the
    composite's joint speed, ranges and guards; and every arm part's clearance floor to the person
    and the obstacles (the fingers count: a chomp can reach the counter)."""
    expr = q - track.q
    out = []
    v_e = np.gradient(expr, DT, axis=0)
    a_e = np.gradient(v_e, DT, axis=0)
    out += _over("expressive joint speed", np.abs(v_e) / EXPR_JOINT_SPEED, "rad/s", np.abs(v_e))
    out += _over("expressive joint accel", np.abs(a_e) / EXPR_JOINT_ACCEL, "rad/s^2", np.abs(a_e))
    head_speed = expressive_head_speed(kin, track, q)
    out += _over("expressive head speed", (head_speed / EXPR_HEAD_SPEED)[:, None], "m/s", head_speed[:, None])
    v = np.abs(np.gradient(q, DT, axis=0))
    room = np.maximum(VEL_LIMITS - np.abs(np.gradient(track.q, DT, axis=0)), 1e-9)
    out += _over("composite joint speed", np.where(v > VEL_LIMITS, np.abs(v_e) / room, 0.0), "rad/s", v)
    lim = guard_limits(kin)
    room = np.maximum(lim - np.abs(track.q), 1e-9)
    out += _over("joint limit/guard", np.where(np.abs(q) > lim, np.abs(expr) / room, 0.0), "rad", np.abs(q))
    if grip is not None:
        g_e = gripper_rate(grip - track.grip)[:, None]
        out += _over("expressive gripper speed", g_e / EXPR_GRIP_SPEED, "strokes/s", g_e)
    return out + _clearance_violations(kin, track, q, func_clear, grip)


def gripper_rate(dgrip):
    """(N,) strokes/s of a gripper offset from each sample to the next (a step shows in full)."""
    return np.abs(np.diff(dgrip, append=dgrip[-1])) / DT


def composite_clearances(kin, track, q, func_clear, grip=None):
    """(N, 2, parts) clearances of the composite q and gripper (default: the functional gripper);
    the functional ones where neither changed."""
    grip = track.grip if grip is None else grip
    out = func_clear.copy()
    for i in np.nonzero(np.any(q != track.q, axis=1) | (grip != track.grip))[0]:
        out[i] = kin.clearances(q[i], grip[i], track.held[i])
    return out


def _clearance_violations(kin, track, q, func_clear, grip=None):
    """A part below its floor: ratio = how much it lost / how much it was allowed to lose."""
    comp = composite_clearances(kin, track, q, func_clear, grip)
    floor = clearance_floor(func_clear)
    out = []
    for i, row, j in zip(*np.nonzero(comp < floor)):
        ratio = (func_clear[i, row, j] - comp[i, row, j]) / (func_clear[i, row, j] - floor[i, row, j])
        what = ("person", "obstacle")[row]
        detail = f"{ARM_PARTS[j]} {comp[i, row, j]:.3f} m < {floor[i, row, j]:.3f} at {i * DT:.2f} s"
        out.append(Violation(f"{what} clearance", int(i), float(ratio), detail, int(row), int(j)))
    return out


def guard_limits(kin):
    """(7,) |q| allowed per joint: the tighter of the joint range and RAMMP's guard."""
    lim = np.minimum(np.abs(kin.q_lo), np.abs(kin.q_hi))
    for j, g in GUARD_LIMITS.items():
        lim[j] = min(lim[j], g)
    return lim


def expressive_head_speed(kin, track, q):
    """(N,) speed of the head (the faster of HEAD_SPEED_SITES) due to the expressive part alone."""
    worst = np.zeros(track.n)
    for site in HEAD_SPEED_SITES:
        offset = kin.fk_track(q, site)[0] - kin.fk_track(track.q, site)[0]
        worst = np.maximum(worst, np.linalg.norm(np.gradient(offset, DT, axis=0), axis=1))
    return worst


def _over(kind, ratio, unit, value):
    """Violations where ratio (N, k) > 1, one per sample (its worst column: a joint when k = 7)."""
    worst = ratio.max(axis=1)
    col = ratio.argmax(axis=1)

    def detail(i):
        where = f" (joint {col[i] + 1})" if ratio.shape[1] == 7 else ""
        return f"{value[i, col[i]]:.3f} {unit}{where} at {i * DT:.2f} s"

    return [Violation(kind, int(i), float(worst[i]), detail(i)) for i in np.nonzero(worst > 1.0 + 1e-9)[0]]


def _activity(c, gains, window):
    """(N,) how fast a contribution moves the arm around each sample: the largest rate of change of
    its offsets times their zone gains (a gain ramp moves even a held offset) within `window`
    samples (head turns in rad/s, head shifts in HEAD_LEVER_M per second, joints in rad/s). Speed
    and acceleration violations are blamed on what moves."""
    gain, grip_gain = gains
    g = gain[:, None]
    rate = (
        np.maximum.reduce(
            [
                np.abs(np.gradient(g * c.head[:, 3:], DT, axis=0)).max(axis=1),
                np.abs(np.gradient(g * c.head[:, :3], DT, axis=0)).max(axis=1) / HEAD_LEVER_M,
                np.abs(np.gradient(g * c.joints, DT, axis=0)).max(axis=1),
                np.abs(np.gradient(grip_gain * c.grip, DT)),
            ]
        )
        * c.scale
    )
    return np.array([rate[max(0, i - window) : i + window + 1].max() for i in range(len(rate))])


def _clearance_effects(kin, track, gains, contributions, v):
    """How much each contribution alone (its joints and its gripper) changes the violated
    clearance at its sample (m)."""
    gain, grip_gain = gains
    i = v.index
    base = kin.clearances(track.q[i], track.grip[i], track.held[i])[v.row, v.part]
    effects = []
    for c in contributions:
        dq = gain[i] * (c.scale * c.joints[i] + kin.head_dq(track.q[i], c.scale * c.head[i]))
        grip = np.clip(track.grip[i] + grip_gain[i] * c.scale * c.grip[i], 0.0, 1.0)
        effects.append(kin.clearances(track.q[i] + dq, grip, track.held[i])[v.row, v.part] - base)
    return np.array(effects)


def _blamed(kin, track, gains, contributions, v, activity):
    """Contributions behind violation v: for a clearance floor, those whose own offset lowers that
    clearance by at least BLAME_SHARE of the worst one's; otherwise (or when none does alone)
    those moving at least BLAME_SHARE as fast there as the fastest."""
    if v.row >= 0:
        effect = _clearance_effects(kin, track, gains, contributions, v)
        if effect.min() < 0.0:
            return list(np.nonzero(effect <= BLAME_SHARE * effect.min())[0])
    here = activity[:, v.index]
    return [] if here.max() <= 0.0 else list(np.nonzero(here >= BLAME_SHARE * here.max())[0])


def _shrink(kin, track, gains, contributions, violations, log):
    """Scale each contribution down by its worst violation (CAP_MARGIN / ratio). Violations are
    visited worst first, at most MAX_BLAME_VISITS of them; False when nothing could be blamed.
    gains: (zone gain, gripper gain), each (N,)."""
    activity = np.array([_activity(c, gains, int(round(CAP_WINDOW_S / DT))) for c in contributions])
    worst = {}
    for v in sorted(violations, key=lambda v: -v.ratio)[:MAX_BLAME_VISITS]:
        for k in _blamed(kin, track, gains, contributions, v, activity):
            worst.setdefault(k, v)
        if len(worst) == len(contributions):
            break
    for k, v in worst.items():
        c = contributions[k]
        c.scale *= CAP_MARGIN / v.ratio
        log.append(f"{c.name}: amplitude x{c.scale:.2f} ({v.kind}: {v.detail})")
    return bool(worst)
