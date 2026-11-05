"""Composition: q = q_task(tau(t)) + gain(zone) * dq_expressive(t).

The task track comes in already time-warped. Behaviors give Contributions: head offsets (turned into
joint offsets by weighted DLS at the task posture), joint offsets and gripper offsets. compose() sums
them, applies the zone gain and the near-user projection, fits the caps, and returns the composed
track (with .func, .gain, .caps and .log). The guarantees:

- PRECISION: the gain is exactly 0 and q is copied from the task track (bit-identical).
- NEAR_USER: gain 0.35, and no offset component that would bring an arm part closer to the person.
- Caps tiered by zone (caps.CAP_PRESETS), the composite under the URDF velocity limits, the joint
  ranges and RAMMP's guards, and every arm part above its clearance floor (person: min of its task
  clearance and 0.35 m, less 5 mm; obstacles: min of its task clearance and 3 cm). A behavior that
  would break one scales its own amplitude down (logged, one factor for the whole behavior, so the
  motion stays smooth); the task motion is never changed.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .caps import (
    CAP_PRESETS, DEFAULT_CAPS, GAIN_RAMP_S, GRIP_GAINS, guard_limits, near_user_weight, sample_caps,
    sliding_max, zone_gain,
)
from .geometry import PARTS, clearances
from .track import DT, Label, Track, Zone
from .urdf import SITE_OFFSETS

NECK_LOCAL = SITE_OFFSETS["neck"]  # the wrist centre in the end-effector frame
HEAD_JOINT_WEIGHTS = np.array([25.0, 25.0, 15.0, 8.0, 1.0, 1.0, 1.0])  # a joint's cost: turns stay in the wrist
DLS_DAMPING = 0.03
HEAD_IK_ITERS = 4
EXPR_GRIP_SPEED = 0.150 / 0.085  # strokes/s, the gripper offset alone: the 2F-85's 150 mm/s over 85 mm
HEAD_SPEED_SITES = ("eye", "tool")  # the head's speed is the faster of these points
NEAR_USER_M = 0.35
PERSON_SLACK_M = 0.005  # a part may come this much closer to the person than min(task, 0.35)
OBSTACLE_KEEP_M = 0.03  # near obstacles a part keeps min(task clearance, this)
OBSTACLE_TOL_M = 0.0005  # numerical tolerance on that rule
CLEARANCE_CUTOFF_M = 0.6  # parts farther from the person than this do not constrain the projection
GRADIENT_STEP = 1e-5  # rad, finite differences of the clearance
PROJECTION_SWEEPS = 2  # passes over the parts' gradients when removing toward-the-user motion
CAP_MARGIN = 0.95  # a behavior over a cap shrinks to this fraction of what would just fit
CAP_WINDOW_S = 0.3  # a behavior active within this of a violation shares the blame
BLAME_SHARE = 0.25  # ... if it is at least this big a share of the biggest one there
MAX_BLAME_VISITS = 40  # violations examined (worst first) per fitting round
HEAD_LEVER_M = 0.1  # head shifts count as turns of shift / HEAD_LEVER_M when sharing blame
FIT_ROUNDS = 8


# ------------------------------------------------------------------------------ head offsets
def rot_err(R_target, R):
    """World-frame rotation vector taking R to R_target."""
    M = R_target @ R.T
    vee = 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s, c = np.linalg.norm(vee), 0.5 * (np.trace(M) - 1.0)
    return vee if s < 1e-12 else vee * (np.arctan2(s, c) / s)


def head_rotation(offset):
    """The head-frame rotation of a head offset [.., .., .., roll, pitch, yaw]: yaw about y, then
    pitch about x, then roll about z (a gimbal), R = Ry(yaw) Rx(pitch) Rz(roll)."""
    roll, pitch, yaw = offset[3], offset[4], offset[5]
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def head_target(chain, p, R, offset):
    """The ee pose after a head offset at ee pose (p, R): turned about the neck, then shifted."""
    R_new = R @ head_rotation(offset)
    neck = p + R @ NECK_LOCAL
    return neck - R_new @ NECK_LOCAL + np.asarray(offset)[:3], R_new


def dls(J, err, damping=DLS_DAMPING, weights=None):
    """Damped least squares step W^-1 J^T (J W^-1 J^T + damping^2 I)^-1 err (W = diag(weights))."""
    w_inv = np.ones(J.shape[1]) if weights is None else 1.0 / np.asarray(weights)
    JW = J * w_inv
    return JW.T @ np.linalg.solve(JW @ J.T + damping**2 * np.eye(J.shape[0]), err)


def head_dq(chain, q, offset6, iters=HEAD_IK_ITERS):
    """Joint offset (7,) that realises a head offset at posture q: weighted DLS on the ee Jacobian,
    iterated so the achieved offset matches. A zero offset gives exactly zero."""
    dq = np.zeros(7)
    if not np.any(offset6):
        return dq
    q = np.asarray(q, dtype=float)
    p_t, R_t = head_target(chain, *chain.fk(q, "ee"), offset6)
    for _ in range(iters):
        p, R = chain.fk(q + dq, "ee")
        dq = dq + dls(chain.jacobian(q + dq, "ee"), np.concatenate([p_t - p, rot_err(R_t, R)]), weights=HEAD_JOINT_WEIGHTS)
    return dq


# ------------------------------------------------------------------------------ contributions
@dataclass
class Contribution:
    """One behavior's share of the expressive layer on the composed track's clock (N samples):
    head (N, 6) offsets [dx, dy, dz] (world m) and [roll, pitch, yaw] (head-frame rad, applied
    yaw -> pitch -> roll about the neck), joints (N, 7), grip (N,), the labels it shows, the
    amplitude scale compose() settled on, and notes compose() copies into the track log."""

    name: str
    head: np.ndarray
    joints: np.ndarray
    grip: np.ndarray
    labels: List[Label] = field(default_factory=list)
    scale: float = 1.0
    notes: list = field(default_factory=list)

    @classmethod
    def empty(cls, name, n, labels=()):
        return cls(name, np.zeros((n, 6)), np.zeros((n, 7)), np.zeros(n), list(labels))

    @classmethod
    def combine(cls, name, parts, labels=None):
        """One contribution that is the sum of `parts` (each at its scale): a compound behavior
        that compose() scales as a whole."""
        out = cls.empty(name, len(parts[0].grip), [lb for p in parts for lb in p.labels] if labels is None else labels)
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


@dataclass
class ComposedTrack(Track):
    """The Puppy Mode track: a Track plus gain (N,), the task track `func` it rides on, the caps
    preset name and the log of what compose() did."""

    gain: Optional[np.ndarray] = None
    func: Optional[Track] = None
    caps: str = DEFAULT_CAPS
    log: List[str] = field(default_factory=list)


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
    joint: int = -1


# ---------------------------------------------------------------------------------- clearance
def clearances_of(chain, scene, q, held):
    """(2, 4) clearances of one posture (geometry.clearances)."""
    return clearances(chain, scene, np.asarray(q, dtype=float), bool(held))


class _Clearance:
    """geometry.clearances with a memo keyed by posture, so a fitting round only pays for the
    postures that changed."""

    def __init__(self, chain, scene):
        self.chain, self.scene, self._memo = chain, scene, {}

    def __call__(self, q, held):
        key = (np.asarray(q, dtype=float).tobytes(), bool(held))
        if key not in self._memo:
            self._memo[key] = clearances_of(self.chain, self.scene, q, held)
        return self._memo[key]

    def gradients(self, q, held):
        """(4, 7) d(person clearance)/dq per arm part by forward differences; zero beyond the cutoff."""
        base = self(q, held)[0]
        out = np.zeros((len(PARTS), 7))
        for k in range(7):
            step = np.zeros(7)
            step[k] = GRADIENT_STEP
            out[:, k] = (clearances_of(self.chain, self.scene, q + step, held)[0] - base) / GRADIENT_STEP
        out[~(base < CLEARANCE_CUTOFF_M)] = 0.0
        return out


def clearance_floor(task_clear):
    """(..., 2, parts) the least clearance each arm part may have: row 0 to the person, row 1 to the
    obstacles."""
    floor = np.empty_like(task_clear)
    floor[..., 0, :] = np.minimum(task_clear[..., 0, :], NEAR_USER_M) - PERSON_SLACK_M
    floor[..., 1, :] = np.minimum(task_clear[..., 1, :], OBSTACLE_KEEP_M) - OBSTACLE_TOL_M
    return floor


def away_from_user(clear, track, expr, near):
    """Near the user, remove from each sample's offset every component that would bring an arm
    part closer to the person (first order, at the task posture), weighted by `near`."""
    expr = expr.copy()
    for i in np.nonzero((near > 0.0) & np.any(expr != 0.0, axis=1))[0]:
        grads = clear.gradients(track.q[i], track.held[i])
        for _ in range(PROJECTION_SWEEPS):
            for g in grads:
                toward = g @ expr[i]
                if toward < 0.0 and g @ g > 0.0:
                    expr[i] -= near[i] * toward / (g @ g) * g
    return expr


# ------------------------------------------------------------------------------------- caps
def fdiff(x, n=1):
    """(N, ...) forward differences / DT**n of order n, aligned to the first sample of each stencil
    (zero at the end). Unlike central differences they see a sample-to-sample alternation."""
    d = np.diff(x, n=n, axis=0) / DT**n
    return np.concatenate([d, np.zeros((n,) + x.shape[1:])])


def gripper_rate(dgrip):
    """(N,) strokes/s of a gripper offset from each sample to the next (a step shows in full)."""
    return np.abs(fdiff(np.asarray(dgrip, dtype=float)))


def expressive_head_speed(chain, track, q):
    """(N,) speed of the head (the faster of HEAD_SPEED_SITES) due to the expressive part alone."""
    changed = np.any(q != track.q, axis=1)
    worst = np.zeros(track.n)
    for site in HEAD_SPEED_SITES:
        offset = np.zeros((track.n, 3))
        for i in np.nonzero(changed)[0]:
            offset[i] = chain.fk(q[i], site)[0] - chain.fk(track.q[i], site)[0]
        worst = np.maximum(worst, np.linalg.norm(fdiff(offset), axis=1))
    return worst


def _over(kind, ratio, unit, value, cap):
    """Violations where ratio (N, k) > 1, one per sample (its worst column: a joint when k = 7)."""
    worst = ratio.max(axis=1)
    col = ratio.argmax(axis=1)
    cap = np.broadcast_to(cap, value.shape)
    seven = ratio.shape[1] == 7

    def detail(i):
        where = f" (joint {col[i] + 1})" if seven else ""
        return f"{value[i, col[i]]:.3f} {unit} > {cap[i, col[i]]:.3f}{where} at {i * DT:.2f} s"

    return [
        Violation(kind, int(i), float(worst[i]), detail(i), joint=int(col[i]) if seven else -1)
        for i in np.nonzero(worst > 1.0 + 1e-9)[0]
    ]


def cap_violations(chain, scene, track, q, grip=None, caps=DEFAULT_CAPS, clear=None):
    """Every cap the composite q (and gripper, default the task one) breaks against the task
    `track`: the expressive part's joint speed, acceleration and head speed (the `caps` preset on
    the track's zones); the composite's joint speed, ranges and guards; the expressive gripper
    speed; and every arm part's clearance floor to the person and the obstacles. Speeds are
    per-sample (forward) differences, so a jitter between samples counts in full."""
    clear = clear or _Clearance(chain, scene)
    limits = sample_caps(track.zone, caps)
    vel = chain.vel_limits
    expr = q - track.q
    v_e = np.abs(fdiff(expr))
    a_e = np.abs(fdiff(expr, 2))
    out = _over("expressive joint speed", v_e / limits.joint_speed, "rad/s", v_e, limits.joint_speed)
    out += _over("expressive joint accel", a_e / limits.joint_accel, "rad/s^2", a_e, limits.joint_accel)
    head = expressive_head_speed(chain, track, q)[:, None]
    out += _over("expressive head speed", head / limits.head_speed[:, None], "m/s", head, limits.head_speed[:, None])
    v = np.abs(fdiff(q))
    room = np.maximum(vel - np.abs(fdiff(track.q)), 1e-9)
    out += _over("composite joint speed", np.where(v > vel, v_e / room, 0.0), "rad/s", v, vel[None])
    lim = guard_limits(chain)
    room = np.maximum(lim - np.abs(track.q), 1e-9)
    out += _over("joint limit/guard", np.where(np.abs(q) > lim, np.abs(expr) / room, 0.0), "rad", np.abs(q), lim[None])
    if grip is not None:
        g_e = gripper_rate(grip - track.grip)[:, None]
        out += _over("expressive gripper speed", g_e / EXPR_GRIP_SPEED, "strokes/s", g_e, np.array([[EXPR_GRIP_SPEED]]))
    return out + _clearance_violations(clear, track, q)


def _clearance_violations(clear, track, q):
    """A part below its floor: ratio = how much it lost / how much it was allowed to lose. The
    capsule model has no gripper, so only the arm's posture matters."""
    out = []
    for i in np.nonzero(np.any(q != track.q, axis=1))[0]:
        base, comp = clear(track.q[i], track.held[i]), clear(q[i], track.held[i])
        floor = clearance_floor(base)
        for row, j in zip(*np.nonzero(comp < floor)):
            ratio = (base[row, j] - comp[row, j]) / (base[row, j] - floor[row, j])
            what = ("person", "obstacle")[row]
            detail = f"{PARTS[j]} {comp[row, j]:.3f} m < {floor[row, j]:.3f} at {i * DT:.2f} s"
            out.append(Violation(f"{what} clearance", int(i), float(ratio), detail, int(row), int(j)))
    return out


# --------------------------------------------------------------------------------- composing
def compose(warped, parts, chain, scene, caps=DEFAULT_CAPS):
    """The Puppy Mode track: the warped task track plus `parts` (each scaled in place when it would
    exceed a cap or a clearance floor), fitted to the CAP_PRESETS preset `caps`.

    The result always satisfies every cap, guard and clearance floor: a contribution that cannot
    be fitted by scaling is switched off (scale 0), the implicated ones first, then all."""
    if caps not in CAP_PRESETS:
        raise ValueError(f"unknown caps preset {caps!r}; expected one of {sorted(CAP_PRESETS)}")
    fit = _Fit(chain, scene, warped, parts, caps)
    log = [f"{c.name}: {note}" for c in parts for note in c.notes]
    q, grip, violations = fit.realise()
    for _ in range(FIT_ROUNDS):
        if not violations or not fit.shrink(violations, log):
            break
        q, grip, violations = fit.realise()  # recompose after every shrink
    if violations:
        log.append(f"still over a cap after fitting: {len(violations)} samples, worst {violations[0].kind}")
        for c in fit.implicated(violations) or parts:
            c.scale = 0.0
            log.append(f"{c.name}: switched off (amplitude x0.00): it cannot be fitted to the caps")
        q, grip, violations = fit.realise()
        if violations:  # the implicated ones were not the whole story
            for c in parts:
                c.scale = 0.0
            log.append("all contributions switched off: fitting could not satisfy the caps")
            q, grip, violations = fit.realise()
    labels = sorted([*warped.labels, *(lb for c in parts for lb in c.labels)], key=lambda lb: lb.t0)
    return ComposedTrack(
        q=q,
        grip=grip,
        zone=warped.zone.copy(),
        held=warped.held.copy(),
        tau=warped.tau.copy(),
        labels=labels,
        source=warped.source if warped.source is not None else warped,
        gain=fit.gain,
        func=warped,
        caps=caps,
        log=log,
    )


class _Fit:
    """One compose() run: the gains, the clearance memo, and the blame bookkeeping."""

    def __init__(self, chain, scene, track, parts, caps):
        self.chain, self.scene, self.track, self.parts, self.caps = chain, scene, track, parts, caps
        self.gain = zone_gain(track.zone)
        self.grip_gain = zone_gain(np.where(track.held, Zone.PRECISION, track.zone), GRIP_GAINS)
        self.near = near_user_weight(track.zone)
        self.clear = _Clearance(chain, scene)
        self.memo = {}
        self.realised = []

    def _head_to_joints(self, head):
        """Joint offsets realising the head offsets at each task posture (where gain > 0)."""
        dq = np.zeros((self.track.n, 7))
        for i in np.nonzero((self.gain > 0.0) & np.any(head != 0.0, axis=1))[0]:
            key = (i, head[i].tobytes())
            if key not in self.memo:
                self.memo[key] = head_dq(self.chain, self.track.q[i], head[i])
            dq[i] = self.memo[key]
        return dq

    def realise(self):
        """(q, grip, violations) of the contributions at their current scales. Also records each
        contribution's own realised joint offsets, gain * (joints + head_dq(head)), for blaming."""
        t = self.track
        head, joints, dgrip = _summed(self.parts, t.n)
        self.realised = [
            self.gain[:, None] * (c.scale * c.joints + self._head_to_joints(c.scale * c.head)) for c in self.parts
        ]
        expr = self.gain[:, None] * (joints + self._head_to_joints(head))
        expr = away_from_user(self.clear, t, expr, self.near)
        q = np.where((self.gain == 0.0)[:, None], t.q, t.q + expr)
        grip = np.where(self.grip_gain == 0.0, t.grip, np.clip(t.grip + self.grip_gain * dgrip, 0.0, 1.0))
        return q, grip, cap_violations(self.chain, self.scene, t, q, grip, self.caps, self.clear)

    def _activity(self, k):
        """(N,) how fast contribution k moves the arm around each sample: its realised joint
        offsets' speed and acceleration (in rad/s), windowed, so blame follows what moves."""
        window = int(round(CAP_WINDOW_S / DT))
        off = self.realised[k]
        rate = np.abs(fdiff(off)).max(axis=1)
        accel = np.abs(fdiff(off, 2)).max(axis=1) * DT
        return sliding_max(np.maximum(rate, accel), window)

    def _grip_activity(self, k):
        c = self.parts[k]
        return sliding_max(gripper_rate(self.grip_gain * c.scale * c.grip), int(round(CAP_WINDOW_S / DT)))

    def _clearance_effects(self, v):
        """How much each contribution alone (its realised offset) changes the violated clearance (m)."""
        i, held = v.index, self.track.held[v.index]
        base = self.clear(self.track.q[i], held)[v.row, v.part]
        return np.array([self.clear(self.track.q[i] + off[i], held)[v.row, v.part] - base for off in self.realised])

    def _weights(self, v):
        """(contributions,) how much each is behind violation v, from its own realised offset."""
        i, parts = v.index, range(len(self.parts))
        if v.row >= 0:  # clearance: together they may break it though neither does alone
            return np.array([np.abs(off[i]).max() for off in self.realised])
        if v.kind == "expressive gripper speed":
            return np.array([self._grip_activity(k)[i] for k in parts])
        if v.kind == "joint limit/guard":
            return np.array([np.abs(off[i, v.joint]) for off in self.realised])
        if v.kind == "composite joint speed":
            return np.array([np.abs(fdiff(off)[i, v.joint]) for off in self.realised])
        return np.array([self._activity(k)[i] for k in parts])

    def blamed(self, v):
        """Contributions behind violation v: for a clearance floor, those whose own offset lowers
        that clearance by at least BLAME_SHARE of the worst one's; otherwise those whose realised
        weight there (see _weights) is at least BLAME_SHARE of the biggest. A held offset has no
        activity of its own but its realised offset still counts."""
        if v.row >= 0:
            effect = self._clearance_effects(v)
            if effect.min() < 0.0:
                return list(np.nonzero(effect <= BLAME_SHARE * effect.min())[0])
        weight = self._weights(v)
        if weight.max() <= 0.0 and v.kind != "expressive gripper speed":
            weight = np.array([np.abs(off[v.index]).max() for off in self.realised])
        return [] if weight.max() <= 0.0 else list(np.nonzero(weight >= BLAME_SHARE * weight.max())[0])

    def implicated(self, violations):
        """Contributions blamed by any of the remaining violations (worst first, bounded)."""
        out = set()
        for v in sorted(violations, key=lambda v: -v.ratio)[:MAX_BLAME_VISITS]:
            out.update(self.blamed(v))
        return [self.parts[k] for k in sorted(out)]

    def shrink(self, violations, log):
        """Scale each contribution down by its worst violation (CAP_MARGIN / ratio). Violations are
        visited worst first, at most MAX_BLAME_VISITS of them; False when nothing could be blamed."""
        worst = {}
        for v in sorted(violations, key=lambda v: -v.ratio)[:MAX_BLAME_VISITS]:
            for k in self.blamed(v):
                worst.setdefault(k, v)
            if len(worst) == len(self.parts):
                break
        for k, v in worst.items():
            c = self.parts[k]
            c.scale *= CAP_MARGIN / v.ratio
            log.append(f"{c.name}: amplitude x{c.scale:.2f} ({v.kind}: {v.detail})")
        return bool(worst)


def _summed(parts, n):
    head, joints, grip = np.zeros((n, 6)), np.zeros((n, 7)), np.zeros(n)
    for c in parts:
        head += c.scale * c.head
        joints += c.scale * c.joints
        grip += c.scale * c.grip
    return head, joints, grip
