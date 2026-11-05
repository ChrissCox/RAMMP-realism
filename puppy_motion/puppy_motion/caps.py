"""Zone gains, their raised-cosine ramps, and the expressive part's caps tiered by zone.

The expressive part rides on the task's motion with a gain per zone (FREE 1, NEAR_USER 0.35,
PRECISION 0). Gains ramp over GAIN_RAMP_S with a raised cosine and finish GAIN_GUARD_SAMPLES before a
lower zone begins, so no sample of a zone ever sees more than its own level. The caps are tiered:
each sample takes the caps of the most restrictive zone within a gain ramp's reach, so a ramp
toward NEAR_USER or PRECISION is already under the gentle caps.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .track import DT, Zone

GAINS = {Zone.FREE: 1.0, Zone.NEAR_USER: 0.35, Zone.PRECISION: 0.0}
GRIP_GAINS = {Zone.FREE: 1.0, Zone.NEAR_USER: 0.0, Zone.PRECISION: 0.0}  # and 0 while holding
GAIN_RAMP_S = 0.3
GAIN_GUARD_SAMPLES = 1  # a ramp down ends this many samples before the lower zone (never inside it)
DEFAULT_CAPS = "default"

# FREE: inside the envelope the task's own motions use in free space (a test holds these under the
# measured per-joint peaks).
TASK_JOINT_SPEED = (0.36, 1.12, 0.59, 0.64, 0.97, 0.87, 0.97)  # rad/s, the task's peak per joint, rounded down
FREE_JOINT_SPEED = 0.85  # rad/s per joint, and no faster than the task moves it
FREE_JOINT_ACCEL = 3.0  # rad/s^2
FREE_HEAD_SPEED = 0.40  # m/s at the eye and tool points
# NEAR_USER (and the gain ramps into it or into PRECISION)
NEAR_JOINT_SPEED = 0.5
NEAR_JOINT_ACCEL = 1.5
NEAR_HEAD_SPEED = 0.16
SNAPPY_WRIST_ACCEL = 9.0  # 'snappy': joints 5-7 in FREE only; never the default

# RAMMP's joint guards, 0-based joint index -> |q| max (rad): |q2| <= 2.24, |q4| <= 144 deg, |q6| <= 115 deg
URDF_GUARDS = {1: 2.24, 3: float(np.radians(144.0)), 5: float(np.radians(115.0))}


@dataclass(frozen=True, eq=False)
class CapSet:
    """The expressive part's limits in one zone: per joint speed (rad/s) and acceleration (rad/s^2),
    each (7,), and the head's speed (m/s, at the eye and tool points)."""

    joint_speed: np.ndarray
    joint_accel: np.ndarray
    head_speed: float


def _within_task(speed):
    return np.minimum(speed, np.array(TASK_JOINT_SPEED))


def _wrist_accel(base, wrist):
    out = np.full(7, base, dtype=float)
    out[4:7] = wrist
    return out


_FREE = CapSet(_within_task(FREE_JOINT_SPEED), np.full(7, FREE_JOINT_ACCEL), FREE_HEAD_SPEED)
_GENTLE = CapSet(_within_task(NEAR_JOINT_SPEED), np.full(7, NEAR_JOINT_ACCEL), NEAR_HEAD_SPEED)
_SNAPPY_FREE = CapSet(_within_task(FREE_JOINT_SPEED), _wrist_accel(FREE_JOINT_ACCEL, SNAPPY_WRIST_ACCEL), FREE_HEAD_SPEED)

# preset -> zone -> caps. PRECISION carries no expression (gain 0): its caps bound the ramps into it.
CAP_PRESETS: Dict[str, Dict[Zone, CapSet]] = {
    "default": {Zone.FREE: _FREE, Zone.NEAR_USER: _GENTLE, Zone.PRECISION: _GENTLE},
    "snappy": {Zone.FREE: _SNAPPY_FREE, Zone.NEAR_USER: _GENTLE, Zone.PRECISION: _GENTLE},
}


def raised_cosine(u):
    """0..1 -> 0..1 half-cosine: zero velocity at both ends (clipped outside)."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def sliding_max(x, reach):
    """Maximum of x over [i - reach, i + reach] at every i (edges repeat)."""
    x = np.asarray(x)
    padded = np.pad(x, reach, mode="edge")
    out = x.copy()
    for k in range(1, 2 * reach + 1):
        out = np.maximum(out, padded[k : k + len(x)])
    return out


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
        w = raised_cosine((k - guard) / n_ramp) if k > guard else 0.0
        for shifted in (padded[reach - k : reach - k + len(level)], padded[reach + k : reach + k + len(level)]):
            out = np.minimum(out, shifted + (1.0 - shifted) * w)
    return out


def zone_gain(zone, gains=GAINS):
    """Per-sample gain from the zones: each sample's level, with raised-cosine ramps of GAIN_RAMP_S
    that finish GAIN_GUARD_SAMPLES before a lower zone starts and begin as long after it ends."""
    return ramped_floor([gains[Zone(int(z))] for z in zone], guard=GAIN_GUARD_SAMPLES)


def near_user_weight(zone):
    """1 on NEAR_USER samples, easing to 0 around them like a gain ramp."""
    return 1.0 - ramped_floor((np.asarray(zone) != Zone.NEAR_USER).astype(float), guard=GAIN_GUARD_SAMPLES)


@dataclass(frozen=True)
class SampleCaps:
    """A preset's caps at every sample of a track: joint_speed and joint_accel (N, 7), head_speed
    (N,), and tier (N,), the zone whose caps apply there (cap_tier)."""

    joint_speed: np.ndarray
    joint_accel: np.ndarray
    head_speed: np.ndarray
    tier: np.ndarray


def cap_tier(zone):
    """(N,) the zone whose caps apply at each sample: the most restrictive zone (PRECISION >
    NEAR_USER > FREE) within a gain ramp's reach."""
    reach = GAIN_GUARD_SAMPLES + max(1, int(round(GAIN_RAMP_S / DT)))
    return sliding_max(np.asarray(zone, dtype=int), reach)


def sample_caps(zone, caps=DEFAULT_CAPS):
    """The preset `caps` (a CAP_PRESETS name) at every sample of a track with zones `zone`."""
    preset = CAP_PRESETS[caps]
    tier = cap_tier(zone)
    by_zone = [preset[z] for z in sorted(Zone)]
    return SampleCaps(
        joint_speed=np.array([c.joint_speed for c in by_zone])[tier],
        joint_accel=np.array([c.joint_accel for c in by_zone])[tier],
        head_speed=np.array([c.head_speed for c in by_zone])[tier],
        tier=tier,
    )


def guard_limits(chain):
    """(7,) |q| allowed per joint: the tighter of the URDF joint range and RAMMP's guard."""
    lim = np.minimum(np.abs(chain.pos_limits[:, 0]), np.abs(chain.pos_limits[:, 1]))
    for j, g in URDF_GUARDS.items():
        lim[j] = min(lim[j], g)
    return lim
