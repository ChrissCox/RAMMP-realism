"""A playable beat: named, moody, and as the points check_beat and the arm's trajectory goal take."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import behaviors as B
from .caps import DEFAULT_CAPS
from .compose import compose
from .geometry import PARTS, clearances
from .track import DT, Track, Zone, constant_track
from .validate import check_beat

NEAR_USER_M = 0.35
MOVED_RAD = 1e-3  # a beat that moves less than this was refused down to nothing
TAIL_S = 0.4
HEAD = PARTS.index("head")


def to_points(track: Track) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """[(t, q, dq), ...] on the DT clock: dq is the central difference of q, exactly 0 at the first
    and last point (the convention check_beat holds beats to)."""
    q = np.asarray(track.q, dtype=float)
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (2 * DT)
    return [(k * DT, q[k].copy(), dq[k]) for k in range(len(q))]


@dataclass
class Beat:
    name: str
    mood: str
    points: List[Tuple[float, np.ndarray, np.ndarray]]
    zone: Zone

    @property
    def duration(self) -> float:
        return self.points[-1][0]


def zone_at(chain, scene, q, held, floor=Zone.FREE) -> Zone:
    """NEAR_USER when the head (with the held container) is under NEAR_USER_M from the person at q,
    else `floor`; never below `floor`."""
    near = clearances(chain, scene, np.asarray(q, dtype=float), bool(held))[0, HEAD] < NEAR_USER_M
    return max(Zone(floor), Zone.NEAR_USER) if near else Zone(floor)


def compose_beat(
    name, mood, builder, length_s, q_now, held, zone, chain, scene, targets, seed,
    size=1.0, caps=DEFAULT_CAPS, tail_s=TAIL_S,
) -> Tuple[Optional[Beat], Optional[str]]:
    """(beat, None) or (None, why): builder(ctx, rng, held) -> [Contribution] composed on a track
    standing still at q_now for length_s, trimmed to its motion plus tail_s, then held to check_beat.
    `size` multiplies every contribution. Anything that cannot be built or does not validate is
    refused with a message (never raised, never returned as a beat)."""
    q_now, held = np.asarray(q_now, dtype=float), bool(held)
    sub = np.random.default_rng(seed)
    track = constant_track(q_now, 0.0, length_s, zone, name)
    track.held[:] = held
    try:
        ctx = B.Context.of(track, chain, scene, targets, seed, caps)
        parts = builder(ctx, sub, held)
        for p in parts:
            p.head *= size
            p.joints *= size
        composed = compose(track, parts, chain, scene, caps)
    except Exception as e:  # a behavior that cannot be built here is skipped, not fatal
        return None, f"{name}: not built ({type(e).__name__}: {e})"
    dev = np.abs(composed.q - q_now).max(axis=1)
    if dev.max() < MOVED_RAD:
        return None, f"{name}: no room to move here"
    end = min(int(np.nonzero(dev > 1e-9)[0][-1]) + int(round(tail_s / DT)), composed.n - 1)
    composed.q = composed.q[: end + 1]
    composed.q[0] = composed.q[-1] = q_now  # exactly at q_now at both ends (they differ by < 1e-9)
    points = to_points(composed)
    found = check_beat(points, q_now, chain, scene, zone, caps, held=held)
    if found:
        return None, f"{name}: refused ({found[0]})"
    return Beat(name, mood, points, zone), None
