"""A playable beat: named, moody, and as the points check_beat and the arm's trajectory goal take."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .track import DT, Track, Zone


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
