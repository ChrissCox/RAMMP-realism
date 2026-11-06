"""Wrap-aware joint arithmetic for continuous joints (no range limit).

The real driver reports continuous joints wrapped to +-pi, while stored paths use whatever turn
they were planned on. The same posture can therefore read 2*pi apart; these helpers compare and
align such readings. Revolute joints are never wrapped."""

import numpy as np

from .caps import guard_limits


def continuous(chain):
    """(7,) bool: the joints with no position limit."""
    return ~np.isfinite(guard_limits(chain))


def shortest_delta(chain, q_from, q_to):
    """q_to - q_from, wrapped to (-pi, pi] on the continuous joints, plain on the others."""
    d = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    c = continuous(chain)
    d[c] = np.pi - np.mod(np.pi - d[c], 2.0 * np.pi)
    return d


def align_turns(chain, q_ref, points):
    """`points` [(t, q, dq), ...] with each continuous joint shifted by one whole number of turns,
    the same for every point, so the first point is within pi of `q_ref` on that joint.
    Velocities are unchanged."""
    q_ref = np.asarray(q_ref, dtype=float)
    shift = np.zeros(7)
    c = continuous(chain)
    shift[c] = 2.0 * np.pi * np.round((q_ref[c] - np.asarray(points[0][1], dtype=float)[c]) / (2.0 * np.pi))
    return [(t, np.asarray(q, dtype=float) + shift, dq) for t, q, dq in points]
