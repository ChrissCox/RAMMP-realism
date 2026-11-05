"""The one clock (DT), zones, Track, labels, and the time-warp.

A Track is a sampled arm motion on the 50 Hz clock: joints, gripper, zone, held flag and labels.
The time-warp maps output time t to functional time tau with speed dtau/dt in [0, 1]: it can
pause (Hold) or slow down (Slow) the functional motion but never speed it up or change its path,
because the warped track is always the functional path evaluated at tau.

Warps are computed in sample units (u = functional sample index). Ramps take an even number of
samples, so a hold stands still exactly on the sample nearest its time and the samples after it
land exactly on functional samples (until a slow-in makes the timing fractional).
"""

from dataclasses import dataclass, field

import numpy as np

from . import curves

DT = 0.02  # s: RAMMP's trajectory rate, 50 Hz
FREE, NEAR_USER, PRECISION = 0, 1, 2
ZONE_NAMES = {FREE: "FREE", NEAR_USER: "NEAR_USER", PRECISION: "PRECISION"}
GRIP_TO_KNUCKLE_RAD = 0.8  # Robotiq knuckle angle (rad) at grip 1.0 (closed)


@dataclass(frozen=True)
class Label:
    """A named span [t0, t1) of a track's output time: a task phase or an expressive beat."""

    name: str
    t0: float
    t1: float


@dataclass
class Track:
    """A sampled arm motion on the DT clock (output time t = k * DT).

    q: (N, 7) rad; grip: (N,) 0 open..1 closed; zone: (N,) FREE/NEAR_USER/PRECISION;
    held: (N,) bool, the container rides in the gripper; tau: (N,) functional time of each sample.
    A resampled (time-warped) track carries `source`, the functional track it follows (tau is that
    track's time), so checks can hold it to that path. A composed (Puppy Mode) track also carries
    `func` (the warped functional track on the same clock), `gain` (N,) and a `log` of what the
    composition changed."""

    q: np.ndarray
    grip: np.ndarray
    zone: np.ndarray
    held: np.ndarray
    tau: np.ndarray
    phases: list = field(default_factory=list)
    beats: list = field(default_factory=list)
    func: "Track | None" = None
    gain: np.ndarray | None = None
    log: list = field(default_factory=list)
    name: str = ""
    source: "Track | None" = None

    @property
    def n(self):
        return len(self.q)

    @property
    def t(self):
        return np.arange(self.n) * DT

    @property
    def duration(self):
        return (self.n - 1) * DT

    def beat_at(self, t):
        """The last beat covering time t, or None."""
        hits = [b for b in self.beats if b.t0 <= t < b.t1]
        return hits[-1] if hits else None


def interp_rows(x, u):
    """Rows of x (N, ...) linearly interpolated at fractional sample indices u.

    Exact (bit for bit) at whole indices, so a warp that lands on samples reproduces them."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, len(x) - 1)
    i = np.minimum(np.floor(u).astype(int), len(x) - 2)
    f = (u - i).reshape(u.shape + (1,) * (x.ndim - 1))
    lo, hi = x[i], x[i + 1]
    return np.where(f == 0.0, lo, np.where(f == 1.0, hi, lo + f * (hi - lo)))


def resample_track(track, u):
    """`track` evaluated at fractional sample indices u: the path is unchanged, only its timing.

    Zones take the stricter neighbour (PRECISION > NEAR_USER > FREE), so a sample between a free
    and a precision sample counts as precision. Labels move to the first output time reaching them."""
    u = np.asarray(u, dtype=float)
    lo = np.clip(np.floor(u).astype(int), 0, track.n - 1)
    hi = np.clip(np.ceil(u).astype(int), 0, track.n - 1)
    src_t = u * DT  # each output sample's time on the source track's own clock

    def moved(lb):
        return Label(lb.name, _first_reaching(src_t, lb.t0), _first_reaching(src_t, lb.t1))

    return Track(
        q=interp_rows(track.q, u),
        grip=interp_rows(track.grip, u),
        zone=np.maximum(track.zone[lo], track.zone[hi]),
        held=track.held[lo] | track.held[hi],
        tau=interp_rows(track.tau, u),
        phases=[moved(p) for p in track.phases],
        beats=[moved(b) for b in track.beats],
        name=track.name,
        source=track if track.source is None else track.source,
    )


def _first_reaching(src_t, t):
    """Output time of the first sample whose source time reaches t (the end if none does)."""
    return int(np.searchsorted(src_t, t - 1e-9)) * DT


@dataclass(frozen=True)
class Hold:
    """Pause the functional motion around functional time `at`: ease to a stop over ramp_s
    (reaching speed 0 at `at`), stand still for hold_s, ease back to full speed over ramp_s."""

    at: float
    hold_s: float
    ramp_s: float

    def __post_init__(self):
        if self.hold_s < 0.0 or self.ramp_s <= 0.0:
            raise ValueError(f"a hold needs hold_s >= 0 and ramp_s > 0 (a bounded stop): {self}")


@dataclass(frozen=True)
class Slow:
    """Run the functional motion at `factor` (< 1) of its speed over functional times [t0, t1],
    easing into and out of it over ramp_s. With hold_s > 0 it first comes to a stop at t0 (easing
    down over ramp_s) and stands still for hold_s, then eases up to `factor`: a lock-on stare
    followed by a slow approach, with no speed bump in between."""

    t0: float
    t1: float
    factor: float
    ramp_s: float
    hold_s: float = 0.0

    def __post_init__(self):
        if not 0.0 < self.factor <= 1.0:
            raise ValueError(f"a slow-in keeps 0 < factor <= 1 (it never speeds up): {self}")
        if self.t1 < self.t0 or self.ramp_s <= 0.0 or self.hold_s < 0.0:
            raise ValueError(f"a slow-in needs t1 >= t0, ramp_s > 0 and hold_s >= 0: {self}")


STEP_TOL = 1e-9  # float noise ignored when rounding a ramp up to whole samples


def _steps(seconds):
    """Seconds -> a whole number of samples (at least 1)."""
    return max(1, int(round(seconds / DT)))


def _ramp_steps(seconds):
    """Seconds -> an even number of samples (at least 2), never shorter than asked (a ramp sized
    for an acceleration cap keeps it): easing 1 -> 0 over it advances a whole number of samples."""
    return 2 * max(1, int(np.ceil(seconds / (2.0 * DT) - STEP_TOL)))


def _ease_speed(n, s0, s1):
    """Functional advance (samples) at each of n+1 output samples while the speed eases from s0
    to s1 with a raised cosine over n samples. Ends exactly on n * (s0 + s1) / 2."""
    x = np.arange(n + 1, dtype=float)
    advance = s0 * x + (s1 - s0) * (x / 2.0 - n / (2.0 * np.pi) * np.sin(np.pi * x / n))
    advance[-1] = n * (s0 + s1) / 2.0
    return advance


def _hold_segment(hold):
    """Advance during a hold: ease 1 -> 0, stand still, ease 0 -> 1. Totals n_ramp samples."""
    n_ramp = _ramp_steps(hold.ramp_s)
    down = _ease_speed(n_ramp, 1.0, 0.0)
    still = np.full(_steps(hold.hold_s), down[-1])
    up = down[-1] + _ease_speed(n_ramp, 0.0, 1.0)[1:]
    return np.concatenate([down, still, up])


def _slow_segment(slow, u_start):
    """Advance during a slow-in: ease 1 -> factor (or 1 -> 0, stand still, 0 -> factor when it
    holds first), cruise at factor until t1, ease back to 1."""
    n_ramp = _ramp_steps(slow.ramp_s)
    if slow.hold_s > 0.0:
        down = _ease_speed(n_ramp, 1.0, 0.0)
        still = np.full(_steps(slow.hold_s), down[-1])
        lead_in = np.concatenate([down, still, down[-1] + _ease_speed(n_ramp, 0.0, slow.factor)[1:]])
    else:
        lead_in = _ease_speed(n_ramp, 1.0, slow.factor)
    up_advance = n_ramp * (1.0 + slow.factor) / 2.0
    to_cruise = slow.t1 / DT - u_start - lead_in[-1] - up_advance
    n_cruise = max(0, int(np.ceil(to_cruise / slow.factor)))
    cruise = lead_in[-1] + slow.factor * np.arange(1, n_cruise + 1, dtype=float)
    last = cruise[-1] if n_cruise else lead_in[-1]
    up = last + _ease_speed(n_ramp, slow.factor, 1.0)[1:]
    return np.concatenate([lead_in, cruise, up])


def warp_indices(n_func, holds=(), slows=()):
    """Fractional functional sample index u of every output sample (u[0] = 0, last = n_func - 1).

    Events run in order of their functional start; one that starts before the previous one ended
    starts where that one ended."""
    u_end = float(n_func - 1)
    pieces, u = [np.array([0.0])], 0.0
    for ev in sorted([*holds, *slows], key=_start_u):
        n_free = max(0, int(round(_start_u(ev) - u)))
        if u + n_free >= u_end:
            break
        pieces.append(u + np.arange(1, n_free + 1, dtype=float))
        u += n_free
        segment = _hold_segment(ev) if isinstance(ev, Hold) else _slow_segment(ev, u)
        pieces.append(u + segment[1:])
        u += segment[-1]
    pieces.append(u + np.arange(1, int(np.floor(u_end - u)) + 1, dtype=float))
    out = np.minimum(np.concatenate(pieces), u_end)
    return out if out[-1] == u_end else np.append(out, u_end)


def _start_u(ev):
    """Functional sample index where a Hold or Slow starts to change the speed: one that stops
    eases down so that it stands still on the sample nearest its time."""
    if isinstance(ev, Hold):
        return round(ev.at / DT) - _ramp_steps(ev.ramp_s) // 2
    if ev.hold_s > 0.0:
        return round(ev.t0 / DT) - _ramp_steps(ev.ramp_s) // 2
    return round(ev.t0 / DT)


def warp(track, holds=(), slows=()):
    """The functional `track` time-warped by holds and slows: the same path with new timing."""
    return resample_track(track, warp_indices(track.n, holds, slows))


def speed(track):
    """dtau/dt at each output sample (forward difference; the last sample repeats)."""
    s = np.diff(track.tau) / DT
    return np.append(s, s[-1]) if len(s) else np.ones(1)


def output_time(track, tau):
    """Output time at which a (warped) track reaches functional time tau: where it starts standing
    still there if it does (a hold stops within half a sample of its time), else its first sample
    at or past tau."""
    span = still_span(track, tau, DT / 2.0)
    if span is not None:
        return span[0]
    return min(int(np.searchsorted(track.tau, tau - 1e-9)), track.n - 1) * DT


def still_span(track, tau, tol=DT):
    """(t0, t1) output times over which a warped track stands still (dtau/dt = 0) within tol of
    functional time tau: the stare of a hold. None if it never stands still there."""
    still = np.nonzero((speed(track) == 0.0) & (np.abs(track.tau - tau) <= tol + 1e-9))[0]
    if not still.size:
        return None
    run_end = still[0] + int(np.argmax(np.diff(np.append(still, still[-1] + 2)) > 1))
    return still[0] * DT, (run_end + 1) * DT


def joint_speed(track):
    """Largest joint speed (rad/s) at each sample."""
    return np.abs(np.gradient(track.q, DT, axis=0)).max(axis=1)


def quiet_time(track, t_lo, t_hi):
    """The time in [t_lo, t_hi] where the joints move slowest: a good place for a hold."""
    i0, i1 = int(round(t_lo / DT)), int(round(t_hi / DT)) + 1
    return (i0 + int(np.argmin(joint_speed(track)[i0:i1]))) * DT


def hold_ramp_s(track, at, accel_cap, min_ramp_s):
    """A hold ramp long enough that easing to a stop at `at` adds at most accel_cap (rad/s^2):
    stopping a joint moving at v over a raised-cosine ramp R adds up to v * pi / (2 R)."""
    v = joint_speed(track)[min(int(round(at / DT)), track.n - 1)]
    return max(min_ramp_s, float(v) * np.pi / (2.0 * accel_cap))


def scripted_track(q, grip, zone=FREE, name="move"):
    """A functional track from joint rows q (N, 7): one phase, nothing held."""
    q = np.asarray(q, dtype=float)
    n = len(q)
    return Track(
        q=q,
        grip=np.broadcast_to(np.asarray(grip, dtype=float), (n,)).copy(),
        zone=np.full(n, zone, dtype=int),
        held=np.zeros(n, dtype=bool),
        tau=np.arange(n) * DT,
        phases=[Label(name, 0.0, n * DT)],
        name=name,
    )


def constant_track(q, grip, duration_s, zone=FREE, name="rest"):
    """A functional track standing still at q for duration_s."""
    n = int(round(duration_s / DT)) + 1
    return scripted_track(np.tile(np.asarray(q, dtype=float), (n, 1)), grip, zone, name)


def joint_move(q0, q1, duration_s, grip=0.0, zone=FREE, name="move"):
    """A functional min-jerk joint move from q0 to q1 (at rest at both ends)."""
    s = curves.min_jerk(np.linspace(0.0, 1.0, int(round(duration_s / DT)) + 1))
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    return scripted_track(q0 + s[:, None] * (q1 - q0), grip, zone, name)


def concat(tracks, name=""):
    """Functional tracks one after another; each must start where the previous one ended (the
    shared sample appears once). Labels shift with their track."""
    parts, phases, beats, offset = [], [], [], 0.0
    for k, tr in enumerate(tracks):
        if k and not np.allclose(tr.q[0], tracks[k - 1].q[-1], atol=1e-6):
            raise ValueError(f"track {k} ({tr.name}) does not start where track {k - 1} ended")
        first = 0 if k == 0 else 1
        parts.append((tr, first))
        phases += [Label(p.name, p.t0 + offset, p.t1 + offset) for p in tr.phases]
        beats += [Label(b.name, b.t0 + offset, b.t1 + offset) for b in tr.beats]
        offset += tr.duration
    n = sum(tr.n - first for tr, first in parts)
    return Track(
        q=np.concatenate([tr.q[first:] for tr, first in parts]),
        grip=np.concatenate([tr.grip[first:] for tr, first in parts]),
        zone=np.concatenate([tr.zone[first:] for tr, first in parts]),
        held=np.concatenate([tr.held[first:] for tr, first in parts]),
        tau=np.arange(n) * DT,
        phases=phases,
        beats=beats,
        name=name,
    )
