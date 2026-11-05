"""Smooth 1-D curves for motion design: easing, envelopes, springs, noise, oscillators.

Everything takes and returns numpy arrays (or floats) and is pure. Time arguments are seconds on
the shared 50 Hz clock (timeline.DT), but nothing here depends on it.
"""

import numpy as np

MIN_JERK_PEAK_SLOPE = 1.875  # max of d/du min_jerk(u)
MIN_JERK_PEAK_CURVE = 10.0 / np.sqrt(3.0)  # max of |d2/du2 min_jerk(u)|
BUMP_PEAK_SLOPE = 2.0 * MIN_JERK_PEAK_SLOPE  # max of d/du bump(u)
BUMP_PEAK_CURVE = 4.0 * MIN_JERK_PEAK_CURVE  # max of |d2/du2 bump(u)|
NOISE_KNOTS_PER_CYCLE = 1.0  # random knots per period of the noise's corner frequency
WOBBLE_FADE_SHARE = 0.35  # a wobble fades to rest over this last share of its settle time
WOBBLE_KICK_PERIODS = 0.5  # a wobble's spring is kicked by a bump this many natural periods long
SINE_EASE_PERIODS = 1.0  # a windowed sine eases in and out over this many periods (its peak acceleration:
# 1.11 x a pure sine's; over half a period it would be 1.58 x)


def min_jerk(u):
    """0..1 -> 0..1 with zero velocity and acceleration at both ends (clipped outside)."""
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def raised_cosine(u):
    """0..1 -> 0..1 half-cosine: zero velocity at both ends (clipped outside)."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def ramp(t, t0, t1, shape=min_jerk):
    """0 before t0, 1 after t1, `shape` in between."""
    if t1 <= t0:
        return (np.asarray(t) >= t0).astype(float)
    return shape((np.asarray(t, dtype=float) - t0) / (t1 - t0))


def envelope(t, t_on, t_off, rise_s, fall_s, shape=min_jerk):
    """Rises over [t_on, t_on + rise_s], holds 1, falls over [t_off, t_off + fall_s]."""
    return ramp(t, t_on, t_on + rise_s, shape) - ramp(t, t_off, t_off + fall_s, shape)


def overshoot_peak_at(overshoot):
    """Where overshoot_ease peaks: the split at which its rise and its settle need the same peak
    acceleration, which keeps the whole move's acceleration lowest."""
    return 1.0 / (1.0 + np.sqrt(overshoot / (1.0 + overshoot)))


def overshoot_ease(u, overshoot):
    """0..1 -> 0..1 passing 1 + overshoot, then settling back: two min-jerk pieces joined at the
    peak, so position, velocity and acceleration stay continuous and both ends are at rest."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if overshoot <= 0.0:
        return min_jerk(u)
    p = overshoot_peak_at(overshoot)
    rise = (1.0 + overshoot) * min_jerk(u / p)
    settle = 1.0 + overshoot * (1.0 - min_jerk((u - p) / (1.0 - p)))
    return np.where(u <= p, rise, settle)


def ease_peaks(overshoot):
    """(peak |slope|, peak |curvature|) of overshoot_ease: a move of amplitude A over T seconds
    peaks at slope * A / T (speed) and curvature * A / T^2 (acceleration)."""
    p = overshoot_peak_at(overshoot)
    return MIN_JERK_PEAK_SLOPE * (1.0 + overshoot) / p, MIN_JERK_PEAK_CURVE * (1.0 + overshoot) / p**2


def paced_duration(amplitude, overshoot, speed_cap, accel_cap):
    """Shortest duration for an overshoot_ease move of `amplitude` to stay under both caps."""
    slope, curve = ease_peaks(overshoot)
    return max(slope * amplitude / speed_cap, float(np.sqrt(curve * amplitude / accel_cap)))


def overshoot_envelope(t, t_on, t_off, turn_s, overshoot):
    """Envelope whose rise overshoots past 1 and whose fall undershoots past 0 before settling."""
    t = np.asarray(t, dtype=float)
    return overshoot_ease((t - t_on) / turn_s, overshoot) - overshoot_ease((t - t_off) / turn_s, overshoot)


def spring_follow(target, dt, freq_hz, damping):
    """Second-order follower of `target` (starts at rest on target[0]); damping < 1 overshoots."""
    target = np.asarray(target, dtype=float)
    w = 2.0 * np.pi * freq_hz
    x = np.empty_like(target)
    x[0], v = target[0], np.zeros_like(target[0])
    for i in range(1, len(target)):
        a = w * w * (target[i] - x[i - 1]) - 2.0 * damping * w * v
        v = v + a * dt
        x[i] = x[i - 1] + v * dt
    return x


def smooth_noise(n, dt, corner_hz, seed):
    """n samples of band-limited noise in [-1, 1]: min-jerk interpolation of random knots."""
    rng = np.random.default_rng(seed)
    knot_dt = 1.0 / (corner_hz * NOISE_KNOTS_PER_CYCLE)
    n_knots = int(np.ceil(n * dt / knot_dt)) + 2
    knots = rng.uniform(-1.0, 1.0, n_knots)
    s = np.arange(n) * dt / knot_dt
    i = np.floor(s).astype(int)
    return knots[i] + (knots[i + 1] - knots[i]) * min_jerk(s - i)


def oscillate(t, period_s, phase=0.0):
    """sin(2 pi t / period + phase)."""
    return np.sin(2.0 * np.pi * np.asarray(t, dtype=float) / period_s + phase)


def bump(u):
    """0 -> 1 -> 0 over u in [0, 1]: two min-jerk halves, at rest at both ends and at the top."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return np.where(u <= 0.5, min_jerk(2.0 * u), min_jerk(2.0 - 2.0 * u))


def bumps(t, t0, count, period_s):
    """`count` bumps back to back from t0, each period_s long; 0 elsewhere."""
    u = (np.asarray(t, dtype=float) - t0) / period_s
    return np.where((u >= 0.0) & (u < count), bump(u - np.floor(u)), 0.0)


def sine_envelope(t, t0, cycles, freq_hz, decay_s=np.inf):
    """windowed_sine's envelope: eased in over the first SINE_EASE_PERIODS from t0 and out over the
    last of `cycles` periods, decaying as exp(-(t - t0) / decay_s)."""
    tau = np.asarray(t, dtype=float) - t0
    ease = min(SINE_EASE_PERIODS, cycles / 2.0) / freq_hz
    end = cycles / freq_hz
    return (ramp(tau, 0.0, ease) - ramp(tau, end - ease, end)) * np.exp(-np.maximum(tau, 0.0) / decay_s)


def windowed_sine(t, t0, cycles, freq_hz, decay_s=np.inf):
    """sin over `cycles` periods from t0 under sine_envelope (so it starts and ends at rest)."""
    tau = np.asarray(t, dtype=float) - t0
    return np.sin(2.0 * np.pi * freq_hz * tau) * sine_envelope(t, t0, cycles, freq_hz, decay_s)


def wobble(t, t0, freq_hz, damping, settle_s):
    """The overshoot-and-settle of a follow-through, on the uniform grid t: a spring (freq_hz,
    damping) kicked from rest at t0 by a smooth bump half a period long. It rises to +1 (its first
    peak), swings back past 0 and dies away, eased to exactly 0 by t0 + settle_s. Smooth (C2), its
    curvature near (2 pi freq_hz)^2."""
    t = np.asarray(t, dtype=float)
    kick = bump((t - t0) * freq_hz / WOBBLE_KICK_PERIODS)
    x = spring_follow(kick, t[1] - t[0], freq_hz, damping)
    fade = WOBBLE_FADE_SHARE * settle_s
    x = x * (1.0 - ramp(t, t0 + settle_s - fade, t0 + settle_s))
    peak = x.max()
    return x / peak if peak > 0.0 else x
