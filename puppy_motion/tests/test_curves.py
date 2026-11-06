import numpy as np
import pytest

from puppy_motion import curves

U = np.linspace(0.0, 1.0, 20001)


def _slopes(f):
    d1 = np.gradient(f, U)
    return d1, np.gradient(d1, U)


def test_min_jerk_starts_and_ends_at_rest():
    f = curves.min_jerk(U)
    d1, d2 = _slopes(f)
    assert f[0] == 0.0 and f[-1] == 1.0
    assert abs(d1[0]) < 1e-3 and abs(d1[-1]) < 1e-3 and abs(d2[1]) < 1e-2
    assert d1.max() == pytest.approx(curves.MIN_JERK_PEAK_SLOPE, rel=1e-3)


@pytest.mark.parametrize("overshoot", [0.05, 0.08, 0.12])
def test_overshoot_ease_peaks_and_settles_smoothly(overshoot):
    f = curves.overshoot_ease(U, overshoot)
    d1, d2 = _slopes(f)
    assert f.max() == pytest.approx(1.0 + overshoot, abs=1e-6)
    assert f[0] == 0.0 and f[-1] == pytest.approx(1.0)
    assert abs(d1[-1]) < 1e-3
    slope, curve = curves.ease_peaks(overshoot)
    assert np.abs(d1).max() == pytest.approx(slope, rel=2e-3)
    assert np.abs(d2).max() == pytest.approx(curve, rel=2e-2)


def test_paced_duration_keeps_a_move_under_both_caps():
    amplitude, speed_cap, accel_cap = 0.6, 0.7, 2.5
    T = curves.paced_duration(amplitude, 0.08, speed_cap, accel_cap)
    t = np.linspace(0.0, T, 4001)
    x = amplitude * curves.overshoot_ease(t / T, 0.08)
    v = np.gradient(x, t)
    assert np.abs(v).max() <= speed_cap * 1.001
    assert np.abs(np.gradient(v, t)).max() <= accel_cap * 1.01


def test_envelopes_are_zero_outside_and_one_inside():
    t = np.linspace(0, 10, 1001)
    e = curves.envelope(t, 2.0, 6.0, 1.0, 1.5)
    assert e[t < 2.0].max() == 0.0 and e[t > 7.5].max() == 0.0
    assert np.all(e[(t >= 3.0) & (t <= 6.0)] == 1.0)
    o = curves.overshoot_envelope(t, 2.0, 6.0, 1.0, 0.1)
    assert o.max() == pytest.approx(1.1, abs=1e-3) and o.min() == pytest.approx(-0.1, abs=1e-3)
    assert o[-1] == 0.0


def test_noise_and_springs():
    a = curves.smooth_noise(500, 0.02, 0.5, seed=3)
    assert np.abs(a).max() <= 1.0 and np.array_equal(a, curves.smooth_noise(500, 0.02, 0.5, seed=3))
    step = np.r_[np.zeros(20), np.ones(400)]
    x = curves.spring_follow(step, 0.02, 1.5, 0.4)
    assert x.max() > 1.05 and x[-1] == pytest.approx(1.0, abs=0.01)
    assert curves.oscillate(0.25, 1.0) == pytest.approx(1.0)


def test_bumps_rise_and_fall_at_rest():
    t = np.linspace(0.0, 3.0, 3001)
    b = curves.bumps(t, 0.5, 3, 0.6)
    assert b.max() == pytest.approx(1.0, abs=1e-6) and b.min() == 0.0
    assert b[t < 0.5].max() == 0.0 and b[t > 2.3].max() == 0.0
    d1 = np.gradient(b, t)
    assert np.abs(d1).max() == pytest.approx(curves.BUMP_PEAK_SLOPE / 0.6, rel=1e-2)
    assert abs(d1[np.searchsorted(t, 1.1)]) < 0.05  # at rest where one bump hands over to the next


def test_windowed_sine_starts_and_ends_at_rest():
    t = np.linspace(0.0, 3.0, 6001)
    w = curves.windowed_sine(t, 0.5, 3, 2.0, decay_s=0.8)
    d1 = np.gradient(w, t)
    assert w[t <= 0.5].max() == 0.0 and np.abs(w[t >= 2.0]).max() < 1e-12
    assert abs(d1[np.searchsorted(t, 0.5)]) < 1e-2 and abs(d1[np.searchsorted(t, 2.0) - 1]) < 1e-1


def test_wobble_overshoots_settles_and_stays_smooth():
    dt = 0.02
    t = np.arange(0.0, 4.0, dt)
    w = curves.wobble(t, 0.5, 0.8, 0.4, 1.8)
    d1 = np.gradient(w, dt)
    d2 = np.gradient(d1, dt)
    assert w.max() == pytest.approx(1.0) and -0.4 < w.min() < -0.1  # one swing back past 0
    assert np.all(w[t <= 0.5] == 0.0) and np.all(w[t >= 2.3] == 0.0)
    assert np.abs(d2).max() < 1.5 * (2 * np.pi * 0.8) ** 2  # curvature near the spring's own
