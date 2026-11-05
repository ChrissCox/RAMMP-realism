"""The time-warp changes timing only: the path is the functional path at tau, and dtau/dt in [0, 1]."""

import numpy as np
import pytest

from puppy_sim import timeline as tl
from puppy_sim.timeline import DT, FREE, PRECISION


def _warp_events(track):
    holds = [tl.Hold(at=5.3, hold_s=1.5, ramp_s=0.4), tl.Hold(at=12.5, hold_s=0.8, ramp_s=0.5)]  # 5.3: in detect
    slows = [tl.Slow(t0=7.0, t1=9.0, factor=0.5, ramp_s=0.3)]
    return holds, slows


def test_warped_track_is_the_functional_path_at_tau(functional):
    holds, slows = _warp_events(functional)
    warped = tl.warp(functional, holds, slows)
    u = tl.warp_indices(functional.n, holds, slows)
    assert warped.q.tobytes() == tl.interp_rows(functional.q, u).tobytes()  # exactly the path at tau
    np.testing.assert_allclose(warped.tau, u * DT, atol=1e-12)
    assert warped.n > functional.n  # the holds add time
    assert warped.tau[0] == 0.0 and warped.tau[-1] == functional.tau[-1]


def test_speed_stays_within_zero_and_one(functional):
    holds, slows = _warp_events(functional)
    s = tl.speed(tl.warp(functional, holds, slows))
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-12
    assert np.isclose(s, 0.0).sum() >= int(round(1.5 / DT))  # the long hold stands still
    assert np.isclose(s, 0.5).sum() > 10  # the slow-in cruises at half speed


def test_a_full_hold_lands_back_on_functional_samples(functional):
    holds = [tl.Hold(at=5.3, hold_s=1.0, ramp_s=0.4)]
    warped = tl.warp(functional, holds)
    u = tl.warp_indices(functional.n, holds)
    after = u >= int(round(5.3 / DT)) + 20  # well after the hold
    assert np.all(u[after] == np.round(u[after]))
    idx = u[after].astype(int)
    assert warped.q[after].tobytes() == functional.q[idx].tobytes()
    assert warped.tau[after].tobytes() == functional.tau[idx].tobytes()


def test_hold_comes_to_rest_at_its_time(functional):
    warped = tl.warp(functional, [tl.Hold(at=5.3, hold_s=1.0, ramp_s=0.4)])
    still = tl.speed(warped) == 0.0
    assert still.any()
    assert warped.tau[np.argmax(still)] == pytest.approx(5.3, abs=DT)


def test_resampled_zone_takes_the_stricter_neighbour():
    track = tl.constant_track(np.zeros(7), 0.0, 0.1)
    track.zone[3:] = PRECISION
    out = tl.resample_track(track, np.array([2.0, 2.5, 3.0]))
    assert list(out.zone) == [FREE, PRECISION, PRECISION]


def test_labels_move_with_the_warp(functional):
    hold_s = 1.0
    warped = tl.warp(functional, [tl.Hold(at=5.3, hold_s=hold_s, ramp_s=0.4)])
    before = {p.name: p.t0 for p in functional.phases}
    after = {p.name: p.t0 for p in warped.phases}
    assert after["home"] == pytest.approx(before["home"], abs=DT)
    assert after["approach"] - before["approach"] == pytest.approx(hold_s + 0.4, abs=2 * DT)


def test_hold_ramp_bounds_the_added_acceleration(functional):
    cap = 3.0
    at = 4.0  # mid-'view': the arm is moving
    ramp = tl.hold_ramp_s(functional, at, cap, min_ramp_s=0.1)
    v = tl.joint_speed(functional)[int(round(at / DT))]
    assert v * np.pi / (2 * ramp) <= cap + 1e-9


def test_scripted_moves_chain_into_one_functional_track(home):
    q1 = home + np.radians([10, -5, 0, 8, 0, -6, 12])
    a = tl.joint_move(home, q1, 1.0, name="out")
    b = tl.constant_track(q1, 0.0, 0.5, name="wait")
    c = tl.joint_move(q1, home, 1.0, name="back")
    seq = tl.concat([a, b, c], name="seq")
    assert seq.n == a.n + b.n + c.n - 2
    assert np.array_equal(seq.q[0], home) and np.allclose(seq.q[-1], home)
    assert [p.name for p in seq.phases] == ["out", "wait", "back"]
    assert seq.phases[2].t0 == pytest.approx(1.5)
    v = np.gradient(a.q, DT, axis=0)
    assert np.abs(v[0]).max() < 1e-2 and np.abs(v[-1]).max() < 1e-2  # min-jerk: at rest at both ends
    with pytest.raises(ValueError):
        tl.concat([a, a])


def test_a_slow_that_stops_first_stares_then_creeps(functional):
    """Lock-on: stop at t0, stand still, ease up to the factor (no bump to full speed), cruise."""
    slow = tl.Slow(t0=8.84, t1=10.4, factor=0.75, ramp_s=0.4, hold_s=1.2)
    warped = tl.warp(functional, slows=[slow])
    s = tl.speed(warped)
    assert s.max() <= 1.0 + 1e-12 and s.min() >= 0.0
    span = tl.still_span(warped, 8.84)
    assert span is not None and span[1] - span[0] == pytest.approx(1.2, abs=DT)
    after = int(round(span[1] / DT))
    assert s[after : after + 40].max() <= 0.75 + 1e-12  # creeps in, never surges back to 1
    assert tl.output_time(warped, 8.84) == pytest.approx(span[0])


def test_a_hold_stands_still_exactly_on_its_sample(functional):
    warped = tl.warp(functional, [tl.Hold(at=4.70, hold_s=1.0, ramp_s=0.3)])
    span = tl.still_span(warped, 4.70)
    i = int(round(span[0] / DT))
    assert warped.tau[i] == pytest.approx(4.70, abs=1e-12)  # an even ramp lands on the sample
    assert warped.q[i].tobytes() == functional.q[int(round(4.70 / DT))].tobytes()
