import numpy as np

from puppy_motion.joints import align_turns, continuous, shortest_delta
from puppy_motion.stage import points_from_q


def test_continuous_joints_are_1_3_5_7(chain):
    assert continuous(chain).tolist() == [True, False, True, False, True, False, True]


def test_shortest_delta_wraps_continuous_joints_only(chain):
    a = np.array([3.1, 1.0, -3.1, 1.0, 0.0, 1.0, 6.0])
    b = np.array([-3.1, 1.0 + 2 * np.pi - 0.1, 3.1, 1.0 + 2 * np.pi, 0.5, 1.0 - 2 * np.pi, 6.0 + 2 * np.pi])
    d = shortest_delta(chain, a, b)
    assert np.allclose(d[[0, 2]], [2 * np.pi - 6.2, 6.2 - 2 * np.pi])
    assert np.allclose(d[[1, 3, 5]], [2 * np.pi - 0.1, 2 * np.pi, -2 * np.pi])  # revolute: plain
    assert np.allclose(d[[4, 6]], [0.5, 0.0], atol=1e-12)


def test_the_range_is_minus_pi_exclusive_to_pi_inclusive(chain):
    z = np.zeros(7)
    assert shortest_delta(chain, z, np.full(7, np.pi))[0] == np.pi
    assert shortest_delta(chain, z, np.full(7, -np.pi))[0] == np.pi
    assert shortest_delta(chain, z, np.full(7, 3 * np.pi))[0] == np.pi


def test_align_turns_shifts_every_point_the_same_way(chain):
    q = np.array([[-5.5, 0.3, -5.0, 0.2, 4.6, 0.1, 4.5], [-5.0, 0.4, -4.5, 0.2, 4.7, 0.1, 4.0]])
    pts = points_from_q(np.vstack([q, q[-1]]))
    ref = np.array([0.78, 0.3, 1.28, 0.2, -1.68, 0.1, -1.78])
    out = align_turns(chain, ref, pts)
    assert np.all(np.abs(out[0][1] - ref)[continuous(chain)] <= np.pi)
    assert np.allclose(out[0][1][1::2], pts[0][1][1::2])  # revolute joints untouched
    for (t, q0, dq0), (t1, q1, dq1) in zip(pts, out):
        assert t == t1 and np.array_equal(dq0, dq1)
        k = (q1 - q0) / (2 * np.pi)
        assert np.allclose(k, np.round(k)) and np.allclose(k, (out[0][1] - pts[0][1]) / (2 * np.pi))


def test_align_turns_leaves_an_aligned_path_alone(chain):
    pts = points_from_q(np.zeros((3, 7)))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(align_turns(chain, np.zeros(7), pts), pts))
