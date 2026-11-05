import glob
import json
import os

import numpy as np
import pytest

from puppy_motion.urdf import Chain

DATA = os.path.join(os.path.dirname(__file__), "data")
RUNS = sorted(glob.glob(os.path.join(DATA, "*.json")))
SITES = ("ee", "tool", "eye", "neck")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _quat_to_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    x, y, z, w = (c / np.sqrt(n) for c in (x, y, z, w))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture(scope="module")
def chain():
    return Chain.from_urdf(_load(RUNS[0])["urdf"])


def test_there_are_three_recorded_runs():
    assert len(RUNS) == 3


def test_fk_matches_the_recorded_end_effector():
    max_pos_err_mm = 0.0
    max_rot_err_deg = 0.0
    for path in RUNS:
        run = _load(path)
        ch = Chain.from_urdf(run["urdf"])
        t = np.asarray(run["t"], dtype=float)
        q = np.asarray(run["q"], dtype=float)[:, :7]
        for stamp, msg in run["ee"]:
            qi = np.array([np.interp(stamp, t, q[:, j]) for j in range(7)])
            p, R = ch.fk(qi, "ee")
            pose = msg["pose"]
            pr = pose["position"]
            o = pose["orientation"]
            p_ref = np.array([pr["x"], pr["y"], pr["z"]])
            R_ref = _quat_to_matrix(o["x"], o["y"], o["z"], o["w"])
            max_pos_err_mm = max(max_pos_err_mm, np.linalg.norm(p - p_ref) * 1000)
            cos = (np.trace(R_ref.T @ R) - 1) / 2
            max_rot_err_deg = max(max_rot_err_deg, np.degrees(np.arccos(np.clip(cos, -1, 1))))
    assert max_pos_err_mm < 0.1 and max_rot_err_deg < 0.02


def test_tool_is_120_mm_along_ee_z(chain):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-2, 2, 7)
        p_ee, R_ee = chain.fk(q, "ee")
        p_tool, _ = chain.fk(q, "tool")
        assert np.linalg.norm(p_tool - (p_ee + R_ee[:, 2] * 0.120)) < 1e-9


def test_eye_and_neck_offsets_in_the_ee_frame(chain):
    q = np.random.default_rng(1).uniform(-2, 2, 7)
    p_ee, R_ee = chain.fk(q, "ee")
    assert np.allclose(chain.fk(q, "eye")[0], p_ee + R_ee @ [0, 0.065, 0.015], atol=1e-12)
    assert np.allclose(chain.fk(q, "neck")[0], p_ee - R_ee[:, 2] * 0.1675, atol=1e-12)


def test_jacobian_matches_finite_differences(chain):
    rng = np.random.default_rng(2)
    eps = 1e-6
    for _ in range(20):
        q = rng.uniform(-2, 2, 7)
        for site in SITES:
            J = chain.jacobian(q, site)
            assert J.shape == (6, 7)
            p0, R0 = chain.fk(q, site)
            J_fd = np.zeros((6, 7))
            for j in range(7):
                dq = q.copy()
                dq[j] += eps
                p1, R1 = chain.fk(dq, site)
                J_fd[:3, j] = (p1 - p0) / eps
                # world-frame angular velocity from R1 R0^T = exp([w] eps)
                dR = (R1 @ R0.T - np.eye(3)) / eps
                J_fd[3:, j] = [dR[2, 1], dR[0, 2], dR[1, 0]]
            assert np.abs(J - J_fd).max() < 1e-5


def test_guards_and_limits_from_the_urdf(chain):
    assert chain.joint_names == tuple(f"arm_joint_{i}" for i in range(1, 8))
    assert chain.vel_limits.tolist() == [1.3963] * 4 + [1.2218] * 3
    assert chain.pos_limits.shape == (7, 2)
    inf = np.inf
    assert chain.pos_limits.tolist() == [
        [-inf, inf], [-2.24, 2.24], [-inf, inf], [-2.57, 2.57],
        [-inf, inf], [-2.09, 2.09], [-inf, inf],
    ]


def test_unknown_site_is_rejected(chain):
    with pytest.raises(ValueError):
        chain.fk(np.zeros(7), "nose")
