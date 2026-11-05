"""The model is the robot: menagerie joint_k == arm_joint_k (same zeros), base tilted 30 deg, and
/ee_state is arm_end_effector_link in arm_mount."""

import numpy as np
import pytest

from puppy_sim import checks, runs

RUN_NAMES = sorted(p.stem for p in runs.RUNS_DIR.glob("*.json"))


@pytest.mark.parametrize("name", RUN_NAMES)
def test_fk_matches_recorded_end_effector(kin, name):
    pos_max, pos_mean, rot_max = checks.fk_residuals(runs.load_run(name), kin)
    assert pos_max < 0.5e-3
    assert pos_mean < 0.1e-3
    assert rot_max < 0.05


def test_tool_frame_is_120_mm_along_end_effector_z(kin, home):
    p_ee, R_ee = kin.fk(home, "ee")
    p_tool, R_tool = kin.fk(home, "tool")
    np.testing.assert_allclose(p_tool, p_ee + 0.120 * R_ee[:, 2], atol=1e-9)
    np.testing.assert_allclose(R_tool, R_ee, atol=1e-9)


def test_neck_is_the_wrist_centre(kin):
    """Joints 5-7 meet 0.167 m behind the end effector on its z axis: head turns pivot there."""
    assert np.linalg.norm(kin.neck_local[:2]) < 0.5e-3
    assert kin.neck_local[2] == pytest.approx(-0.1675, abs=1e-3)
