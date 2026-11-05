"""Puppy Mode stays inside the envelope RAMMP's own task motions already use: the envelope is
measured from the recorded runs, and the FREE caps are held to it."""

import numpy as np
import pytest

from puppy_sim import envelope, layer
from puppy_sim.timeline import FREE

# what the lead measured (FREE samples while moving, smoothed as runs.py does)
LEAD_SPEED_MAX = (0.36, 1.12, 0.60, 0.65, 0.98, 0.87, 0.98)
LEAD_ACCEL_MAX = (0.84, 3.22, 1.40, 1.44, 2.18, 1.90, 2.08)


@pytest.fixture(scope="module")
def measured():
    return envelope.measured()


def test_the_envelope_is_the_one_measured_from_the_runs(measured):
    assert measured.runs == 3
    np.testing.assert_allclose(measured.joint_speed_max, LEAD_SPEED_MAX, atol=0.006)
    np.testing.assert_allclose(measured.joint_accel_max, LEAD_ACCEL_MAX, atol=0.006)
    assert measured.eye_speed_max == pytest.approx(0.45, abs=0.005)
    assert measured.tool_speed_max == pytest.approx(0.40, abs=0.005)


def test_free_caps_stay_inside_the_tasks_own_envelope(measured):
    """No expressive joint faster than the task itself moves that joint, the head no faster than
    the task moves its tool point, and the acceleration cap within the task's own peak."""
    free = layer.CAP_PRESETS["default"][FREE]
    assert np.all(np.array(free.joint_speed) <= measured.joint_speed_max)
    assert np.all(np.array(layer.TASK_JOINT_SPEED) <= measured.joint_speed_max)
    assert free.head_speed <= measured.tool_speed_max
    assert max(free.joint_accel) <= measured.joint_accel_max.max()


def test_every_preset_is_gentler_near_the_user():
    for preset in layer.CAP_PRESETS.values():
        free, near = preset[FREE], preset[1]
        assert near.head_speed < free.head_speed
        assert np.all(np.array(near.joint_speed) <= np.array(free.joint_speed))
        assert np.all(np.array(near.joint_accel) < np.array(free.joint_accel))


def test_the_summary_names_the_envelope_and_the_caps():
    task, caps = envelope.summary()
    assert "0.40 m/s" in task and "FREE" in caps and "NEAR_USER" in caps
