"""Head offsets, look-at and the guarded distance."""

import mujoco
import numpy as np
import pytest

from puppy_sim import kin as K


def test_head_offset_is_realised_by_the_iterated_dls(kin, home):
    offset = np.array([0.01, -0.005, 0.008, np.radians(15), np.radians(-10), np.radians(20)])
    p_t, R_t = kin.head_target(*kin.fk(home, "ee"), offset)
    p, R = kin.fk(home + kin.head_dq(home, offset), "ee")
    assert np.linalg.norm(p - p_t) < 1e-3
    assert np.degrees(np.linalg.norm(K.rot_err(R_t, R))) < 0.5


def test_zero_offset_gives_exactly_zero(kin, home):
    assert np.array_equal(kin.head_dq(home, np.zeros(6)), np.zeros(7))


def test_head_turns_stay_in_the_wrist(kin, home):
    dq = kin.head_dq(home, np.array([0, 0, 0, 0.2, 0.2, -0.3]))
    assert np.abs(dq[:4]).max() < 0.1 * np.abs(dq[4:]).max()


def test_look_at_puts_the_target_on_the_gaze(kin, home, sc):
    target = sc.placement.center
    offset = kin.look_at_offset(home, target)
    q = home + kin.head_dq(home, offset)
    eye, R = kin.fk(q, "eye")
    d = (target - eye) / np.linalg.norm(target - eye)
    assert np.degrees(np.arccos(R[:, 2] @ d)) < 1.0
    capped = kin.look_at_offset(home, sc.person_head, max_turn_rad=np.radians(18))
    assert np.degrees(np.arccos(np.cos(capped[4]) * np.cos(capped[5]))) == pytest.approx(18.0, abs=0.5)


def _vertex_box_distance(kin, mesh_geom, box_geom):
    """Upper bound on a mesh-box distance: the nearest mesh vertex to the box."""
    m, d = kin.m, kin.d
    mid = m.geom_dataid[mesh_geom]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    world = d.geom_xpos[mesh_geom] + v @ d.geom_xmat[mesh_geom].reshape(3, 3).T
    R = d.geom_xmat[box_geom].reshape(3, 3)
    local = (world - d.geom_xpos[box_geom]) @ R
    return float(np.linalg.norm(np.maximum(np.abs(local) - m.geom_size[box_geom], 0.0), axis=1).min())


def test_guarded_distance_never_reports_far_pairs_as_touching(kin, sc, run):
    boxes = [g for g in kin.obstacle_geoms(False) if kin.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    meshes = [g for part in K.ARM_PARTS for g in sc.groups[part] if kin.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    rng = np.random.default_rng(3)
    fromto = np.zeros(6)
    for _ in range(25):
        kin.pose(run.q[rng.integers(0, len(run.q))] + rng.normal(0.0, 0.05, 7), 0.5)
        for a in meshes:
            for b in boxes:
                guarded = kin.geom_distance(int(a), int(b), K.CLEARANCE_CUTOFF_M, fromto)
                upper = _vertex_box_distance(kin, a, b)
                assert guarded <= upper + 1e-6
                assert guarded > 0.0 or upper < 0.01


def test_cartesian_path_keeps_the_tool_on_a_straight_line(kin, home):
    p0, R0 = kin.fk(home, "tool")
    p1 = p0 + np.array([0.08, -0.05, -0.04])
    q, worst = kin.cartesian_path(home, p1, R0, 1.5)
    assert worst < 1e-4
    line = (p1 - p0) / np.linalg.norm(p1 - p0)
    for qi in q[:: len(q) // 10]:
        p, R = kin.fk(qi, "tool")
        off = (p - p0) - ((p - p0) @ line) * line
        assert np.linalg.norm(off) < 1e-3
        assert np.degrees(np.linalg.norm(K.rot_err(R0, R))) < 0.1


def test_a_gravity_tilt_lowers_the_gaze_even_with_the_face_on_its_side(kin, functional):
    q = functional.q[int(round(5.3 / 0.02))]  # the view posture: the face's up is world -y
    _, R = kin.fk(q, "ee")
    assert abs(R[1, 1]) > 0.9
    tilt = np.radians(10.0)
    offset = np.concatenate([np.zeros(3), kin.gravity_turn(q, 0.0, tilt)])
    _, R1 = kin.fk(q + kin.head_dq(q, offset), "ee")
    el0, az0 = K.elevation_azimuth(R[:, 2])
    el1, az1 = K.elevation_azimuth(R1[:, 2])
    assert np.degrees(el0 - el1) == pytest.approx(10.0, abs=0.5)
    assert abs(np.degrees(K.wrap(az1 - az0))) < 0.5


def test_aim_and_the_look_at_modes(kin, home, sc):
    target = sc.placement.center
    for mode in ("shortest", "pan_tilt"):
        offset = kin.look_at_offset(home, target, mode=mode)
        eye, R = kin.fk(home + kin.head_dq(home, offset), "eye")
        d = (target - eye) / np.linalg.norm(target - eye)
        assert np.degrees(np.arccos(R[:, 2] @ d)) < 1.0, mode
    level = kin.look_at_offset(home, sc.person_head, max_turn_rad=np.radians(15), mode="pan")
    _, R0 = kin.fk(home, "ee")
    _, R1 = kin.fk(home + kin.head_dq(home, level), "ee")
    up = R1 @ R0.T @ np.array([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(up[2])) < 0.5  # a pan keeps the vertical vertical
    assert np.degrees(np.arccos(R0[:, 2] @ R1[:, 2])) == pytest.approx(15.0, abs=0.5)
