"""The scene: the container sits where the real run closed the gripper, the brim between the
fingertips; it is carried once held; the button is reachable; the person is clear of the run."""

import mujoco
import numpy as np
import pytest

from puppy_sim import runs, scene
from puppy_sim.kin import GUARD_LIMITS


def test_tool_frame_at_closing_sits_at_the_brim(sc, run, kin):
    pl = sc.placement
    p_tool, R_tool = kin.fk(run.q[runs.closing_index(run)], "tool")
    np.testing.assert_allclose(pl.tool_pos, p_tool, atol=1e-9)
    edge_offset = pl.brim_edge - p_tool  # GRASP_DEPTH_M = -8 mm: the tool point stops short of the edge
    assert np.linalg.norm(edge_offset) == pytest.approx(-scene.GRASP_DEPTH_M, abs=1e-6)
    assert edge_offset @ R_tool[:, 2] == pytest.approx(-scene.GRASP_DEPTH_M, abs=1e-6)
    radial = np.hypot(*(pl.brim_edge - pl.center)[:2])
    assert radial == pytest.approx(scene.BRIM_RADIUS_M, abs=1e-6)
    assert pl.brim_edge[2] == pytest.approx(pl.center[2])
    assert pl.table_top_z == pytest.approx(pl.center[2] - scene.CONTAINER_HEIGHT_M)
    assert np.degrees(np.arccos(-R_tool[2, 2])) == pytest.approx(75.0, abs=0.5)  # the side approach


def test_brim_is_between_the_fingertips_when_closed(sc, run, kin):
    i = runs.closing_index(run) + 40  # settled on the brim
    kin.pose(run.q[i], run.grip[i], held=True)
    m = sc.model
    pads = {
        side: np.mean([kin.d.geom_xpos[m.geom(f"g_{side}_pad{k}").id] for k in (1, 2)], axis=0)
        for side in ("left", "right")
    }
    heights = sorted(p[2] for p in pads.values())
    brim_top = sc.placement.center[2]
    assert heights[0] < brim_top - scene.BRIM_THICK_M / 2 < heights[1]  # one pad above, one below
    for p in pads.values():
        radial = np.hypot(*(p - sc.placement.center)[:2])
        assert scene.BRIM_RADIUS_M - scene.BRIM_WIDTH_M - 0.02 < radial < scene.BRIM_RADIUS_M + 0.005


def test_container_rides_rigidly_once_held(functional, kin, sc):
    held = np.flatnonzero(functional.held)
    assert held.size and functional.grip[held[0]] > runs.GRIP_HELD
    poses = []
    for i in (held[0], held[-1]):
        kin.pose(functional.q[i], functional.grip[i], True)
        p_tool, R_tool = kin.site("tool")
        c = kin.d.xpos[sc.model.body("container").id]
        poses.append(R_tool.T @ (c - p_tool))
    np.testing.assert_allclose(poses[0], poses[1], atol=1e-9)
    kin.pose(functional.q[held[0]], functional.grip[held[0]], True)
    np.testing.assert_allclose(kin.d.xpos[sc.model.body("container").id], sc.placement.center, atol=2e-3)


def test_button_is_reachable_by_ik(sc, kin, home):
    for p, R in sc.press_targets(kin):
        q, pos_err, rot_err = kin.ik(home, p, R, "tool")
        assert pos_err < 1e-3 and np.degrees(rot_err) < 0.5
        for j, lim in GUARD_LIMITS.items():
            assert abs(q[j]) <= lim


def test_pads_close_to_the_brim_thickness(sc, run, kin):
    i = runs.closing_index(run) + 40
    kin.pose(run.q[i], run.grip[i])
    m = sc.model
    gap = np.linalg.norm(kin.d.geom_xpos[m.geom("g_left_pad1").id] - kin.d.geom_xpos[m.geom("g_right_pad1").id])
    pad_half = m.geom_size[m.geom("g_left_pad1").id][1]
    assert gap - 2 * pad_half == pytest.approx(scene.BRIM_THICK_M, abs=0.002)


def test_the_real_run_never_touches_the_person(functional, kin):
    person = kin.clearance_track(functional)[:, 0]
    assert person.min() > 0.03  # every arm part, the whole run


def test_lathe_meshes_are_closed_solids(sc):
    for name in ("brim_mesh", "wall_mesh"):
        mid = mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_MESH, name)
        assert sc.model.mesh_vertnum[mid] > 0
    verts, faces = scene.lathe([(0.1, 0.0), (0.2, 0.0), (0.2, 0.1), (0.1, 0.1)], segments=16)
    edges = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    assert all(edges.get((b, a), 0) == 1 for (a, b) in edges)  # every edge has its twin: watertight
