"""Loading real runs: the goal's phase timeline, terminal status, resampling and the zone table."""

import numpy as np
import pytest

from puppy_sim import runs
from puppy_sim.timeline import DT, FREE, PRECISION


def test_phase_timeline_matches_the_feedback(run):
    starts = {(p, m): t0 for p, m, t0, _ in runs.phase_spans(run)}
    assert starts[("home", "")] == pytest.approx(5.15, abs=0.01)
    assert starts[("view", "")] == pytest.approx(6.43, abs=0.01)
    assert starts[("detect", "")] == pytest.approx(9.28, abs=0.01)
    assert starts[("grasp", "opening")] == pytest.approx(9.83, abs=0.01)
    assert starts[("grasp", runs.APPROACH_MESSAGE)] == pytest.approx(10.78, abs=0.01)
    assert starts[("grasp", "closing")] == pytest.approx(14.98, abs=0.01)
    assert starts[("raise", "")] == pytest.approx(15.86, abs=0.01)
    assert run.terminal == (pytest.approx(17.26, abs=0.01), "done")
    assert all(t > 5.0 for t, _, _ in run.feedback)  # the earlier goal's 'locate' at 2.53 is not ours


def test_aborted_run_ends_aborted():
    r = runs.load_run("grab_container_20261028_191809")
    assert r.terminal[1] == "aborted"
    assert r.feedback[-1][1] == "detect"


def test_zone_table_and_fail_safe():
    assert runs.zone_of("grab_container", "detect", "") == PRECISION
    assert runs.zone_of("grab_container", "grasp", "closing") == PRECISION
    assert runs.zone_of("grab_container", "view", "") == FREE
    assert runs.zone_of("grab_container", "mystery", "") == PRECISION
    assert runs.zone_of("some_other_capability", "home", "") == PRECISION


def test_functional_track_follows_the_recording(run, functional):
    t0, _ = runs.replay_window(run)
    raw = np.column_stack([np.interp(t0 + functional.t, run.t, run.q[:, j]) for j in range(7)])
    assert np.abs(functional.q - raw).max() < 0.01  # light smoothing only
    assert functional.tau[1] - functional.tau[0] == pytest.approx(DT)
    zones = {p.name: functional.zone[int(round((p.t0 + p.t1) / 2 / DT))] for p in functional.phases}
    assert zones["detect"] == PRECISION and zones["close"] == PRECISION and zones["raise"] == PRECISION
    assert zones["view"] == FREE and zones["done"] == FREE
    approach = next(p for p in functional.phases if p.name == "approach")
    a, b = int(round(approach.t0 / DT)), int(round(approach.t1 / DT))
    assert functional.zone[a] == FREE and functional.zone[b - 1] == PRECISION  # the final approach
