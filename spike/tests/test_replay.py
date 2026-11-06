"""Replays driven by puppy_motion's Director: the events a run implies, the beats spliced in as holds,
the PRECISION samples untouched, the budgets kept, and every beat valid under check_beat."""

import numpy as np
import pytest

from puppy_motion import director
from puppy_sim import checks, replay, runs, scene
from puppy_sim.timeline import DT, PRECISION

RUNS = ("grab_container_20261029_132745", "grab_container_20261029_131558", "grab_container_20261028_191809")
FAILED_RUN = "grab_container_20261028_191809"
PRESETS = ("demo", "subtle")


@pytest.fixture(scope="module")
def replays():
    return {(r, p): replay.direct_run(r, p) for r in RUNS for p in PRESETS}


def beat_names(track):
    return {b.name for b in track.beats}


def test_events_follow_the_recording():
    run = runs.load_run(RUNS[0])
    plain = runs.functional_track(run, scene.MotionKin())
    events = replay.events_from_run(plain, scene.motion_chain(RUNS[0]))
    names = [type(e).__name__ for _, e in events]
    assert names[0] == "TaskRequested" and names[-1] == "AfterMove" and "Succeeded" in names
    samples = [round(t / DT) for t, _ in events]
    assert samples == sorted(samples)
    phases = [(e.name, e.message) for _, e in events if type(e).__name__ == "Phase"]
    assert phases == [replay._today(p, m) for _, p, m in run.feedback] + [("done", "")]
    approach = [e for _, e in events if type(e).__name__ == "BeforeMove" and e.precise_end]
    assert len(approach) == 1  # only the stop before the move that ends at the grasp pose


def test_failed_run_ends_with_failed_true():
    plain = runs.functional_track(runs.load_run(FAILED_RUN), scene.MotionKin())
    events = replay.events_from_run(plain, scene.motion_chain(FAILED_RUN))
    last = [e for _, e in events if type(e).__name__ in ("Succeeded", "Failed")]
    assert [type(e).__name__ for e in last] == ["Failed"] and last[0].recoverable is True


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("run", RUNS)
def test_replay_precision_identical(replays, run, preset):
    plain, puppy = replays[run, preset]
    u = puppy.tau / DT
    p_mask = puppy.zone == PRECISION
    assert p_mask.any() and np.all(np.abs(u[p_mask] - np.round(u[p_mask])) < 1e-9)
    p_idx = np.round(u[p_mask]).astype(int)
    assert np.array_equal(puppy.q[p_mask], plain.q[p_idx])
    assert set(np.flatnonzero(plain.zone == PRECISION)) <= set(p_idx)  # every precise sample is played


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("run", RUNS)
def test_replay_within_budget(replays, run, preset):
    plain, puppy = replays[run, preset]
    assert puppy.duration - plain.duration <= director.BUDGETS[preset] + DT
    assert puppy.spliced


def test_failure_run_reacts(replays):
    plain, puppy = replays[FAILED_RUN, "demo"]
    assert {"sorry"} <= beat_names(puppy)
    assert "sorry" in beat_names(replays[FAILED_RUN, "subtle"][1])


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("run", RUNS)
def test_every_spliced_beat_validates(replays, run, preset):
    from puppy_motion.validate import check_beat

    plain, puppy = replays[run, preset]
    chain, person = scene.motion_chain(run), scene.person_scene()
    for sp in puppy.spliced:
        b = sp.beat
        assert check_beat(b.points, sp.q_now, chain, person, b.zone, puppy.caps, held=sp.held) == []
        assert np.array_equal(puppy.q[sp.i0 : sp.i0 + len(b.points)], np.array([p[1] for p in b.points]))
        assert np.abs(plain.q[int(round(puppy.tau[sp.i0] / DT))] - sp.q_now).max() < 1e-3
    assert all(c.ok for c in checks.replay_checks(plain, puppy)), [c for c in checks.replay_checks(plain, puppy) if not c.ok]


def test_replay_is_deterministic():
    a = replay.direct_run(RUNS[0], "demo")[1]
    b = replay.direct_run(RUNS[0], "demo")[1]
    assert np.array_equal(a.q, b.q)


def test_standoff_splits_the_approach_and_creeps_in(monkeypatch):
    monkeypatch.setitem(director.BUDGETS, "demo", 30.0)
    plain, puppy = replay.direct_run(RUNS[0], "demo", lock_on="standoff")
    assert "lock_on" in beat_names(puppy) and puppy.tier == "standoff"
    t_split, t_close = replay.standoff_split(plain, scene.motion_chain(RUNS[0]))
    chain = scene.motion_chain(RUNS[0])
    run = runs.load_run(RUNS[0])
    p_grasp = chain.fk(run.q[runs.closing_index(run)], "tool")[0]
    d = np.linalg.norm(chain.fk(plain.q[int(round(t_split / DT))], "tool")[0] - p_grasp)
    assert 0.10 <= d < 0.20  # the last free sample outside the final approach (marked at 0.11 m)
    lock = next(s for s in puppy.spliced if s.beat.name == "lock_on")
    assert abs(puppy.tau[lock.i0] - t_split) < 1e-6
    speed = np.diff(puppy.tau) / DT
    creep = (puppy.tau[:-1] > t_split + 0.5) & (puppy.tau[:-1] < t_close - 0.5)
    assert np.allclose(speed[creep], replay.STANDOFF_SLOW, atol=1e-6)
    assert all(c.ok for c in checks.replay_checks(plain, puppy)), [c for c in checks.replay_checks(plain, puppy) if not c.ok]
    split_events = [type(e).__name__ for t, e in replay.events_from_run(plain, chain, "standoff") if abs(t - t_split) < 1e-9]
    assert split_events == ["AfterMove", "BeforeMove"]


def test_standoff_subtitle_names_the_capability_change():
    from puppy_sim.clips import registry

    clip = registry()["replay_director_standoff"]
    assert "capability change" in clip.subtitle and clip.subtitle == replay.STANDOFF_SUBTITLE


@pytest.mark.parametrize("name", ["replay_director", "replay_director_standoff"])
def test_replay_clips_pass_every_check(name):
    for clip, label, _, results, _ in checks.check_named(name):
        assert checks.passed(results), (label, [c for c in results if not c.ok])
