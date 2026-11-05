"""The director: its table maps task events to beats, it never puts a spatial offset on a
PRECISION sample (unknown phases included), pauses and slows the task only by the time-warp, and
its tracks pass every safety check."""

import dataclasses

import numpy as np
import pytest

from puppy_sim import checks, runs
from puppy_sim import director as D
from puppy_sim.timeline import DT, PRECISION, speed

SUCCESS = "grab_container_20261029_132745"
FAILURE = "grab_container_20261028_191809"
OTHER = "grab_container_20261029_131558"


@pytest.fixture(scope="module")
def directed(sc):
    """{run name: (functional, puppy, contributions)} for the three real runs."""
    k = sc.kin()
    out = {}
    for name in (SUCCESS, FAILURE, OTHER):
        func = runs.functional_track(runs.load_run(name), k)
        puppy, parts = D.direct_track(func, "grab_container", sc, k)
        out[name] = (func, puppy, parts)
    return out


@pytest.mark.parametrize("name", [SUCCESS, FAILURE, OTHER])
def test_the_director_never_places_a_spatial_offset_in_precision(directed, name):
    _, puppy, parts = directed[name]
    prec = puppy.zone == PRECISION
    assert prec.any()
    for c in parts:
        assert not np.any(c.head[prec]) and not np.any(c.joints[prec]), c.name


@pytest.mark.parametrize("name", [SUCCESS, FAILURE, OTHER])
def test_directed_tracks_pass_every_check(directed, sc, name):
    _, puppy, _ = directed[name]
    failed = [(c.name, c.shown, c.detail) for c in checks.check_track(puppy, sc.kin()) if not c.ok]
    assert not failed


def test_unknown_phases_are_precision_and_get_no_beats(sc):
    """Rename the run's 'view' phase: it becomes PRECISION, and nothing moves there but the task."""
    k = sc.kin()
    run = runs.load_run(SUCCESS)
    renamed = tuple((t, "mystery" if phase == "view" else phase, msg) for t, phase, msg in run.feedback)
    func = runs.functional_track(dataclasses.replace(run, feedback=renamed), k)
    mystery = next(p for p in func.phases if p.name == "mystery")
    span = (func.t >= mystery.t0) & (func.t < mystery.t1)
    assert np.all(func.zone[span] == PRECISION)
    assert "arrived_view" not in {e.name for e in D.find_events(func, k)}
    puppy, parts = D.direct_track(func, "grab_container", sc, k)
    inside = (puppy.tau >= mystery.t0) & (puppy.tau < mystery.t1)
    assert np.all(puppy.zone[inside] == PRECISION)
    for c in parts:
        assert not np.any(c.head[inside]) and not np.any(c.joints[inside]), c.name
    assert puppy.q[inside].tobytes() == puppy.func.q[inside].tobytes()


def test_the_rules_are_a_table_of_known_events_and_beats(directed, sc):
    rules = D.RULES["grab_container"]
    assert {r.beat for r in rules} <= set(D.BEATS)
    k = sc.kin()
    found = {e.name for func, _, _ in directed.values() for e in D.find_events(func, k)}
    assert {r.event for r in rules} <= found  # every event of the table happens in a real run


def _beats(puppy):
    names = []
    for b in sorted(puppy.beats, key=lambda b: b.t0):
        if not names or names[-1] != b.name:
            names.append(b.name)
    return names


def test_the_success_run_gets_its_beats_in_order(directed):
    _, puppy, _ = directed[SUCCESS]
    assert _beats(puppy) == ["EAGER", "CURIOUS", "FOCUS", "PROUD", "ATTENTIVE"]


def test_the_failure_run_searches_droops_and_shakes_it_off(directed):
    _, puppy, _ = directed[FAILURE]
    names = _beats(puppy)
    assert names[-3:] == ["PUZZLED", "SORRY", "SHAKE IT OFF"]
    assert names.index("FOCUS") < names.index("PUZZLED")  # after the failed detect


def test_lock_on_stops_at_the_boundary_then_creeps_in(directed, sc):
    func, puppy, _ = directed[SUCCESS]
    ev = next(e for e in D.find_events(func, sc.kin()) if e.name == "final_approach")
    s = speed(puppy)
    stare = (np.abs(puppy.tau - ev.t) <= DT) & (s == 0.0)
    assert stare.sum() * DT >= D.RULES["grab_container"][6].pause_s - DT
    assert np.all(puppy.zone[stare] != PRECISION)  # the stare is still FREE
    after = np.nonzero(stare)[0][-1] + int(round(1.0 / DT))
    cruise = (puppy.tau > puppy.tau[after]) & (puppy.tau < ev.t_end - 0.6)  # before it eases back up
    assert cruise.any() and s[cruise].max() <= 0.75 + 1e-9  # the slow-in


def test_the_task_only_changes_timing(directed):
    """The warped track is the functional path at tau; dtau/dt in [0, 1]; same start and end."""
    func, puppy, _ = directed[SUCCESS]
    s = speed(puppy)
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert puppy.duration > func.duration
    assert np.array_equal(puppy.func.q[0], func.q[0]) and np.array_equal(puppy.func.q[-1], func.q[-1])
    assert np.abs(puppy.q[-1] - func.q[-1]).max() < 1e-9  # and Puppy Mode ends where the task ended


def test_a_held_container_stays_level(directed, sc):
    """Proud and attentive while holding: pan-only looks, a lift-only perk, a small bob."""
    k = sc.kin()
    _, puppy, _ = directed[SUCCESS]
    worst = 0.0
    for i in np.nonzero(puppy.held)[0][::5]:
        _, R_func = k.fk(puppy.func.q[i], "tool")
        _, R = k.fk(puppy.q[i], "tool")
        up = R @ R_func.T @ np.array([0.0, 0.0, 1.0])  # where the plain pose's vertical went
        worst = max(worst, np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    assert worst < 0.5  # (DLS residue): breathing and micro-life keep the chin level too


def test_direct_returns_the_plain_track_and_is_deterministic(sc):
    run = runs.load_run(FAILURE)
    plain, puppy = D.direct(run, sc)
    plain2, puppy2 = D.direct(run, sc)
    assert plain.q.tobytes() == runs.functional_track(run, sc.kin()).q.tobytes()
    assert puppy.q.tobytes() == puppy2.q.tobytes()
