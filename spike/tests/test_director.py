"""The director: its table maps task events to beats (several per event), it decides event by event
within an absolute time budget, its default (boundary) beats play only where the arm is stopped
between moves and leave every move exactly as recorded, it never puts a spatial offset on a
PRECISION sample (unknown phases included), and its tracks pass every safety check."""

import dataclasses

import numpy as np
import pytest

from puppy_sim import checks, runs
from puppy_sim import director as D
from puppy_sim import timeline as tl
from puppy_sim.timeline import DT, PRECISION, STOP_RAD_S, speed

SUCCESS = "grab_container_20261029_132745"
FAILURE = "grab_container_20261028_191809"
OTHER = "grab_container_20261029_131558"
RUNS = (SUCCESS, FAILURE, OTHER)


@pytest.fixture(scope="module")
def functionals(sc):
    k = sc.kin()
    return {name: runs.functional_track(runs.load_run(name), k) for name in RUNS}


@pytest.fixture(scope="module")
def directed(sc, functionals):
    """{run name: (functional, puppy, contributions)} with the defaults: boundary tier, 'demo'
    budget, lock-on at the view pose."""
    k = sc.kin()
    return {name: (f, *D.direct_track(f, "grab_container", sc, k)) for name, f in functionals.items()}


def _beats(puppy):
    names = []
    for b in sorted(puppy.beats, key=lambda b: b.t0):
        if not names or names[-1] != b.name:
            names.append(b.name)
    return names


def _source_moving(puppy):
    """(N,) output samples on a moving sample of the source (the plain functional track)."""
    src = puppy.func.source
    u = np.interp(puppy.func.tau, src.tau, np.arange(src.n, dtype=float))
    moving = tl.joint_speed(src) >= STOP_RAD_S
    return moving[np.floor(u).astype(int)] | moving[np.ceil(u).astype(int)], u


@pytest.mark.parametrize("name", RUNS)
def test_the_director_never_places_a_spatial_offset_in_precision(directed, name):
    _, puppy, parts = directed[name]
    prec = puppy.zone == PRECISION
    assert prec.any()
    for c in parts:
        assert not np.any(c.head[prec]) and not np.any(c.joints[prec]), c.name


@pytest.mark.parametrize("name", RUNS)
def test_directed_tracks_pass_every_check(directed, sc, name):
    _, puppy, _ = directed[name]
    failed = [(c.name, c.shown, c.detail) for c in checks.check_track(puppy, sc.kin()) if not c.ok]
    assert not failed


@pytest.mark.parametrize("name", RUNS)
def test_boundary_beats_leave_every_move_exactly_as_recorded(directed, name):
    """On every sample where the plain task moves: the same speed (dtau/dt exactly 1 into and out
    of it) and the composite bit for bit the functional pose (no beat rides on a move)."""
    _, puppy, _ = directed[name]
    assert puppy.tier == D.BOUNDARY
    on_move, u = _source_moving(puppy)
    assert on_move.sum() > 200
    step = np.abs(np.diff(u) - 1.0) < 1e-6
    assert np.all(step[on_move[:-1]]) and np.all(step[on_move[1:]])
    assert puppy.q[on_move].tobytes() == puppy.func.q[on_move].tobytes()


@pytest.mark.parametrize("name", RUNS)
def test_pauses_go_only_where_the_arm_is_already_stopped(sc, functionals, name):
    """Every hold the default plan inserts, with its ramps, lies inside a stage (the arm stopped
    between moves, outside PRECISION), and there is no slow-in."""
    func = functionals[name]
    k = sc.kin()
    found = D.find_events(func, k)
    rules = [r for r in D.RULES["grab_container"] if r.tier == D.BOUNDARY and r.lock_on in (None, "view")]
    planned = D.plan(func, found, rules, sc, k, budget=D.BUDGETS["demo"])
    assert planned.holds and not planned.slows
    u = tl.warp(func, planned.holds, planned.slows).tau / DT
    retimed = np.nonzero(np.abs(np.diff(u) - 1.0) > 1e-9)[0]
    in_stage = D.stage_mask(func)
    assert retimed.size and np.all(in_stage[np.floor(u[retimed]).astype(int)])
    assert np.all(in_stage[np.ceil(u[retimed + 1]).astype(int)])


def test_unknown_phases_are_precision_and_get_no_beats(sc):
    """Rename the run's 'view' phase: it becomes PRECISION, and nothing moves there but the task."""
    k = sc.kin()
    run = runs.load_run(SUCCESS)
    renamed = tuple((t, "mystery" if phase == "view" else phase, msg) for t, phase, msg in run.feedback)
    func = runs.functional_track(dataclasses.replace(run, feedback=renamed), k)
    mystery = next(p for p in func.phases if p.name == "mystery")
    span = (func.t >= mystery.t0) & (func.t < mystery.t1)
    assert np.all(func.zone[span] == PRECISION)
    assert "arrived_view" not in {e.name for e in D.find_events(func, k)[1]}
    puppy, parts = D.direct_track(func, "grab_container", sc, k, budget=None)
    inside = (puppy.tau >= mystery.t0) & (puppy.tau < mystery.t1)
    assert np.all(puppy.zone[inside] == PRECISION)
    for c in parts:
        assert not np.any(c.head[inside]) and not np.any(c.joints[inside]), c.name
    assert puppy.q[inside].tobytes() == puppy.func.q[inside].tobytes()


def test_the_rules_are_a_table_of_known_events_and_beats(functionals, sc):
    rules = D.RULES["grab_container"]
    assert {r.beat for r in rules} <= set(D.BEATS)
    k = sc.kin()
    found = {e.name for func in functionals.values() for e in D.find_events(func, k)[1]}
    assert {r.event for r in rules} <= found  # every event of the table happens in a real run


def test_several_rules_on_one_event_all_play_in_table_order(sc, functionals, monkeypatch):
    """A second rule on an event no longer replaces the first (the built-in precision rule stays),
    and the failure's search and sorry both play, in table order."""
    k = sc.kin()
    extra = D.Rule("precision", "stillness", "STILL")
    monkeypatch.setitem(D.RULES, "two", D.RULES["grab_container"] + (extra,))
    puppy, _ = D.direct_track(functionals[SUCCESS], "two", sc, k, budget=None)
    names = {b.name for b in puppy.beats}
    assert {"FOCUS", "STILL"} <= names
    failed, _ = D.direct_track(functionals[FAILURE], "grab_container", sc, k, budget=None)
    order = _beats(failed)
    assert order.index("PUZZLED") < order.index("SORRY") < order.index("SHAKE IT OFF")


def test_the_success_run_gets_its_beats_in_order(sc, functionals, directed):
    assert _beats(directed[SUCCESS][1]) == ["CURIOUS", "FOCUS", "PROUD"]
    every, _ = D.direct_track(functionals[SUCCESS], "grab_container", sc, sc.kin(), budget=None)
    assert _beats(every) == ["EAGER", "CURIOUS", "FOCUS", "PROUD", "ATTENTIVE"]


def test_the_failure_run_looks_puzzled_and_droops(directed):
    names = _beats(directed[FAILURE][1])
    assert names[-2:] == ["PUZZLED", "SORRY"]
    assert names.index("FOCUS") < names.index("PUZZLED")  # after the failed detect


@pytest.mark.parametrize("budget", sorted(D.BUDGETS))
@pytest.mark.parametrize("name", RUNS)
def test_a_budget_is_never_exceeded(sc, functionals, name, budget):
    func = functionals[name]
    puppy, _ = D.direct_track(func, "grab_container", sc, sc.kin(), budget=budget)
    assert puppy.duration - func.duration <= D.BUDGETS[budget].seconds + 1e-9
    assert puppy.budget == D.BUDGETS[budget] and puppy.log[0].startswith("time: plain")


def test_a_tighter_budget_shrinks_and_drops_the_less_important_beats_first(sc, functionals):
    """subtle keeps only the most important (the lock-on stare); demo adds the reaction and the
    curious tilt; no budget plays everything."""
    k = sc.kin()
    sets = [
        set(_beats(D.direct_track(functionals[SUCCESS], "grab_container", sc, k, budget=b)[0]))
        for b in ("subtle", "demo", None)
    ]
    assert sets[0] < sets[1] < sets[2]
    assert "FOCUS" in sets[0] and {"PROUD", "CURIOUS"} <= sets[1] and "EAGER" not in sets[1]


@pytest.mark.parametrize("name", (SUCCESS, FAILURE))
def test_decisions_use_only_the_events_so_far(sc, functionals, name):
    """The director run on the first k events decides those exactly as it does knowing them all:
    no decision depends on an event that has not happened yet."""
    func = functionals[name]
    k = sc.kin()
    stages, events = D.find_events(func, k)
    rules = [r for r in D.RULES["grab_container"] if r.tier == D.BOUNDARY and r.lock_on in (None, "view")]
    full = D.plan(func, (stages, events), rules, sc, k, budget=D.BUDGETS["demo"]).log
    for cut in range(1, len(events)):
        part = D.plan(func, (stages, events[:cut]), rules, sc, k, budget=D.BUDGETS["demo"]).log
        assert full[: len(part)] == part


def test_lock_on_at_the_view_pose_stares_before_the_approach(directed):
    """Default: the stare plays in the stop after detect, before the approach starts (one blended
    call), and the approach then runs at full speed."""
    func, puppy, _ = directed[SUCCESS]
    focus = [b for b in puppy.beats if b.name == "FOCUS" and puppy.zone[int(b.t0 / DT)] != PRECISION]
    assert len(focus) == 1
    stare = focus[0]
    detect = next(p for p in puppy.phases if p.name == "detect")
    on_move, _ = _source_moving(puppy)
    approach_moves = np.nonzero(on_move & (puppy.t >= detect.t1))[0][0] * DT
    assert detect.t1 <= stare.t0 < stare.t1 <= approach_moves
    assert np.all(np.abs(speed(puppy)[on_move] - 1.0) < 1e-9)


def test_lock_on_at_the_standoff_stops_then_creeps_in(sc, functionals):
    """The explicit standoff option: stop at the pre-grasp standoff, stare, creep in at x0.75; it
    retimes the approach (the capability would have to split its call) and says so."""
    func = functionals[SUCCESS]
    k = sc.kin()
    puppy, _ = D.direct_track(func, "grab_container", sc, k, lock_on="standoff", budget=None)
    assert puppy.tier == D.STANDOFF
    ev = next(e for e in D.find_events(func, k)[1] if e.name == "final_approach")
    s = speed(puppy)
    stare = (np.abs(puppy.tau - ev.t) <= DT) & (s == 0.0)
    rule = next(r for r in D.RULES["grab_container"] if r.lock_on == "standoff")
    assert stare.sum() * DT >= rule.pause_s - DT
    assert np.all(puppy.zone[stare] != PRECISION)  # the stare is still FREE
    after = np.nonzero(stare)[0][-1] + int(round(1.0 / DT))
    ramp = D._standoff(func, ev, rule, rule.pause_s).ramp_s  # it eases back up to full speed over this
    cruise = (puppy.tau > puppy.tau[after]) & (puppy.tau < ev.t_end - ramp)
    assert cruise.any() and s[cruise].max() <= rule.slow + 1e-9
    tier = next(c for c in checks.check_track(puppy, k) if c.name == "tier")
    assert tier.ok and "retimed" in tier.shown
    with pytest.raises(ValueError):
        D.direct_track(func, "grab_container", sc, k, lock_on="halfway")


def test_in_motion_beats_are_opt_in(sc, functionals, directed):
    """lead_gaze rides on the moves only when asked; the check then reports what it layers."""
    k = sc.kin()
    assert all(c.name != "lead_gaze" for c in directed[SUCCESS][2])
    puppy, parts = D.direct_track(functionals[SUCCESS], "grab_container", sc, k, tiers=(D.BOUNDARY, D.IN_MOTION))
    assert puppy.tier == D.IN_MOTION and any(c.name == "lead_gaze" and c.active().any() for c in parts)
    on_move, _ = _source_moving(puppy)
    assert np.any(puppy.q[on_move] != puppy.func.q[on_move])
    tier = next(c for c in checks.check_track(puppy, k) if c.name == "tier")
    assert tier.ok and "layered" in tier.shown


def test_the_task_only_changes_timing(directed):
    """The warped track is the functional path at tau; dtau/dt in [0, 1]; same start and end."""
    func, puppy, _ = directed[SUCCESS]
    s = speed(puppy)
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9
    assert puppy.duration > func.duration
    assert np.array_equal(puppy.func.q[0], func.q[0]) and np.array_equal(puppy.func.q[-1], func.q[-1])
    assert np.abs(puppy.q[-1] - func.q[-1]).max() < 1e-9  # and Puppy Mode ends where the task ended


def test_a_held_container_stays_level(directed, sc):
    """Proud while holding: a pan-only look; breathing and micro-life keep the chin level too."""
    k = sc.kin()
    _, puppy, _ = directed[SUCCESS]
    worst = 0.0
    for i in np.nonzero(puppy.held)[0][::5]:
        _, R_func = k.fk(puppy.func.q[i], "tool")
        _, R = k.fk(puppy.q[i], "tool")
        up = R @ R_func.T @ np.array([0.0, 0.0, 1.0])  # where the plain pose's vertical went
        worst = max(worst, np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    assert worst < 0.5  # (DLS residue)


def test_direct_returns_the_plain_track_and_is_deterministic(sc):
    run = runs.load_run(FAILURE)
    plain, puppy = D.direct(run, sc)
    plain2, puppy2 = D.direct(run, sc)
    assert plain.q.tobytes() == runs.functional_track(run, sc.kin()).q.tobytes()
    assert puppy.q.tobytes() == puppy2.q.tobytes()
