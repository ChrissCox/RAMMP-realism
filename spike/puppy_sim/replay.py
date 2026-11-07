"""A recorded run replayed through puppy_motion's Director, so the preview shows what the real hook
would do: the same events, in the same order, answered by the same beats.

events_from_run turns a run into the events a capability would have reported (Phase from the recorded
feedback, AfterMove / BeforeMove at the arm's stops, Succeeded / Failed from the terminal status).
direct_run feeds them to Director.on_event with the recorded posture at that time and splices each
returned beat into the plain track as a time-warp hold of the beat's length (the arm is stopped when
an event arrives, so the beat starts and ends at the recorded posture). Nothing else about the run
changes: the path, and every PRECISION sample, play exactly as recorded.

lock_on='standoff' is the one option that changes the task: grab_container's approach is one driver
call, so stopping 10 cm short of the grasp needs the capability to split it in two (AfterMove +
BeforeMove(precise_end=True) at the split). The split sits at the last free sample before the final
approach is marked PRECISION (a little outside 10 cm), never inside it. The creep after the stare
runs the rest of the approach at 0.75 x speed.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from . import runs, scene, timeline
from .timeline import DT, PRECISION, Budget, Hold, Label, Slow

STOP_MIN_S = 0.2  # a stop: every joint slower than STOP_RAD_S for this long (or across a phase change)
HOLD_EDGE = 3  # samples a stop's ends keep clear
HOLD_RAMP_S = 0.04  # a stop's hold eases in and out over 2 samples (the arm is stopped: nothing visible); the
# beat covers these samples too, so a hold adds exactly the beat's length
STANDOFF_RAMP_MIN_S = 0.3
STANDOFF_SLOW = 0.75
STANDOFF_M = 0.10  # the capability's split point: the tool this far from the grasp pose
STOPPED_RAD = 1e-3  # a stop moves the arm less than this over the hold's easing samples
PHASE_REACH = 3  # samples
SPLIT_MARGIN = 3  # samples the split keeps clear of the final approach (besides its ramp)
CONTAINER_TARGET = "container"
STANDOFF_SUBTITLE = (
    "needs grab_container to split its reach into two calls (view -> about 11 cm short, then -> grasp): "
    "a capability change; the creep's extra time is outside the beat budget"
)
DISPLAY = {  # the chip text of a beat in the gallery
    "eager": "EAGER", "wind_up": "WIND UP", "follow_through": "FOLLOW THROUGH", "curious": "CURIOUS",
    "lock_on": "LOCK ON", "proud": "PROUD", "attentive": "ATTENTIVE", "search": "SEARCH", "sorry": "SORRY",
    "shake_off": "SHAKE IT OFF",
}
RANK = {"TaskRequested": 0, "AfterMove": 1, "Phase": 2, "BeforeMove": 3, "Succeeded": 4, "Failed": 4}


@dataclass(frozen=True)
class Spliced:
    """One beat played into a puppy track: where (output sample i0..i1), and what it was given."""

    beat: object  # puppy_motion.beat.Beat
    event: str  # the event class name it answered
    i0: int
    q_now: np.ndarray
    held: bool


def _pm():
    from puppy_motion import director

    return director


def _run_of(track):
    return runs.load_run(track.name)


def _stage_runs(track):
    """[(i0, i1, before, after)]: the stops of a functional track (see timeline.STOP_RAD_S)."""
    still = (timeline.joint_speed(track) < timeline.STOP_RAD_S) & (track.zone != PRECISION)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], still.astype(int), [0]])))
    starts = [int(round(p.t0 / DT)) for p in track.phases]
    out = []
    for i0, i1 in zip(edges[::2], edges[1::2] - 1):
        long_enough = (i1 - i0 + 1) * DT >= STOP_MIN_S or any(i0 <= k <= i1 for k in starts)
        if not long_enough or i1 - i0 < 2 * HOLD_EDGE:
            continue
        before = "start" if i0 == 0 else ("precision" if track.zone[i0 - 1] == PRECISION else "move")
        after = "end" if i1 == track.n - 1 else ("precision" if track.zone[i1 + 1] == PRECISION else "move")
        out.append((int(i0), int(i1), before, after))
    return out


def _today(phase, msg):
    """The recorded runs report the reach as ('grasp', APPROACH_MESSAGE); grasp.py today reports the
    same move under its own phase, 'approach' (the zone table's precise_end row)."""
    return ("approach", msg) if (phase, msg) == ("grasp", runs.APPROACH_MESSAGE) else (phase, msg)


def _entry_at(run, window_t0, t):
    """The zone table's entry of the phase active at functional time t (None before the goal)."""
    from puppy_motion.zones import zone_of

    entry = None
    for ts, phase, msg in run.feedback:
        if ts - window_t0 <= t + 1e-9:
            entry = zone_of(run.capability, *_today(phase, msg))
    return entry


def standoff_split(track, chain):
    """(t_split, t_close), or None when the run has no approach to split: where a split approach would stop (the last free sample before the final
    approach, the tool a little over STANDOFF_M from the grasp pose) and where the gripper closes."""
    approach = next((p for p in track.phases if p.name == "approach"), None)
    close = next((p for p in track.phases if p.name == "close"), None)
    if approach is None or close is None:
        return None  # a run that never reached the approach has no split
    i0, i1 = int(round(approach.t0 / DT)), int(round(approach.t1 / DT))
    precise = np.flatnonzero(track.zone[i0 : i1 + 1] == PRECISION)
    first = i0 + (int(precise[0]) if precise.size else i1 - i0)
    ramp = timeline._ramp_steps(STANDOFF_RAMP_MIN_S) // 2
    i_split = max(i0 + 1, first - ramp - SPLIT_MARGIN)
    return i_split * DT, close.t0


def events_from_run(track, chain, lock_on="view", t0=None):
    """[(t, Event)] in time order for the functional track of a recorded run: what grab_container
    would have told its director. t is the track's functional time; t0 is the run time of its first
    sample (default: the start of the replay window)."""
    d = _pm()
    run = _run_of(track)
    t0 = runs.replay_window(run)[0] if t0 is None else t0
    out = []
    if run.feedback:
        out.append((max(run.feedback[0][0] - t0, 0.0), d.TaskRequested()))
    for ts, phase, msg in run.feedback:
        if t0 <= ts < t0 + track.duration:
            out.append((ts - t0, d.Phase(*_today(phase, msg))))
    split = standoff_split(track, chain) if lock_on == "standoff" else None
    standoff = split is not None  # no approach to split: the view path
    if standoff:
        t_split, _ = split
    for i0, i1, before, after in _stage_runs(track):
        if before != "start":
            out.append((i0 * DT, d.AfterMove()))
        if after != "end":
            entry = _entry_at(run, t0, (i1 + 1) * DT)
            precise = bool(entry is not None and entry.precise_end) and not standoff
            out.append((i1 * DT, d.BeforeMove(precise)))
    if standoff:
        out += [(t_split, d.AfterMove()), (t_split, d.BeforeMove(True))]
    if run.terminal is not None and run.terminal[0] - t0 <= track.duration:  # (not in a window that ends earlier)
        t_term = run.terminal[0] - t0
        out.append((t_term, d.Phase(run.terminal[1], "")))  # the terminal state is the last phase ('done', 'aborted')
        if run.terminal[1] == "done":
            out.append((t_term, d.Succeeded()))
        else:  # aborted / canceled: the goal failed where it stood; a retry is possible
            out.append((t_term, d.Failed(True)))
    # a phase that starts within PHASE_REACH samples after a stop ends was set before the move: the
    # capability reports it at the stop's end, so the BeforeMove that follows it is in a PRECISION phase
    ends = [i1 for _, i1, _, after in _stage_runs(track) if after != "end"]
    shifted = []
    for t, ev in out:
        i = round(t / DT)
        if isinstance(ev, d.Phase):
            i = next((i1 for i1 in ends if i1 <= i <= i1 + PHASE_REACH), i)
        shifted.append((i, t, ev))
    # the terminal status comes once the last move has returned: after every AfterMove
    last_after = max((i for i, _, ev in shifted if isinstance(ev, d.AfterMove)), default=0)
    terminal = (d.Succeeded, d.Failed)
    t_end = max((i for i, _, ev in shifted if isinstance(ev, terminal)), default=None)
    if t_end is not None:
        shifted = [(max(i, last_after) if i >= t_end else i, t, ev) for i, t, ev in shifted]
    shifted.sort(key=lambda e: (e[0], RANK[type(e[2]).__name__]))
    return [(i * DT, ev) for i, _, ev in shifted]


def _accel_cap(caps):
    from puppy_motion.caps import CAP_PRESETS
    from puppy_motion.track import Zone

    return CAP_PRESETS[caps][Zone.FREE].joint_accel


def _touches_precision(track, idx, reach):
    """True when a PRECISION sample is within `reach` samples of idx (or the span leaves the track)."""
    return idx - reach < 0 or idx + reach >= track.n or bool((track.zone[idx - reach : idx + reach + 1] == PRECISION).any())


def _stand(track, idx):
    """Output index where the hold at functional sample idx starts to stand still (dtau/dt = 0): the
    stand nearest idx (after a slow-in the later stands fall between functional samples)."""
    still = np.flatnonzero(np.diff(track.tau) == 0.0)
    starts = still[np.concatenate([[True], np.diff(still) > 1])]
    return int(starts[np.argmin(np.abs(track.tau[starts] - idx * DT))])


def _dropped(ev, drop):
    """True when `ev` is named in drop: its class name, or 'BeforeMove(True)' / 'BeforeMove(False)'."""
    name = type(ev).__name__
    return name in drop or (name == "BeforeMove" and f"BeforeMove({ev.precise_end})" in drop)


def direct_run(run_name, preset="demo", lock_on="view", caps="default", seed=0, window=None, drop=()):
    """(plain, puppy) for a recorded run: its functional track, and the track the puppy_motion
    Director makes of it (each returned beat is a time-warp hold of its length at its event).
    window: (t0, t1) seconds of the replay window to show (default: all of it); drop: names of event
    classes (or 'BeforeMove(False)') the capability is not asked to report (a preview showing one kind of beat)."""
    d = _pm()
    chain = scene.motion_chain(run_name)
    person = scene.person_scene()
    run = runs.load_run(run_name)
    kin = scene.MotionKin(chain, person)
    start = runs.replay_window(run)[0]
    plain = runs.functional_track(run, kin, None if window is None else (start + window[0], start + window[1]))
    from puppy_motion.behaviors import Targets

    targets = Targets(person_head=person.person_head(), container=scene.build().placement.center.copy())
    split = standoff_split(plain, chain) if lock_on == "standoff" else None
    standoff = split is not None
    notes = []
    if lock_on == "standoff" and not standoff:
        notes.append("lock_on='standoff': this run never reaches the approach, so there is no split; played as 'view'")
        lock_on = "view"
    director = d.Director(run.capability, chain, person, targets, preset, lock_on, caps, seed)
    events = events_from_run(plain, chain, lock_on, start + (0.0 if window is None else window[0]))
    events = [(t, ev) for t, ev in events if not _dropped(ev, drop)]
    puppy = play_events(plain, events, director, split, caps, preset, notes)
    puppy.name = plain.name
    return plain, puppy


def stop_events(track, precise_before=()):
    """[(t, Event)] for the stops of a scripted track, as a capability's moves would report them:
    AfterMove at the start of each stop the arm arrives in, BeforeMove at the end of each stop it
    leaves (precise_end for the stops listed in `precise_before`: seconds near the stop's end, the
    move that follows ends at a precise pose), sorted as events_from_run's."""
    d = _pm()
    out = []
    for i0, i1, before, after in _stage_runs(track):
        if before != "start":
            out.append((i0 * DT, d.AfterMove()))
        if after != "end":
            precise = any(abs(t - i1 * DT) < 0.5 for t in precise_before)
            # a stop followed by PRECISION takes its hold a little earlier: the hold eases around it
            out.append(((i1 - (HOLD_EDGE if after == "precision" else 0)) * DT, d.BeforeMove(precise)))
    return out


def sorted_events(events):
    """events in time order, simultaneous ones in RANK order."""
    return sorted(events, key=lambda e: (round(e[0] / DT), RANK[type(e[1]).__name__]))


def direct_scripted(plain, events, tasks=(("grab_container", 0.0),), preset="demo", caps="default", seed=0,
                    targets=None, budget_s=None, e=None):
    """The puppy track of a scripted (not recorded) task: `plain` is its preview Track, `events` the
    [(t, Event)] its capability would report, in time order. tasks: [(capability, t_from)], one
    Director each from its time on (consecutive tasks); budget_s lifts each one's time budget (the
    track's budget is then the sum). Same director, holds and checks as a recorded run (play_events);
    `targets` default to the preview scene's (the person, the container)."""
    from . import lib

    e = e or lib.env()
    d = _pm()
    directors = [(t, d.Director(cap, e.chain, e.person, targets or e.targets, preset, "view", caps, seed)) for cap, t in tasks]
    each = d.BUDGETS[preset] if budget_s is None else budget_s
    for _, dr in directors:
        dr.budget_s = each
    puppy = play_events(plain, events, directors, None, caps, preset, [], budget_s=each * len(tasks))
    if budget_s is not None:
        puppy.budget = timeline.Budget(f"{preset} (lifted to {budget_s:.0f} s for this preview)", each * len(tasks))
    elif len(tasks) > 1:
        puppy.budget = timeline.Budget(f"{preset} x{len(tasks)} tasks", each * len(tasks))
    puppy.name = plain.name
    return puppy


def play_events(plain, events, director, split, caps, preset, notes, budget_s=None):
    """The puppy track: `events` fed to `director`, each beat it returns spliced in as a hold at its
    event (the standoff `split`, (t_split, t_close), makes the hold there a creep). `director` may be
    a list of (t_from, Director): consecutive tasks, each decided by its own director from its time on;
    budget_s is then their budgets' sum (default: the preset's, for one task)."""
    d = _pm()
    directors = director if isinstance(director, list) else [(0.0, director)]
    budget_s = d.BUDGETS[preset] if budget_s is None else budget_s
    accel = _accel_cap(caps)
    standoff = split is not None
    if standoff:
        t_split, t_close = split
        i_split = int(round(t_split / DT))

    # group the events of one sample, decide each in order, and place one hold per group
    groups, last_end = [], -1
    for t, ev in events:
        idx = min(max(int(round(t / DT)), 0), plain.n - 2)
        if groups and idx == groups[-1][0]:
            groups[-1][1].append(ev)
        else:
            groups.append((idx, [ev]))
    holds, slows, placed = [], [], []  # placed: (idx, [(event name, beat)], q_now, held)
    for idx, evs in groups:
        at_split = standoff and idx == i_split
        half = 1 if not at_split else timeline._ramp_steps(STANDOFF_RAMP_MIN_S) // 2
        if placed and idx - half < last_end:  # the previous hold has not yet eased back up: wait for it
            idx = last_end + half
        stay = idx
        while idx < plain.n - 1 and _touches_precision(plain, idx, half + 1):
            idx += 1  # a hold eases over the samples around idx: none of them may be PRECISION
        if idx >= plain.n - 1:
            break
        q_now, held = plain.q[idx].copy(), bool(plain.held[idx])
        director = [dr for t_from, dr in directors if t_from <= idx * DT + 1e-9][-1]
        got = [(type(ev).__name__, b) for ev in evs for b in director.on_event(ev, q_now, held)]
        if got and (_touches_precision(plain, idx, half + 1) or np.abs(q_now - plain.q[stay]).max() > STOPPED_RAD):
            director.log.append(f"{[n for n, _ in got]} dropped at {idx * DT:.2f} s: the arm is not stopped clear of PRECISION")
            got = []
        if not got:
            continue
        n_beat = sum(len(b.points) - 1 for _, b in got)  # samples the beats take, end to end
        if at_split:
            ramp = timeline.hold_ramp_s(plain, idx * DT, accel, STANDOFF_RAMP_MIN_S)
            half = timeline._ramp_steps(ramp) // 2
            slows.append(Slow(idx * DT, t_close, STANDOFF_SLOW, ramp, n_beat * DT))
        else:  # the beat also covers the hold's two easing samples: the hold adds n_beat samples in all
            holds.append(Hold(idx * DT, max(n_beat - 2, 1) * DT, HOLD_RAMP_S))
        last_end = idx + half
        placed.append((idx, got, q_now, held, at_split))

    warped = timeline.warp(plain, holds, slows)
    puppy = warped
    puppy.q = warped.q.copy()
    puppy.func = replace(warped, q=warped.q.copy())  # the plain path on the output clock, as a composed track has
    puppy.gain = (warped.zone != PRECISION).astype(float)
    puppy.spliced, puppy.beats = [], []
    for idx, got, q_now, held, at_split in placed:
        a = _stand(puppy, idx)
        if not at_split:
            a -= 1  # the beat starts on the hold's first easing sample
        for ev_name, beat in got:
            n = len(beat.points)
            if a < 0 or a + n > puppy.n or np.abs(puppy.q[a] - q_now).max() > STOPPED_RAD:
                raise RuntimeError(f"cannot splice {beat.name}: the arm is not stopped where the beat starts")
            puppy.q[a : a + n] = np.array([p[1] for p in beat.points])
            puppy.spliced.append(Spliced(beat, ev_name, a, q_now, held))
            puppy.beats.append(Label(beat.name, a * DT, (a + n - 1) * DT))
            a += n - 1
    puppy.caps = caps
    puppy.budget = Budget(preset, budget_s)
    puppy.tier = "standoff" if standoff else "boundary"
    puppy.log = [
        f"time: plain {plain.duration:.1f} s -> {puppy.duration:.1f} s (+{puppy.duration - plain.duration:.1f} s of "
        f"the {budget_s:.1f} s '{preset}' budget)",
        *notes,
        *[line for _, dr in directors for line in dr.log],
    ]
    return puppy


def display_names(track):
    """The track with its beats named for the gallery's chips (DISPLAY); returns the track."""
    track.beats = [Label(DISPLAY.get(b.name, b.name), b.t0, b.t1) for b in track.beats]
    return track


@contextmanager
def exaggerated(scales):
    """Make some of puppy_motion's beat sizes bigger for a preview that has to show them: scales is
    {"behaviors.WIND_UP_SHIFT_M": 1.7, ...} (module.constant -> factor). The Director composes and
    validates the bigger beats as usual (they still fit the caps and the clearance floors); its beat
    cache is emptied around the change."""
    import importlib

    pm = _pm()
    old = []
    pm._CACHE.clear()
    try:
        for key, factor in scales.items():
            module, name = key.split(".")
            mod = importlib.import_module(f"puppy_motion.{module}")
            old.append((mod, name, getattr(mod, name)))
            setattr(mod, name, getattr(mod, name) * factor)
        yield
    finally:
        for mod, name, value in old:
            setattr(mod, name, value)
        pm._CACHE.clear()
