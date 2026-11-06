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

from dataclasses import dataclass

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
    "needs grab_container to split its reach into two calls (view -> 10 cm short, then -> grasp): "
    "a capability change"
)
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
    """(t_split, t_close): where a split approach would stop (the last free sample before the final
    approach, the tool a little over STANDOFF_M from the grasp pose) and where the gripper closes."""
    run = _run_of(track)
    approach = next(p for p in track.phases if p.name == "approach")
    close = next(p for p in track.phases if p.name == "close")
    i0, i1 = int(round(approach.t0 / DT)), int(round(approach.t1 / DT))
    precise = np.flatnonzero(track.zone[i0 : i1 + 1] == PRECISION)
    first = i0 + (int(precise[0]) if precise.size else i1 - i0)
    ramp = timeline._ramp_steps(STANDOFF_RAMP_MIN_S) // 2
    i_split = max(i0 + 1, first - ramp - SPLIT_MARGIN)
    return i_split * DT, close.t0


def events_from_run(track, chain, lock_on="view"):
    """[(t, Event)] in time order for the functional track of a recorded run: what grab_container
    would have told its director. t is the track's functional time."""
    d = _pm()
    run = _run_of(track)
    t0 = runs.replay_window(run)[0]
    out = []
    if run.feedback:
        out.append((max(run.feedback[0][0] - t0, 0.0), d.TaskRequested()))
    for ts, phase, msg in run.feedback:
        if t0 <= ts < t0 + track.duration:
            out.append((ts - t0, d.Phase(*_today(phase, msg))))
    standoff = lock_on == "standoff"
    if standoff:
        t_split, _ = standoff_split(track, chain)
    for i0, i1, before, after in _stage_runs(track):
        if before != "start":
            out.append((i0 * DT, d.AfterMove()))
        if after != "end":
            entry = _entry_at(run, t0, (i1 + 1) * DT)
            precise = bool(entry is not None and entry.precise_end) and not standoff
            out.append((i1 * DT, d.BeforeMove(precise)))
    if standoff:
        out += [(t_split, d.AfterMove()), (t_split, d.BeforeMove(True))]
    if run.terminal is not None:
        t_term = run.terminal[0] - t0
        out.append((t_term, d.Phase(run.terminal[1], "")))  # the terminal state is the last phase ('done', 'aborted')
        if run.terminal[1] == "done":
            out.append((t_term, d.Succeeded()))
        else:  # aborted / canceled: the goal failed where it stood; a retry is possible
            out.append((t_term, d.Failed(True)))
    # a phase that starts within PHASE_REACH_S after a stop ends was set before the move: it comes first
    ends = [i1 for _, i1, _, after in _stage_runs(track) if after != "end"]

    def key(e):
        i = round(e[0] / DT)
        if isinstance(e[1], d.Phase):
            i = next((i1 for i1 in ends if i1 <= i <= i1 + PHASE_REACH), i)
        return (i, RANK[type(e[1]).__name__])

    out.sort(key=key)
    return out


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


def direct_run(run_name, preset="demo", lock_on="view", caps="default", seed=0):
    """(plain, puppy) for a recorded run: its functional track, and the track the puppy_motion
    Director makes of it (each returned beat is a time-warp hold of its length at its event)."""
    d = _pm()
    chain = scene.motion_chain(run_name)
    person = scene.person_scene()
    run = runs.load_run(run_name)
    kin = scene.MotionKin(chain, person)
    plain = runs.functional_track(run, kin)
    from puppy_motion.behaviors import Targets

    targets = Targets(person_head=person.person_head(), container=scene.build().placement.center.copy())
    director = d.Director(run.capability, chain, person, targets, preset, lock_on, caps, seed)
    accel = _accel_cap(caps)
    standoff = lock_on == "standoff"
    t_split = t_close = None
    if standoff:
        t_split, t_close = standoff_split(plain, chain)
        i_split = int(round(t_split / DT))

    # group the events of one sample, decide each in order, and place one hold per group
    groups, last_end = [], -1
    for t, ev in events_from_run(plain, chain, lock_on):
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
    puppy.name = plain.name
    puppy.caps = caps
    puppy.budget = Budget(preset, d.BUDGETS[preset])
    puppy.tier = "standoff" if standoff else "boundary"
    puppy.log = [
        f"time: plain {plain.duration:.1f} s -> {puppy.duration:.1f} s (+{puppy.duration - plain.duration:.1f} s of "
        f"the {d.BUDGETS[preset]:.1f} s '{preset}' budget)",
        *director.log,
    ]
    return plain, puppy
