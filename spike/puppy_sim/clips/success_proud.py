"""Proud of itself, both ways, in one pick-and-put-back at the counter (the camera following the
head, the user behind it).

(b) Holding: the real run grab_container_20261029_132745 from its raise to its end. When it
succeeds, holding the container, puppy_motion's Director plays its muted proud: a level look at the
user (a pan only, rising as it turns), no tilt, no wag, no chomp, so the container never tips.
(a) Empty-handed: the arm then sets the container back down, lets go and backs out (scripted, all
precise), and moves to a ready pose with the face upright. There the capability reports another
success and the proud is free: a bigger look at the user, rising as it turns, and chomps
of the open gripper while it holds the look.
Both are decided by the Director from the events a capability would report (a success after each
task), play only while the arm is stopped between moves, and stay within the 'demo' budget of
their task (6 s each); every move plays exactly as recorded or scripted (replay.direct_scripted).
"""

from dataclasses import dataclass, replace

import numpy as np

from .. import curves, lib, replay, runs, scene
from ..kin import UP
from ..timeline import (
    DT,
    FREE,
    PRECISION,
    concat,
    constant_track,
    joint_move,
    scripted_track,
)
from . import Clip, durations


def min_jerk_grip(g0, g1, seconds):
    """(N,) a gripper command easing from g0 to g1 over `seconds`."""
    return g0 + (g1 - g0) * curves.min_jerk(np.linspace(0.0, 1.0, int(round(seconds / DT)) + 1))


# ------------------------------------------------------------------------------- the clip
RAISE_FROM_S = 12.6  # the reference run's own clock: the clip starts late in its raise
LOWER_S = 0.9  # set it back down where it was picked up (the run's closing pose)
LET_GO_S = 0.45
OPEN_GRIP = 0.3  # the gripper opens this far to let go
BACK_OUT_M = 0.06  # straight out along the approach, clear of the brim
BACK_OUT_S = 0.6
READY_BACK_M = 0.13  # the ready pose: this far back from the grasp along the approach,
READY_UP_M = 0.04  # this much higher,
READY_DOWN_RAD = 0.0  # the face upright, looking along the approach's heading and this much down
READY_S = 1.7  # the face rolls upright on the way (joint 7 turns about 1.5 rad)
READY_REST_S = 0.4
HOLDING_CHIP, EMPTY_CHIP = "PROUD (holding)", "PROUD"
EDGE_S = 3 * DT  # an event arrives this far into a stop (replay.HOLD_EDGE samples)


def put_back(kin, sc, q_from, grip):
    """Scripted, after the run: set the container down at the run's closing pose (held), open the
    gripper (let go), back straight out along the approach (all PRECISION), then move to the ready
    pose with the face upright (FREE once 0.11 m from the grasp pose) and stop there."""
    run = runs.load_run(scene.REFERENCE_RUN)
    q_close = run.q[runs.closing_index(run)]  # the container's pose in the gripper is defined here
    lower = joint_move(q_from, q_close, LOWER_S, grip, PRECISION, "set down")
    lower.held[:] = True
    n = int(round(LET_GO_S / DT)) + 1
    let_go = scripted_track(np.tile(q_close, (n, 1)), min_jerk_grip(grip, OPEN_GRIP, LET_GO_S), PRECISION, "let go")
    pl = sc.placement
    p, R = kin.fk(q_close, "tool")
    q_out, _ = kin.cartesian_path(q_close, p - BACK_OUT_M * pl.approach, R, BACK_OUT_S)
    out = scripted_track(q_out, OPEN_GRIP, PRECISION, "back out")
    q_ready, _, _ = kin.ik(q_out[-1], *ready_pose(p, pl.approach))
    ready = joint_move(q_out[-1], q_ready, READY_S, OPEN_GRIP, FREE, "to ready")
    tool = kin.fk_track(ready.q, "tool")[0]
    ready.zone[np.linalg.norm(tool - p, axis=1) < runs.FINAL_APPROACH_M] = PRECISION
    rest = constant_track(q_ready, OPEN_GRIP, READY_REST_S, name="ready")
    return [lower, let_go, out, ready, rest]


def ready_pose(p_grasp, approach):
    """(p, R) of the tool at the ready pose: back and up from the grasp, the face upright (ee y up),
    looking along the approach's heading and READY_DOWN_RAD down."""
    heading = np.array([approach[0], approach[1], 0.0]) / np.linalg.norm(approach[:2])
    gaze = np.cos(READY_DOWN_RAD) * heading - np.sin(READY_DOWN_RAD) * UP
    up = UP - (UP @ gaze) * gaze
    up = up / np.linalg.norm(up)
    p = p_grasp - READY_BACK_M * approach + READY_UP_M * UP
    return p, np.column_stack([np.cross(up, gaze), up, gaze])


def functional(sc, kin):
    """The plain task: the run from late in its raise to its end, then the scripted put-back."""
    run = runs.load_run(scene.REFERENCE_RUN)
    start, end = runs.replay_window(run)
    tail = runs.functional_track(run, kin, window=(start + RAISE_FROM_S, end))
    func = concat([tail, *put_back(kin, sc, tail.q[-1], float(tail.grip[-1]))], name="pick and put back")
    runs.mark_near_user(func, kin)
    return func


def build():
    from puppy_motion import director as pm

    e = lib.env()
    func = functional(e.sc, e.kin)
    stops = replay._stage_runs(func)  # (i0, i1, before, after)
    holding = next(s for s in stops if func.held[s[0]])  # the stop where it holds the container, after the run
    empty = stops[-1]  # the ready pose, the last stop
    t_pick, t_put = holding[0] * DT + EDGE_S, empty[0] * DT + EDGE_S
    events = [
        (t_pick, pm.Phase("done", "")),
        (t_pick, pm.Succeeded()),
        (t_put, pm.Phase("done", "")),
        (t_put, pm.Succeeded()),
    ]
    tasks = (("grab_container", 0.0), ("grab_container", holding[1] * DT))  # picking, then putting back
    puppy = replay.direct_scripted(func, events, tasks, e=e)
    for sp, lb in zip(puppy.spliced, puppy.beats):  # the chips name the proud by what it holds
        if sp.beat.name == "proud":
            name = HOLDING_CHIP if sp.held else EMPTY_CHIP
            puppy.beats[puppy.beats.index(lb)] = replace(lb, name=name)
    replay.display_names(puppy)
    return puppy


@dataclass(frozen=True)
class TimedClip(Clip):
    """A single Puppy Mode clip whose subtitle adds both durations and the budget (as a pair's does)."""

    def subtitle_for(self, tracks):
        (puppy,) = tracks
        return [self.subtitle, durations(puppy.source, puppy)]


CLIP = TimedClip(
    name="success_proud",
    title="Proud of itself",
    subtitle="after a success, while the arm waits between moves: holding the container it only looks at you "
    "(level); empty-handed it also chomps",
    camera="close_follow",
    build=build,
)
