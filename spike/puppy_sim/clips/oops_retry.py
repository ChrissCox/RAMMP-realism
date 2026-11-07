"""Oops, and try again: a scripted miss at the counter, Puppy Mode's reactions between the moves,
then a retry that succeeds (illustrative: the retry is the capability's choice; Puppy Mode only adds
the reactions while the arm is stopped).

The task, scripted on the container the reference run grasped: from the pre-grasp standoff, 12 cm
back along the run's approach, the approach (PRECISION) ends MISS_HIGH_M too high, so the lower jaw
passes over the brim and the gripper closes on air (PRECISION). The capability notices (the gripper
closed all the way) and the arm holds where it stopped (FREE, as the zone table has it for
'aborted'); it backs off to a higher standoff, its gaze a little less steep, reopening the
gripper; it approaches again to the run's own grasp pose, closes on the brim and lifts the
container to the run's own raised pose (all PRECISION); it stops, holding.

Puppy Mode, only in those stops, decided by puppy_motion's Director from the events the capability
would report. Where it missed it stays stock still (above the brim nothing may move toward the
container: a reaction there would cross the obstacle floor). Once it has backed off to the retry
standoff (the zone table's 'aborted' phase, then a recoverable failure) it looks around for the container (search), droops with puppy eyes up toward the user (sorry) and
shakes it off; at the retry standoff, the move that follows ends at a precise pose, so it leans in
and stares (lock-on); after the raise a small proud, muted while holding. The reactions go past
the 'demo' budget's 6 s (all of them are shown here: it is lifted for this preview); the subtitle
states what each part adds.
The camera follows the head like close_follow, a little further back and aimed lower, so the
container stays in frame while the arm waits above it.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from .. import lib, render, replay, runs, scene
from ..kin import UP, across_gaze
from ..timeline import DT, FREE, PRECISION, concat, constant_track, joint_move, scripted_track
from . import Clip
from .success_proud import HOLDING_CHIP, min_jerk_grip

# the scripted task
STANDOFF_M = 0.12  # the pre-grasp standoff, back along the run's approach
OPEN_GRIP = 0.45  # half open (56 mm between the fingertips): the lower jaw stays higher
MISS_HIGH_M = 0.055  # the first approach ends this much too high: the lower jaw clears the brim
RETRY_FROM = (0.13, 0.11, 0.05)  # the retry standoff: back along the approach, up, to the left (m)
RETRY_PITCH_RAD = np.radians(-8.0)  # ... its gaze this much less steep (the wrist stays clear of its singularity)
BRIM_GRIP = 0.95  # closed on the brim (the run's own value); on air it closes all the way (1.0)
START_S = 0.3
APPROACH_S = 1.4  # 12 cm: RAMMP's 0.16 m/s at the peak
CLOSE_S = 0.6
MISSED_S = 0.4  # the arm holds where it stopped while the capability decides
BACK_OFF_S = 1.2
RETRY_S = 0.4
RETRY_APPROACH_S = 1.8  # 18 cm, diagonally down to the grasp
RAISE_S = 1.2  # to the run's own raised pose (6 cm up, the gaze level)
END_S = 0.4

BIGGER = {"director.LOCK_LEAN_M": 1.6}  # the lock-on leans in further than the director's, to show
LIFTED_BUDGET_S = 30.0  # the reactions to the miss go past the demo budget's 6 s; all are shown
EDGE_S = 3 * DT  # an event arrives this far into a stop
MISS_PHASES = ("missed", "back off", "retry")  # where the reactions to the miss add their time
CAMERA = "oops_follow"
render.CAMERAS.setdefault(  # close_follow, further back and aimed lower: the container stays in frame
    CAMERA, render.Camera(None, 0.85, 148.0, -12.0, 40.0, head_offset=(0.06, -0.04, -0.08), follow_s=0.5)
)


def functional(sc, kin):
    """The plain task (see the module doc), its zones set per move."""
    run = runs.load_run(scene.REFERENCE_RUN)
    q_grasp = run.q[runs.closing_index(run)]  # the container rides in the gripper from this pose
    p, R = kin.fk(q_grasp, "tool")
    approach = sc.placement.approach
    left = np.cross(UP, approach) / np.linalg.norm(np.cross(UP, approach))
    back, up, side = RETRY_FROM
    eyeing = Rotation.from_rotvec(RETRY_PITCH_RAD * across_gaze(R)).as_matrix() @ R
    q_start = kin.cartesian_path(q_grasp, p - STANDOFF_M * approach, R, APPROACH_S)[0][-1]
    q_miss, _ = kin.cartesian_path(q_start, p + MISS_HIGH_M * UP, R, APPROACH_S)
    q_back, _ = kin.cartesian_path(q_miss[-1], p - back * approach + up * UP + side * left, eyeing, BACK_OFF_S)
    q_in, _ = kin.cartesian_path(q_back[-1], p, R, RETRY_APPROACH_S)
    on_air, on_brim = min_jerk_grip(OPEN_GRIP, 1.0, CLOSE_S), min_jerk_grip(OPEN_GRIP, BRIM_GRIP, CLOSE_S)
    close = scripted_track(np.tile(q_in[-1], (len(on_brim), 1)), on_brim, PRECISION, "close")
    close.held[close.grip > runs.GRIP_HELD] = True
    q_raised = runs.functional_track(run, kin).q[-1]  # the run's own raised pose, holding
    raised = joint_move(q_in[-1], q_raised, RAISE_S, BRIM_GRIP, PRECISION, "raise")
    done = constant_track(q_raised, BRIM_GRIP, END_S, name="done")
    raised.held[:] = done.held[:] = True
    func = concat(
        [
            constant_track(q_start, OPEN_GRIP, START_S, name="standoff"),
            scripted_track(q_miss, OPEN_GRIP, PRECISION, "approach"),
            scripted_track(np.tile(q_miss[-1], (len(on_air), 1)), on_air, PRECISION, "close on air"),
            constant_track(q_miss[-1], 1.0, MISSED_S, name="missed"),
            scripted_track(q_back, min_jerk_grip(1.0, OPEN_GRIP, BACK_OFF_S), FREE, "back off"),
            constant_track(q_back[-1], OPEN_GRIP, RETRY_S, name="retry"),
            scripted_track(q_in, OPEN_GRIP, PRECISION, "approach"),
            close,
            raised,
            done,
        ],
        name="oops, retry",
    )
    runs.mark_near_user(func, kin)
    return func


def build():
    """Puppy Mode through the Director, as the capability would drive it: the miss is an aborted
    phase and a recoverable failure, reported once the arm has backed off to the retry standoff (above
    the brim, where it missed, it stays stock still: the library does not know the container is
    there, and a droop would cross it); the retry's move ends at a precise pose; then the success."""
    from puppy_motion import director as pm

    e = lib.env()
    func = scripted(e)
    standoff, missed, retry, done = replay._stage_runs(func)
    events = replay.sorted_events(
        [
            (retry[0] * DT + EDGE_S, pm.Phase("aborted", "")),
            (retry[0] * DT + EDGE_S, pm.Failed(True)),
            (retry[0] * DT + 2 * EDGE_S, pm.Phase("grasp", "opening")),
            (retry[1] * DT - EDGE_S, pm.BeforeMove(True)),
            (done[0] * DT + EDGE_S, pm.Phase("done", "")),
            (done[0] * DT + EDGE_S, pm.Succeeded()),
        ]
    )
    with replay.exaggerated(BIGGER):
        puppy = replay.direct_scripted(func, events, budget_s=LIFTED_BUDGET_S, e=e)
    for sp, lb in zip(puppy.spliced, puppy.beats):
        if sp.beat.name == "proud":
            puppy.beats[puppy.beats.index(lb)] = replace(lb, name=HOLDING_CHIP)
    return replay.display_names(puppy)


def scripted(e):
    return functional(e.sc, e.kin)


def added_s(track, phases):
    """Seconds the time-warp of a composed track adds while its functional motion is in any of
    `phases` (by name, on the output clock)."""
    func = track.func if track.func is not None else track
    inside = np.zeros(func.n, dtype=bool)
    for p in func.phases:
        if p.name in phases:
            inside |= (func.t >= p.t0) & (func.t < p.t1)
    advance = np.diff(func.tau, prepend=func.tau[0])  # functional time each output sample advances
    return float(np.sum(DT - advance[inside]))


@dataclass(frozen=True)
class OopsClip(Clip):
    """The subtitle plus a measured line: both durations, and what the reactions to the miss and
    the proud each add."""

    def subtitle_for(self, tracks):
        (puppy,) = tracks
        miss, proud = added_s(puppy, MISS_PHASES), added_s(puppy, ("done",))
        return [
            self.subtitle,
            f"plain {puppy.plain_duration:.1f} s  |  Puppy Mode {puppy.duration:.1f} s: the reactions to the miss add "
            f"{miss:.1f} s (past the 'demo' budget's 6 s; all shown), the proud {proud:.1f} s",
        ]


CLIP = OopsClip(
    name="oops_retry",
    title="Oops! Missed, and try again",
    subtitle="Illustrative: the retry is the capability's choice; Puppy Mode only adds the reactions between moves.",
    camera=CAMERA,
    build=build,
)
