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

Puppy Mode, only in those stops: where it missed, it freezes (above the brim nothing may move
toward the container: any reaction there would cross the obstacle floor); at the retry standoff a
double take at the container (look, glance away, snap back: it is still there), a droop with puppy
eyes up toward the user, a shake-off, and a wind-up just before the retry (the retry approach
dips the gaze, so the wind-up rears the head back against it); after the raise a small
proud, muted while holding. The reactions to the miss go past the 'demo' budget's 6 s (all of
them are shown here); the subtitle states what each part adds.

The camera follows the head like close_follow, a little further back and aimed lower, so the
container stays in frame while the arm waits above it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .. import behaviors as B
from .. import render, runs, scene
from ..kin import UP, across_gaze
from ..timeline import DT, FREE, PRECISION, Label, concat, constant_track, joint_move, scripted_track
from . import Clip
from .success_proud import HOLDING_CHIP, min_jerk_grip, proud_level, staged, stop_at

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

# the beats (the sorry as the director's full one: the eyes as the droop arrives, back in one move)
FREEZE_S = 0.5  # stock still where it missed
GLANCE_SPOT = (-0.02, 0.17, 0.0)  # the double take's glance away: a spot on the counter, from the container (m)
TAKE_GAP_S = 0.1  # the sorry starts this long after the double take
SORRY_QUICK = 0.8  # the droop sags this much quicker than its design
EYES_AFTER_S = 0.7  # the sheepish look up at the user starts as the droop arrives
EYES_HOLD_S = 0.3  # held this long; then the eyes come down as the droop lifts
SHAKE_GAP_S = 0.05  # the shake-off starts this long after
SHAKE_CYCLES = 2.0  # the director's full shake is 3
PROUD_LOOK_RAD = np.radians(12.0)  # a small proud (the director's): holding, a level look at the user
CHIPS = {"freeze": "FREEZE", "take": "DOUBLE TAKE", "droop": "SORRY", "eyes": "PUPPY EYES", "shake": "SHAKE IT OFF"}
WIND_CHIP = "WIND UP"
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


def freeze(ctx, t0):
    """Stock still for a beat where the gripper closed on air: the pause is Puppy Mode's (a hold)."""
    return [ctx.empty("freeze", [Label(CHIPS["freeze"], t0, t0 + FREEZE_S)])]


def oops(ctx, t0):
    """A double take at the container (look at it, glance away, snap back with a lean in: it is
    still there), then the sorry."""
    container = ctx.container
    take = B.double_take(ctx, container, container + np.array(GLANCE_SPOT), t0, beat=CHIPS["take"])
    return [take, *sorry(ctx, take.span()[1] + TAKE_GAP_S)]


def sorry(ctx, t0):
    """A droop; as it arrives, the sheepish look up toward the user (puppy eyes), held a moment;
    the eyes come down as the droop lifts; then a shake to shake it off."""
    t_eyes = t0 + EYES_AFTER_S
    turn_s = (B.puppy_eyes(ctx, t_eyes, t_eyes).span()[1] - t_eyes) / 2.0  # up and straight back
    t_back = t_eyes + turn_s + EYES_HOLD_S
    eyes = B.puppy_eyes(ctx, t_eyes, t_back)
    sag = B.droop(ctx, t0, t_back, quick=SORRY_QUICK)
    t_shake = max(eyes.span()[1], sag.span()[1]) + SHAKE_GAP_S
    shake = B.shake_off(ctx, t_shake, cycles=SHAKE_CYCLES)
    chips = [
        Label(CHIPS["droop"], t0, t_eyes),
        Label(CHIPS["eyes"], t_eyes, t_back + turn_s),
        Label(CHIPS["shake"], t_shake, shake.span()[1]),
    ]
    return [sag, eyes, shake, ctx.empty("chips", chips)]


def wind_up(ctx, t1):
    """The wind-up, back at rest at t1 (the stop's end margin, just before the retry approach)."""
    return [B.wind_up(ctx, t1 - B.wind_up_s(ctx, t1, t1), t1, beat=WIND_CHIP)]


def wind_up_length(ctx, t1):
    return B.wind_up_s(ctx, t1, t1)


def build():
    sc = scene.build()
    kin = sc.kin()
    func = functional(sc, kin)
    missed, retry, done = (stop_at(func, phase) for phase in ("missed", "retry", "done"))
    plays = {missed: freeze, retry: oops, done: proud_level(PROUD_LOOK_RAD, HOLDING_CHIP)}
    puppy, _ = staged(func, plays, sc, kin, finales={retry: (wind_up, wind_up_length)})
    return puppy


def added_s(track, phases):
    """Seconds the time-warp of a composed track adds while its functional motion is in any of
    `phases` (by name, on the output clock)."""
    func = track.func
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
