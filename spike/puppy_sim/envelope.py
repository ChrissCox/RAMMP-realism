"""The motion envelope RAMMP's own task already uses, measured from the recorded runs: per joint
speed and acceleration, and the speed of the head's eye and tool points, over the FREE samples
where the arm moves, on the functional tracks (resampled to 50 Hz and smoothed as runs.py does).
Puppy Mode's FREE caps (layer.CAP_PRESETS) stay inside it: a test holds them to it, and `check`
prints both.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import layer, runs, scene
from .timeline import DT, FREE, NEAR_USER, joint_speed

MOVING_RAD_S = 0.05  # a sample counts as moving when a joint is faster than this
ACCEL_PERCENTILE = 99
POINT_PERCENTILE = 95


@dataclass(frozen=True)
class Envelope:
    """Per joint maxima (7,) of speed (rad/s) and acceleration (rad/s^2), the acceleration's
    ACCEL_PERCENTILE, and the eye and tool points' peak and POINT_PERCENTILE speeds (m/s)."""

    joint_speed_max: np.ndarray
    joint_accel_max: np.ndarray
    joint_accel_pct: np.ndarray
    eye_speed_max: float
    eye_speed_pct: float
    tool_speed_max: float
    tool_speed_pct: float
    runs: int
    samples: int


def _point_speed(kin, q, site):
    return np.linalg.norm(np.gradient(kin.fk_track(q, site)[0], DT, axis=0), axis=1)


@lru_cache(maxsize=1)
def measured():
    """The envelope of every recorded run in runs.RUNS_DIR."""
    kin = scene.build().kin()
    v_all, a_all, eye, tool = [], [], [], []
    names = sorted(p.stem for p in runs.RUNS_DIR.glob("*.json"))
    for name in names:
        track = runs.functional_track(runs.load_run(name), kin)
        v = np.gradient(track.q, DT, axis=0)
        a = np.gradient(v, DT, axis=0)
        keep = (track.zone == FREE) & (joint_speed(track) > MOVING_RAD_S)
        v_all.append(np.abs(v[keep]))
        a_all.append(np.abs(a[keep]))
        eye.append(_point_speed(kin, track.q, "eye")[keep])
        tool.append(_point_speed(kin, track.q, "tool")[keep])
    v_all, a_all, eye, tool = (np.concatenate(x) for x in (v_all, a_all, eye, tool))
    return Envelope(
        joint_speed_max=v_all.max(axis=0),
        joint_accel_max=a_all.max(axis=0),
        joint_accel_pct=np.percentile(a_all, ACCEL_PERCENTILE, axis=0),
        eye_speed_max=float(eye.max()),
        eye_speed_pct=float(np.percentile(eye, POINT_PERCENTILE)),
        tool_speed_max=float(tool.max()),
        tool_speed_pct=float(np.percentile(tool, POINT_PERCENTILE)),
        runs=len(names),
        samples=len(eye),
    )


def _span(values, fmt="{:.2f}"):
    return (
        f"{fmt.format(min(values))}"
        if min(values) == max(values)
        else f"{fmt.format(min(values))}-{fmt.format(max(values))}"
    )


def summary(caps=layer.DEFAULT_CAPS):
    """Two lines: the measured envelope and the preset's caps by zone."""
    env = measured()
    preset = layer.CAP_PRESETS[caps]
    task = (
        f"task envelope ({env.runs} runs, {env.samples} FREE samples while moving): joints "
        f"{_span(env.joint_speed_max)} rad/s, {_span(env.joint_accel_max)} rad/s^2 "
        f"(p{ACCEL_PERCENTILE} {_span(env.joint_accel_pct)}); eye {env.eye_speed_max:.2f} m/s "
        f"(p{POINT_PERCENTILE} {env.eye_speed_pct:.2f}), tool {env.tool_speed_max:.2f} m/s "
        f"(p{POINT_PERCENTILE} {env.tool_speed_pct:.2f})"
    )
    tiers = "; ".join(
        f"{name} {_span(c.joint_speed)} rad/s, {_span(c.joint_accel, '{:.1f}')} rad/s^2, head {c.head_speed:.2f} m/s"
        for name, c in (("FREE", preset[FREE]), ("NEAR_USER", preset[NEAR_USER]))
    )
    return [task, f"expressive caps '{caps}': {tiers}"]
