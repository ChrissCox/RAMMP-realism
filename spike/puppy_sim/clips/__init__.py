"""The clip registry. Each module in this package exposes CLIP = Clip(...) or PairClip(...);
build() returns one Track (a single view) or, for a PairClip, a (plain, puppy) pair shown side by
side on the same scene and camera. rest_reel() builds a vocabulary reel at a rest pose."""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import behaviors, layer, runs, scene
from ..scene import REFERENCE_RUN
from ..timeline import Label, constant_track

REST_GRIP = 0.79  # the reference run's gripper while it waited at home
REEL_GAP_S = 0.6  # stillness between a reel's primitives
REEL_MEASURE_S = 200.0  # a reel's primitives are measured on a still track this long


@dataclass(frozen=True)
class Clip:
    """name: file stem in media/; camera: a render.CAMERAS key; mood: the chip shown when no beat
    is active (None: no chip); run: the run whose grasp places the container in the scene."""

    name: str
    title: str
    subtitle: str
    camera: str
    build: Callable
    mood: str | None = None
    run: str = REFERENCE_RUN

    def subtitle_for(self, tracks):
        """The subtitle shown over the built tracks: a line or a list of lines."""
        return self.subtitle


def durations(plain, puppy):
    """'plain 15.1 s | Puppy Mode 20.3 s (+5.2 s; demo budget 6.0 s)': both durations, what Puppy
    Mode adds and, for a directed track, its time budget."""
    added = puppy.duration - plain.duration
    budget = f"; {puppy.budget.name} budget {puppy.budget.seconds:.1f} s" if puppy.budget is not None else ""
    return f"plain {plain.duration:.1f} s  |  Puppy Mode {puppy.duration:.1f} s (+{added:.1f} s{budget})"


@dataclass(frozen=True)
class PairClip(Clip):
    """Plain vs Puppy Mode side by side (1600x720): build() returns (plain, puppy). Both start
    together, the shorter side holds its last frame, each header ('Plain (today)', 'Puppy Mode')
    runs its own clock, and the subtitle adds both durations. headers: other labels for the two
    sides (None: those two)."""

    headers: tuple[str, str] | None = None

    def subtitle_for(self, tracks):
        plain, puppy = tracks
        return [self.subtitle, durations(plain, puppy)]


def labelled(chip, parts, t0):
    """parts with the chip over [t0, their end] (chip None: they keep their own beats)."""
    if chip is None:
        return parts
    end = max(p.span()[1] for p in parts if p.span())
    for p in parts:
        p.beats = []
    parts[0].beats = [Label(chip, t0, end)]
    return parts


def rest_reel(slots, q=None, gap_s=REEL_GAP_S, caps=layer.DEFAULT_CAPS, name="rest at home"):
    """A composed Track playing each (chip, play) of `slots` in turn at rest at joints q (default:
    home), gap_s apart, each chip naming its primitive over its span. play(ctx, t0) returns the
    primitive's [Contribution] from t0 (chip None: they carry their own beats). Each is measured on
    a long still track first, then composed once on the reel's own track."""
    sc = scene.build()
    kin = sc.kin()
    q = np.array(runs.HOME_JOINTS) if q is None else np.asarray(q, dtype=float)
    probe = behaviors.Context.of(constant_track(q, REST_GRIP, REEL_MEASURE_S), sc, kin, caps=caps)
    plan, t = [], gap_s
    for chip, play in slots:
        plan.append((chip, play, t))
        t = max(p.span()[1] for p in play(probe, t) if p.span()) + gap_s
    rest = constant_track(q, REST_GRIP, t, name=name)
    runs.mark_near_user(rest, kin)
    ctx = behaviors.Context.of(rest, sc, kin, caps=caps)
    parts = [c for chip, play, t0 in plan for c in labelled(chip, play(ctx, t0), t0)]
    return layer.compose(rest, parts, kin, caps)


def registry():
    """{name: Clip} for every clip module in this package, sorted by name."""
    clips = {}
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        clip = getattr(module, "CLIP", None)
        if clip is not None:
            clips[clip.name] = clip
    return dict(sorted(clips.items()))


def select(names):
    """Clips for CLI arguments: names, or 'all'."""
    clips = registry()
    if names in (["all"], []):
        return list(clips.values())
    unknown = [n for n in names if n not in clips]
    if unknown:
        raise SystemExit(f"unknown clip(s) {unknown}; known: {list(clips)}")
    return [clips[n] for n in names]
