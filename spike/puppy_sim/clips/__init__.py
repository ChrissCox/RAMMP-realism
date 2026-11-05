"""The clip registry. Each module in this package exposes CLIP = Clip(...); build() returns one
Track (a single view) or a (plain, puppy) pair (side by side, same scene, same camera)."""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from ..scene import REFERENCE_RUN


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
