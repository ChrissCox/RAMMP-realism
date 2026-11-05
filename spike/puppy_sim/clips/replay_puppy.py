"""The real grab_container run through the director: the same path as replay_plain, with Puppy
Mode's beats in its free moments and only timing changes in its precise ones."""

from .. import director, runs, scene
from . import Clip


def build():
    sc = scene.build()
    _, puppy = director.direct(runs.load_run(scene.REFERENCE_RUN), sc)
    return puppy


CLIP = Clip(
    name="replay_puppy",
    title="Grab container with Puppy Mode",
    subtitle=f"{scene.REFERENCE_RUN} through the director  |  same path, beats in the free moments",
    camera="wide",
    build=build,
    mood="CALM",
)
