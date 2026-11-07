"""The real grab_container run replayed with no personality: what the arm does today, seen from
beyond the counter (the face while it works there, the user behind it)."""

from .. import runs, scene
from . import Clip


def build():
    return runs.functional_track(runs.load_run(scene.REFERENCE_RUN), scene.MotionKin())


CLIP = Clip(
    name="replay_plain",
    title="Grab container: a real run, replayed",
    subtitle=f"{scene.REFERENCE_RUN}  |  no personality: what the arm does today",
    camera="counter",
    build=build,
)
