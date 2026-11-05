"""The real grab_container run replayed with no personality: what the arm does today."""

from .. import runs, scene
from . import Clip


def build():
    sc = scene.build()
    return runs.functional_track(runs.load_run(scene.REFERENCE_RUN), sc.kin())


CLIP = Clip(
    name="replay_plain",
    title="Grab container: a real run, replayed",
    subtitle=f"{scene.REFERENCE_RUN}  |  no personality: what the arm does today",
    camera="wide",
    build=build,
)
