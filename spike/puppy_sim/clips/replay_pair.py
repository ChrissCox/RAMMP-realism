"""The real grab_container run, plain and through the director side by side: the same path, with
Puppy Mode's beats only where the arm is stopped between moves (the boundary tier), the 'demo'
time budget decided event by event, and the lock-on stare at the view pose before the approach."""

from .. import director, runs, scene
from . import PairClip


def build():
    return director.direct(runs.load_run(scene.REFERENCE_RUN), scene.build())


CLIP = PairClip(
    name="replay_pair",
    title="Grab container: plain vs Puppy Mode",
    subtitle=f"{scene.REFERENCE_RUN}  |  beats only between moves, 'demo' budget, lock-on at the view pose",
    camera="counter",
    build=build,
    mood="CALM",
)
