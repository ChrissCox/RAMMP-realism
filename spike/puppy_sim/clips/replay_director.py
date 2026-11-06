"""The real grab_container run, plain and through puppy_motion's Director side by side: the events a
capability would report (phases, stops, the terminal status) go to Director.on_event with the
recorded posture, and each beat it returns plays as a hold in the arm's stop. The same beats, rules,
budget (demo, 6 s) and validator as the real hook; the MuJoCo model only renders.
"""

from .. import replay, scene
from . import PairClip


def build():
    return replay.direct_run(scene.REFERENCE_RUN, "demo")


CLIP = PairClip(
    name="replay_director",
    title="Grab container: plain vs Puppy Mode (puppy_motion's director)",
    subtitle="a real run; the director decides event by event within 6 s, beats only while the arm is stopped",
    camera="counter",
    build=build,
    mood="CALM",
)
