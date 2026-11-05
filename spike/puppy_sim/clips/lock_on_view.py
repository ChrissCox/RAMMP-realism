"""The lock-on at the view pose, plain vs Puppy Mode, on the real run: grab_container detects the
container; then, while the gripper opens, Puppy Mode leans toward the container and stares at it;
the approach and the grasp run exactly as recorded. This is what the first robot version can do
(lock_on='view'): the approach (view -> pre-grasp -> grasp) is one driver call, so the stare has to
come before it. Only the lock-on plays here (the director's other beats are left out), seen from
beyond the counter at the container's height, so the face and the container share the frame.
"""

from .. import director, render, runs, scene
from . import PairClip

CAMERA = "lock_on"  # beyond the counter, a little above it, looking up at the face: both eyes, the container below
render.CAMERAS.setdefault(
    CAMERA, render.Camera(lookat=(0.71, -0.20, 0.36), distance=0.55, azimuth=205.0, elevation=6.0, fovy=40.0)
)
WINDOW_S = (4.5, 13.0)  # the plain task's seconds shown: arriving at the view pose to the raise
ONLY = ("lock_on", "stillness")  # the director's beats that play (stillness: the FOCUS label on precise steps)


def directed(lock_on):
    """(plain, puppy) for the reference run's WINDOW_S, only the lock-on playing, in `lock_on` mode."""
    sc = scene.build()
    kin = sc.kin()
    func = runs.functional_track(runs.load_run(scene.REFERENCE_RUN), kin)
    skip = tuple(b for b in director.BEATS if b not in ONLY)
    puppy, _ = director.direct_track(func, "grab_container", sc, kin, lock_on=lock_on, skip=skip)
    return director.excerpt(func, puppy, *WINDOW_S)


def build():
    return directed("view")


CLIP = PairClip(
    name="lock_on_view",
    title="Lock-on: a stare at the container before reaching for it",
    subtitle="after the camera check, while the gripper opens: lean in and stare, then the reach runs as recorded",
    camera=CAMERA,
    build=build,
    mood="CALM",
)
