"""lock_on_view's Puppy Mode track (the lean toward the container and the stare at the view pose,
then the reach, the grip and the raise as recorded), executed two ways side by side by the
dynamic model: position mode and the driver's joint impedance mode with the session default
gains. The driver's model of the arm does not include the container, so once the arm holds it,
its weight is carried by the joint springs and the arm sags (the same in the plain task)."""

from .compliance import compliance_pair
from .lock_on_view import CAMERA, directed


def source():
    return directed("view")[1]


CLIP = compliance_pair(
    name="impedance_lock_on",
    title="Stiff or springy joints: the lock-on stare",
    subtitle="the lean toward the container and the stare, then the reach, the grip and the lift; right: every "
    "joint a spring, so the held container's weight makes the arm sag",
    camera=CAMERA,
    source=source,
    sides=("position", "impedance:default"),
    headers=("Position mode (stiff)", "Impedance 'default' (springy)"),
    report=(("LOCK ON", "the lean and stare"),),
)
