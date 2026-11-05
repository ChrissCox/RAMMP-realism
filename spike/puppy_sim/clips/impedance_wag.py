"""The happy wag at home (wag_compare's, inside today's envelope), executed two ways side by side by
the dynamic model: position mode (Kortex's own servo: the arm goes where it is told) and the
driver's joint impedance mode with the session default gains (medium: every joint a spring of
80 or 30 N m/rad, damping ratio 0.5). The wag is sent as a trajectory with its velocities and
accelerations, as the task's own moves are, so the driver feeds them forward; see dynamics.py for
the model and what it assumes."""

from .. import layer
from .compliance import compliance_pair
from .wag_compare import CAMERA, CHIP, wag_track


def source():
    return wag_track(layer.DEFAULT_CAPS)


CLIP = compliance_pair(
    name="impedance_wag",
    title="Stiff or springy joints: the same happy wag",
    subtitle="left: position mode, the arm goes exactly where it is told; right: impedance mode with the "
    "driver's default gains, every joint a spring (simulated)",
    camera=CAMERA,
    source=source,
    sides=("position", "impedance:default"),
    headers=("Position mode (stiff)", "Impedance 'default' (springy)"),
    report=((CHIP, "the wag"),),
)
