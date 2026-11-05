"""The happy wag of impedance_wag again, on an arm whose joints stick a little: Coulomb friction the
driver's impedance law does not cancel, ASSUMED at 1 N m on the four big joints and 0.5 N m on the
wrist (the low end of the 1-2.5 N m residual the driver measured at rest, which it puts down to
friction plus model error). The menagerie model has no friction, so this is a what-if: the one
case the model finds where compliance clearly shrinks a beat. Position mode (left) is unaffected:
Kortex's servo pushes through friction. Right: the driver's 'soft' gains."""

from .. import dynamics
from .compliance import compliance_pair
from .impedance_wag import source
from .wag_compare import CAMERA, CHIP

FRICTION_NM = (1.0,) * 4 + (0.5,) * 3

CLIP = compliance_pair(
    name="impedance_friction",
    title="If the joints stick a little: the wag on soft springs",
    subtitle="what-if: joint friction the driver does not cancel (assumed 1 N m at the big joints, 0.5 N m at "
    "the wrist); right: impedance with the driver's 'soft' gains",
    camera=CAMERA,
    source=source,
    sides=("position", "impedance:soft"),
    headers=("Position mode (stiff)", "Impedance 'soft', sticky joints"),
    report=((CHIP, "the wag"),),
    plant=dynamics.Plant(friction_nm=FRICTION_NM),
)
