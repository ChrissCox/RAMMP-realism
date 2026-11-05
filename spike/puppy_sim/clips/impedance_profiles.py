"""success_proud's Puppy Mode track (the muted proud while holding the container, then the put-back
and the proud with a wag and chomps empty-handed), executed under two of the driver's impedance
gain profiles side by side by the dynamic model: 'soft' (springs of 40 or 15 N m/rad, damping
ratio 0.4) and 'stiff' (160 or 60 N m/rad, 0.7). While it holds the container its weight, which
the driver does not model, bends the soft springs about four times as far as the stiff ones."""

from .compliance import compliance_pair
from .success_proud import EMPTY_CHIP, HOLDING_CHIP
from .success_proud import build as source

CLIP = compliance_pair(
    name="impedance_profiles",
    title="Soft or stiff springs: the proud look",
    subtitle="both sides in impedance mode, every joint a spring: the driver's 'soft' gains (left) and its "
    "'stiff' gains (right), simulated",
    camera="close_follow",
    source=source,
    sides=("impedance:soft", "impedance:stiff"),
    headers=("Impedance 'soft'", "Impedance 'stiff'"),
    report=((HOLDING_CHIP, "proud while holding"), (EMPTY_CHIP, "proud empty-handed")),
)
