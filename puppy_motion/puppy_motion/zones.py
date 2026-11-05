"""Zone lookup: which expressive-motion zone a capability's (phase, message) pair belongs to.

The table lives in data/zones.yaml. A pair the table does not know is PRECISION (fail-safe: a
renamed phase or a new message may be a precise step).
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml

from .track import Zone

__all__ = ["Zone", "ZoneEntry", "zone_of"]


@dataclass(frozen=True)
class ZoneEntry:
    """The zone of a phase and whether the motion it covers ends at a precise pose (the move that
    ends at the grasp or pre-press pose), so the zone after it must be reached with the arm settled."""

    zone: Zone
    precise_end: bool = False


UNKNOWN = ZoneEntry(Zone.PRECISION, False)


@lru_cache(maxsize=1)
def _load():
    """({capability: {(phase, message): ZoneEntry}}, {capability: {phase: ZoneEntry}}): exact pairs,
    and the phases whose message is dynamic (`any_message: true`)."""
    text = resources.files("puppy_motion").joinpath("data/zones.yaml").read_text(encoding="utf-8")
    exact, any_message = {}, {}
    for capability, rows in yaml.safe_load(text).items():
        exact[capability], any_message[capability] = {}, {}
        for row in rows:
            entry = ZoneEntry(Zone[row["zone"]], bool(row.get("precise_end", False)))
            if row.get("any_message", False):
                any_message[capability][row["phase"]] = entry
            else:
                exact[capability][(row["phase"], row.get("message", ""))] = entry
    return exact, any_message


def zone_of(capability: str, phase: str, message: str) -> ZoneEntry:
    """The table's entry for a (phase, message); PRECISION (not precise-ended) when unknown."""
    exact, any_message = _load()
    entry = exact.get(capability, {}).get((phase, message))
    if entry is None:
        entry = any_message.get(capability, {}).get(phase)
    return entry if entry is not None else UNKNOWN
