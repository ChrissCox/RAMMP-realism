"""Every idle expression at the stage pose, side by side at size 0.25 (the page default) and 1.0,
on flatbot's empty chair. Each is requested from puppy_motion's IdleDirector as the playground's
buttons do (request(name, q, held, t)), at seed 0, so what plays here is what check_beat passed."""

import numpy as np

from ..timeline import DT, FREE
from . import PairClip
from .playground_common import (
    FLATBOT,
    LEGEND,
    Assembler,
    cached,
    chain,
    chair,
    points_q,
    stage_pose,
    title_case,
    zone_of,
)

SIZES = (0.25, 1.0)
GAP_S = 0.8
LEAD_S = 1.0


def expression_beats(size, names=None):
    """{name: (points, zone) or None (refused / not built)} at the stage pose, seed 0; the director's
    log says why a beat is missing. Cached on puppy_motion's source."""
    from puppy_motion import idle
    from puppy_motion.behaviors import Targets

    names = tuple(names or idle.EXPRESSIONS)

    def make():
        ch, sc = chain(), chair()
        q = stage_pose(ch, sc)
        director = idle.IdleDirector(ch, sc, Targets(person_head=idle.LOOK_TARGET), size, 0)
        out = {}
        for name in names:
            beat = director.request(name, q, False, 0.0)
            out[name] = None if beat is None else (beat.points, int(beat.zone))
        return out, director.log

    return cached(f"playground_expressions|{size}|{','.join(names)}", make)


def tracks():
    """(track at 0.25, track at 1.0): the same slots on one clock; the shorter side of a slot holds."""
    from puppy_motion import idle

    q = stage_pose()
    built = {s: expression_beats(s) for s in SIZES}
    asm = [Assembler(q) for _ in SIZES]
    for a in asm:
        a.hold(LEAD_S, phase="stage pose")
    for name in idle.EXPRESSIONS:
        rows = []
        for s in SIZES:
            beats, _ = built[s]
            rows.append(None if beats[name] is None else points_q(beats[name][0]))
        length = max([len(r) for r in rows if r is not None] + [1])
        for a, r, s in zip(asm, rows, SIZES):
            chip = title_case(name) + ("" if r is not None else " (refused)")
            if r is None:
                a.hold(length * DT, chip=chip)
            else:
                zone = zone_of(type("B", (), {"zone": built[s][0][name][1]}))
                a.add(r, zone=zone, chip=chip)
                if len(r) < length:
                    a.hold((length - len(r)) * DT, chip=chip)
            a.hold(GAP_S)
    return tuple(a.track("playground_expressions") for a in asm)


def build():
    return tracks()


class ExpressionsClip(PairClip):
    def subtitle_for(self, tracks):
        refused = [
            f"{n} at {s}" for s in SIZES for n, b in expression_beats(s)[0].items() if b is None
        ]
        line = f"every idle expression at the stage pose, seed 0; refused: {', '.join(refused) or 'none'}"
        return [self.subtitle, line]


CLIP = ExpressionsClip(
    name="playground_expressions",
    title="Playground expressions: size 0.25 | 1.0",
    subtitle=f"flatbot's empty chair; left size 0.25 (page default), right size 1.0.  {LEGEND}",
    camera="playground_close",
    build=build,
    run=FLATBOT,
    headers=("size 0.25", "size 1.0"),
)
