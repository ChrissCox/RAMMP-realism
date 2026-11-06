"""Two minutes of auto mode at the stage pose, on flatbot's empty chair: puppy_motion's
IdleDirector picks a beat (calm, curious, playful moods), the arm plays it, waits the director's
pause, and asks again, as the playground's engine does when the arm has been idle. Seed 0, size 1.0
(the page default is 0.25: this is the most visible case). The chip names each beat."""

from ..timeline import DT, FREE
from . import Clip
from .playground_common import FLATBOT, LEGEND, Assembler, cached, chain, chair, points_q, stage_pose, title_case, zone_of

DURATION_S = 120.0
SIZE = 1.0
SEED = 0


def sequence(duration_s=DURATION_S, size=SIZE, seed=SEED):
    """[(name, mood, points or None, zone, pause_s)] as the director chooses them from the stage
    pose; each beat is composed at the arm's posture where the last one ended. Cached."""
    from puppy_motion import idle
    from puppy_motion.behaviors import Targets

    def make():
        ch, sc = chain(), chair()
        q = stage_pose(ch, sc)
        director = idle.IdleDirector(ch, sc, Targets(person_head=idle.LOOK_TARGET), size, seed)
        out, t = [], 0.0
        while t < duration_s:
            beat, pause = director.next(q, False, t)
            if beat is None:
                out.append((None, director.mood, None, int(FREE), pause))
            else:
                out.append((beat.name, beat.mood, beat.points, int(beat.zone), pause))
                q = beat.points[-1][1]
                t += beat.duration
            t += pause
        return out, director.log

    return cached(f"playground_auto|{duration_s}|{size}|{seed}", make)


def build():
    steps, _ = sequence()
    asm = Assembler(stage_pose())
    asm.hold(1.0, phase="stage pose")
    for name, mood, points, zone, pause in steps:
        if points is not None:
            asm.add(points_q(points), zone=zone, chip=f"{title_case(name)} ({mood})", phase=mood)
        asm.hold(pause)
    return asm.track("playground_auto")


class AutoClip(Clip):
    def subtitle_for(self, tracks):
        steps, log = sequence()
        played = [s[0] for s in steps if s[0]]
        return [
            f"{self.subtitle}  {LEGEND}",
            f"{len(played)} beats in {tracks[0].duration:.0f} s, {len(steps) - len(played)} skipped by the director",
        ]


CLIP = AutoClip(
    name="playground_auto",
    title="Playground auto mode: 2 minutes at the stage pose",
    subtitle="IdleDirector, seed 0, size 1.0; the timeline's labels are the mood.",
    camera="playground_close",
    build=build,
    run=FLATBOT,
)
