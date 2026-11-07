"""The bridge from the preview's tracks to puppy_motion: the gallery clips build their expressive
motion with puppy_motion's behaviors and compose() and get back a preview Track (timeline.Track),
which the renderer, the checks and the driver simulation already understand.

Env holds what puppy_motion needs besides the track: the Chain of the run's URDF, the geometry
Scene (the mannequin), the Targets (the person's head, the container) and the preview scene/kin that
still scripts the plain task motions. context() makes a puppy_motion Context on a preview track;
compose() composes contributions on it and returns the preview Track."""

from dataclasses import dataclass, replace

from . import scene as scene_mod
from .timeline import Label


@dataclass(frozen=True, eq=False)
class Env:
    sc: object  # the preview scene (scripts the plain moves; its kin checks the result)
    kin: object  # sc.kin()
    chain: object  # puppy_motion Chain
    person: object  # puppy_motion geometry Scene
    targets: object  # puppy_motion Targets

    @property
    def container(self):
        return self.targets.container

    @property
    def person_head(self):
        return self.targets.person_head


_ENV = {}


def env(run_name=scene_mod.REFERENCE_RUN):
    """The Env of a run's robot (cached; the preview scene is built once)."""
    if run_name not in _ENV:
        from puppy_motion.behaviors import Targets

        sc = scene_mod.build()
        person = scene_mod.person_scene()
        targets = Targets(person_head=person.person_head(), container=sc.placement.center.copy())
        _ENV[run_name] = Env(sc, sc.kin(), scene_mod.motion_chain(run_name), person, targets)
    return _ENV[run_name]


def to_lib(track):
    """The puppy_motion Track of a preview track (same samples; labels are not carried)."""
    from puppy_motion.track import Track as PmTrack

    return PmTrack(
        q=track.q.copy(), grip=track.grip.copy(), zone=track.zone.copy(), held=track.held.copy(), tau=track.tau.copy()
    )


def context(track, e=None, caps="default", seed=0):
    """A puppy_motion behaviors.Context on a preview track."""
    from puppy_motion.behaviors import Context

    e = e or env()
    return Context.of(to_lib(track), e.chain, e.person, e.targets, seed, caps)


def compose(track, parts, e=None, caps="default"):
    """The Puppy Mode track: `track` (a preview Track) plus puppy_motion contributions `parts`,
    composed by puppy_motion.compose.compose under the cap preset `caps`."""
    from puppy_motion import compose as pm_compose

    e = e or env()
    out = pm_compose.compose(to_lib(track), parts, e.chain, e.person, caps)
    beats = sorted((Label(lb.name, lb.t0, lb.t1) for c in parts for lb in c.labels), key=lambda b: b.t0)
    return replace(
        track,
        q=out.q,
        grip=out.grip,
        zone=track.zone.copy(),
        held=track.held.copy(),
        tau=track.tau.copy(),
        phases=list(track.phases),
        beats=beats,
        func=track,
        gain=out.gain,
        log=list(out.log),
        caps=caps,
        source=None,
        budget=None,
        tier="",
        commanded=None,
        control="",
        spliced=None,
    )

