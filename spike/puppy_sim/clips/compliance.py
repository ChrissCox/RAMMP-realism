"""Shared by the impedance_* clips: one Puppy Mode track, executed two ways side by side by the
approximate dynamic model (dynamics.py): what the arm would do in position mode (it goes exactly
where it is told) and in the driver's impedance mode (every joint a spring), or under two
impedance profiles. The commanded track is the same on both sides; check holds it, the cameras
follow it, and the subtitle states each beat's on-screen travel on each side (the close-camera
legibility measure) and how far the arm strays from its command."""

from dataclasses import dataclass
from functools import partial

from .. import dynamics, render, scene
from . import PairClip


def _build(source, sides, plant):
    sc = scene.build()
    track = source()
    return tuple(dynamics.achieved(track, side, sc, plant) for side in sides)


def short_name(spec):
    """'position' or the profile's name in quotes, for a subtitle."""
    ctl = dynamics.control(spec)
    return "position" if ctl.mode == dynamics.POSITION else f"'{ctl.gains.name}'"


@dataclass(frozen=True)
class CompliancePair(PairClip):
    """`sides`: two dynamics.control specs, simulated on `plant`. `report`: (beat name, words)
    pairs, the beats whose travel the subtitle states (the largest of that name)."""

    sides: tuple = ("position", "impedance:default")
    plant: dynamics.Plant = dynamics.DEFAULT_PLANT
    report: tuple = ()

    def subtitle_for(self, tracks):
        kin = scene.build(self.run).kin()
        cmd = tracks[0].commanded
        names = [short_name(s) for s in self.sides]
        lines = [self.subtitle]
        for beat, words in self.report:
            px = [beat_px(t, kin, beat) for t in tracks]
            change = 100.0 * (px[1] / px[0] - 1.0)
            lines.append(f"{words}: {names[0]} {px[0]:.0f} px, {names[1]} {px[1]:.0f} px on screen ({change:+.0f} %)")
        err = [float(dynamics.point_error_mm(kin, cmd.q, t.q).max()) for t in tracks]
        lines.append(
            f"the face strays from its command by up to {err[0]:.0f} mm ({names[0]}) / {err[1]:.0f} mm ({names[1]})"
        )
        return lines


def beat_px(track, kin, beat):
    """The largest on-screen travel (px, eye or mouth) of the beats named `beat` on `track`."""
    rows = [r for r in render.legibility(track, kin) if r.beat == beat]
    return max((max(r.eye_px, r.mouth_px) for r in rows), default=float("nan"))


def compliance_pair(name, title, subtitle, camera, source, sides, headers, report, plant=dynamics.DEFAULT_PLANT):
    """A CompliancePair whose build() executes source() (a composed track) under both sides."""
    return CompliancePair(
        name=name,
        title=title,
        subtitle=subtitle,
        camera=camera,
        build=partial(_build, source, tuple(sides), plant),
        headers=tuple(headers),
        sides=tuple(sides),
        plant=plant,
        report=tuple(report),
    )


def ratio(track, kin, beat):
    """Achieved over commanded on-screen travel of `beat` on an achieved track."""
    return beat_px(track, kin, beat) / beat_px(track.commanded, kin, beat)
