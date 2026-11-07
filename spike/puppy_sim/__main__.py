"""CLI (offline only, nothing talks to a robot):

python -m puppy_sim list
python -m puppy_sim render <clips|all> [--fast]   media/<clip>.mp4, .gif, _sheet.png
                                                  (--fast: half size, no GIF, media/_work/fast/)
       [--control impedance:default]              ... as the arm would execute it (simulated dynamics),
                                                  into media/<clip>__<control>.*
python -m puppy_sim check <clips|all>             safety checks; exits 1 on a violation
python -m puppy_sim sheet <clips|all>             just the contact sheets (quick look)
python -m puppy_sim peaks <clips|all>             media/<clip>_peaks.png and each beat's legibility
python -m puppy_sim compliance <clips|all>        each beat's legibility as commanded and as the arm
       [--control C ...]                          would execute it (default: position and impedance
                                                  soft/default/stiff), tracking error, PRECISION deviation
Simulated execution options (render --control, compliance): --positions-only (no velocity and
acceleration feedforward), --positions-velocities (velocities, no accelerations), --armature A (kg m^2), --friction N (N m, one value or 7 comma-separated),
--payload KG (a held container).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from . import checks, dynamics, envelope, render, runs, scene
from .clips import registry, select

MAX_WORKERS = 6  # clips render in parallel processes, each with its own GL context


def _parallel(fn, names):
    """fn(name) for each clip name, in worker processes when there is more than one."""
    if len(names) == 1:
        return [fn(names[0])]
    with ProcessPoolExecutor(max_workers=min(len(names), MAX_WORKERS, os.cpu_count() or 1)) as pool:
        return list(pool.map(fn, names))


def cmd_list(_args):
    for clip in registry().values():
        print(f"{clip.name:20s} {clip.camera:12s} {clip.title}: {clip.subtitle}")


def _plant(args):
    """The simulated arm from the CLI options (the defaults: dynamics.DEFAULT_PLANT)."""
    base = dynamics.DEFAULT_PLANT
    friction = base.friction_nm
    if args.friction is not None:
        values = tuple(float(v) for v in args.friction.split(","))
        friction = values[0] if len(values) == 1 else values
    armature = base.armature if args.armature is None else args.armature
    payload = base.payload_kg if args.payload is None else args.payload
    return dynamics.Plant(armature=armature, payload_kg=payload, friction_nm=friction)


def _control(spec, args):
    ff = dynamics.VELOCITIES if args.positions_velocities else not args.positions_only
    return dynamics.control(spec, feedforward=ff)


def cmd_render(args):
    names = [c.name for c in select(args.clips)]
    ctl = None if args.control is None else _control(args.control, args)
    worker = partial(render.render_named, fast=args.fast, control=ctl, plant=_plant(args) if ctl else None)
    for name, outputs, seconds in _parallel(worker, names):
        sizes = "  ".join(f"{k} {mb:.2f} MB" for k, (_, mb) in outputs.items())
        print(f"{name:20s} {seconds:6.1f} s  {sizes}")
        for path, _ in outputs.values():
            print(f"    {path}")


def cmd_sheet(args):
    for name, path in _parallel(render.sheet_named, [c.name for c in select(args.clips)]):
        print(f"{name:20s} {path}")


def cmd_peaks(args):
    clips = select(args.clips)
    for clip, (name, path, rows) in zip(clips, _parallel(render.peaks_named, [c.name for c in clips])):
        print(f"{name}: {path}  (on-screen travel in the '{clip.camera}' camera at 1280x720)")
        print(render.format_legibility(rows))


def cmd_compliance(args):
    controls = [_control(c, args) for c in (args.control or dynamics.COMPARE)]
    plant = _plant(args)
    print(f"simulated arm: {plant.describe()}; legibility in the '{render.LEGIBILITY_CAMERA}' camera at 1280x720")
    names = [c.name for c in select(args.clips)]
    worker = partial(dynamics.compliance_named, controls=controls, plant=plant)
    rows = [row for result in _parallel(worker, names) for row in result]
    if rows:
        print(dynamics.format_responses(rows))


def cmd_check(args):
    k = scene.build().kin()
    for name in sorted(p.stem for p in runs.RUNS_DIR.glob("*.json")):
        pos_max, pos_mean, rot_max = checks.fk_residuals(runs.load_run(name), k)
        print(
            f"FK vs /ee_state {name}: arm_end_effector_link in arm_mount, "
            f"max {pos_max * 1000:.2f} mm (mean {pos_mean * 1000:.3f}), max {rot_max:.3f} deg"
        )
    for line in envelope.summary():
        print(line)
    rows = [row for result in _parallel(checks.check_named, [c.name for c in select(args.clips)]) for row in result]
    print(checks.format_table([r[:4] for r in rows]))
    for name, label, _, results, log in rows:
        for c in results:
            if not c.ok or c.detail:
                print(f"  {name}/{label} {c.name}: {c.shown} {c.detail}".rstrip())
        for line in log:
            print(f"  {name}/{label} fit: {line}")
    return 0 if all(checks.passed(r[3]) for r in rows) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m puppy_sim", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the registered clips")
    for name, text in (
        ("render", "render clips to media/"),
        ("check", "run the safety checks"),
        ("sheet", "render just the contact sheets"),
        ("peaks", "per-beat peak frames and legibility"),
        ("compliance", "legibility and tracking as the arm would execute each clip (simulated)"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("clips", nargs="*", default=["all"])
        if name == "render":
            p.add_argument("--fast", action="store_true", help="half size, no GIF, into media/_work/fast/")
            p.add_argument("--control", help="simulate the execution: 'position' or 'impedance[:soft|default|...]'")
        if name == "compliance":
            p.add_argument("--control", action="append", help="a control to compare (repeatable)")
        if name in ("render", "compliance"):
            p.add_argument("--positions-only", action="store_true", help="no velocity/acceleration feedforward")
            p.add_argument(
                "--positions-velocities",
                action="store_true",
                help="positions + central-difference velocities, no accelerations (what execute_joint_trajectory sends)",
            )
            p.add_argument("--armature", type=float, help="joint armature of the simulated arm, kg m^2")
            p.add_argument("--friction", help="Coulomb joint friction, N m: one value or 7 comma-separated")
            p.add_argument("--payload", type=float, help="mass of a held container, kg")
    args = parser.parse_args(argv)
    handler = {
        "list": cmd_list,
        "render": cmd_render,
        "check": cmd_check,
        "sheet": cmd_sheet,
        "peaks": cmd_peaks,
        "compliance": cmd_compliance,
    }
    return handler[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
