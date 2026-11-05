"""CLI (offline only, nothing talks to a robot):

python -m puppy_sim list
python -m puppy_sim render <clips|all> [--fast]   media/<clip>.mp4, .gif, _sheet.png
                                                  (--fast: half size, no GIF, media/_work/fast/)
python -m puppy_sim check <clips|all>             safety checks; exits 1 on a violation
python -m puppy_sim sheet <clips|all>             just the contact sheets (quick look)
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from . import checks, render, runs, scene
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
        print(f"{clip.name:20s} {clip.camera:6s} {clip.title}: {clip.subtitle}")


def cmd_render(args):
    names = [c.name for c in select(args.clips)]
    for name, outputs, seconds in _parallel(partial(render.render_named, fast=args.fast), names):
        sizes = "  ".join(f"{k} {mb:.2f} MB" for k, (_, mb) in outputs.items())
        print(f"{name:20s} {seconds:6.1f} s  {sizes}")
        for path, _ in outputs.values():
            print(f"    {path}")


def cmd_sheet(args):
    for name, path in _parallel(render.sheet_named, [c.name for c in select(args.clips)]):
        print(f"{name:20s} {path}")


def cmd_check(args):
    k = scene.build().kin()
    for name in sorted(p.stem for p in runs.RUNS_DIR.glob("*.json")):
        pos_max, pos_mean, rot_max = checks.fk_residuals(runs.load_run(name), k)
        print(
            f"FK vs /ee_state {name}: arm_end_effector_link in arm_mount, "
            f"max {pos_max * 1000:.2f} mm (mean {pos_mean * 1000:.3f}), max {rot_max:.3f} deg"
        )
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
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("clips", nargs="*", default=["all"])
        if name == "render":
            p.add_argument("--fast", action="store_true", help="half size, no GIF, into media/_work/fast/")
    args = parser.parse_args(argv)
    handler = {"list": cmd_list, "render": cmd_render, "check": cmd_check, "sheet": cmd_sheet}[args.cmd]
    return handler(args) or 0


if __name__ == "__main__":
    sys.exit(main())
