"""Run a capability engine with the Puppy Mode hook installed.

    python3 -m puppy_mode_ros.launch <module>:<EngineClass> <module>:<main> [engine args...]

Imports both modules, calls hook.install on the class, then calls the engine's own main with the
remaining arguments (sys.argv is set to them, so a main that reads sys.argv sees its own).
"""

import importlib
import sys

from puppy_mode_ros import hook


def resolve(spec):
    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr:
        raise SystemExit(f"expected <module>:<name>, got {spec!r}")
    return getattr(importlib.import_module(module), attr)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 2:
        raise SystemExit(__doc__)
    engine_cls, engine_main, rest = resolve(argv[0]), resolve(argv[1]), argv[2:]
    hook.install(engine_cls)
    sys.argv = [argv[1], *rest]
    return engine_main()


if __name__ == "__main__":
    sys.exit(main())
