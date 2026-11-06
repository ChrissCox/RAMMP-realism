"""The Puppy Mode hook point. RAMMP's capability_engine is not patched: install() wraps a
capability subclass's `execute` at class level.

Per goal the wrapper reads the engine's `puppy.enabled` parameter. False (the default): the
original `execute` is called directly, with the same arguments, and nothing else runs. True:
`run(engine, action_id, params, ctx, original)` decides what to do.

Parameters: install() also wraps the class's `__init__` so the engine node declares puppy.enabled
(False), puppy.preset ("subtle"), puppy.person_zone ("") and puppy.lock_on ("view") right after
it is built (so they show in `ros2 param list` and take -p overrides). The goal wrapper declares
any that are still missing on first use too, so a node built before install() works as well.
Both are idempotent and never redeclare a parameter the engine already has.
"""

import functools

PARAM_DEFAULTS = {
    "puppy.enabled": False,
    "puppy.preset": "subtle",
    "puppy.person_zone": "",
    "puppy.lock_on": "view",
}


def declare_params(engine):
    """Declare the puppy.* parameters the engine node does not have yet."""
    for name, default in PARAM_DEFAULTS.items():
        if not engine.has_parameter(name):
            engine.declare_parameter(name, default)


def run(engine, action_id, params, ctx, original):
    """Puppy Mode on, for one goal. Task 13 fills this in; for now it is the original action."""
    return original(engine, action_id, params, ctx)


def install(engine_cls):
    """Wrap engine_cls.__init__ and engine_cls.execute. Returns engine_cls; a second call is a no-op."""
    if engine_cls.__dict__.get("_puppy_installed"):
        return engine_cls
    original_init = engine_cls.__init__
    original_execute = engine_cls.execute

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        declare_params(self)

    @functools.wraps(original_execute)
    def execute(self, action_id, params, ctx):
        declare_params(self)
        if not self.get_parameter("puppy.enabled").value:
            return original_execute(self, action_id, params, ctx)
        return run(self, action_id, params, ctx, original_execute)

    engine_cls.__init__ = __init__
    engine_cls.execute = execute
    engine_cls._puppy_installed = True
    return engine_cls
