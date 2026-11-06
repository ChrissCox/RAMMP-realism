"""The Puppy Mode hook point. RAMMP's capability_engine is not patched: install() wraps a
capability subclass's `execute` at class level.

Per goal the wrapper reads the engine's `puppy.enabled` parameter. False (the default): the
original `execute` is called directly, with the same arguments, and nothing else runs. True:
`run(engine, action_id, params, ctx, original)` decides what to do.

Parameters: install() also wraps the class's `__init__` so the engine node declares puppy.enabled
(False), puppy.preset ("subtle"), puppy.person_zone ("") and puppy.lock_on ("view") once, right
after it is built (so they show in `ros2 param list` and take -p overrides). A parameter the engine
already declares is left alone. A node built before install() has no such parameter: the goal
wrapper then declares them on the first goal (the only lazy path).

The off path is inert: per goal the wrapper does one get_parameter inside try/except. Only a value
that `is True` engages the hook; anything else (False, the string "false", a read error) runs the
original execute, and a read error logs a warning. Both wrappers carry `_puppy_wrapped`, and
install() checks it through getattr, so installing a subclass of an installed class wraps nothing
twice (hook.run runs once per goal).
"""

import functools

from rclpy.exceptions import ParameterNotDeclaredException

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


def _enabled(engine):
    """True only if puppy.enabled is the bool True. Never raises: an error logs and reads False."""
    try:
        try:
            value = engine.get_parameter("puppy.enabled").value
        except ParameterNotDeclaredException:  # a node built before install()
            declare_params(engine)
            value = engine.get_parameter("puppy.enabled").value
        return value is True
    except Exception as error:  # noqa: BLE001 -- the off path must never fail a goal
        try:
            engine.get_logger().warning(f"puppy.enabled unreadable ({error!r}); running without Puppy Mode")
        except Exception:  # noqa: BLE001
            pass
        return False


def install(engine_cls):
    """Wrap engine_cls.__init__ and engine_cls.execute. Returns engine_cls. Anything already
    wrapped (on the class or inherited from an installed base class) is not wrapped again."""
    original_init = engine_cls.__init__
    original_execute = engine_cls.execute

    if not getattr(original_init, "_puppy_wrapped", False):

        @functools.wraps(original_init)
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:  # Puppy Mode must never stop the engine itself from starting
                declare_params(self)
            except Exception as error:  # noqa: BLE001
                try:
                    self.get_logger().warning(f"puppy.* parameters not declared ({error!r}); Puppy Mode stays off")
                except Exception:  # noqa: BLE001
                    pass

        __init__._puppy_wrapped = True
        engine_cls.__init__ = __init__

    if not getattr(original_execute, "_puppy_wrapped", False):

        @functools.wraps(original_execute)
        def execute(self, action_id, params, ctx):
            if not _enabled(self):
                return original_execute(self, action_id, params, ctx)
            return run(self, action_id, params, ctx, original_execute)

        execute._puppy_wrapped = True
        engine_cls.execute = execute
    return engine_cls
