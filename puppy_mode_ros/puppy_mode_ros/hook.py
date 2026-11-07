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
import math

from capability_engine.driver_state import ControlTaken, EStopped
from capability_engine.engine import Cancelled, EngineError
from rclpy.exceptions import ParameterNotDeclaredException

PASS = (Cancelled, EStopped, ControlTaken)

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


GRIPPER_OPEN_MAX = 0.05  # /gripper_state position (0 open .. 1 closed) above this: something is held


def held_source(engine):
    """A zero-argument callable: whether the container rides in the gripper. Fail-safe: True when
    the gripper state is missing, stale (the arm interface raises), not present or NaN."""

    def held():
        try:
            grip = engine.arm.gripper_state()
            position = float(grip.position)
            if not getattr(grip, "present", True) or not math.isfinite(position):
                return True
            return position > GRIPPER_OPEN_MAX
        except PASS:
            raise
        except Exception:  # noqa: BLE001  (no reading is a held container: muted beats)
            return True

    return held


def _param(engine, name):
    return engine.get_parameter(name).value


def _scene(text):
    """The person zone: "flatbot" is the chair scene, anything else the Scene.from_param JSON."""
    if text == "flatbot":
        from puppy_motion.flatbot import chair_scene

        return chair_scene()
    from puppy_motion.geometry import Scene

    return Scene.from_param(text)


def _targets(scene):
    """Targets for `scene`. The person's head is its highest sphere; a scene of boxes only (the
    flatbot chair layers) has none, so it is the centre of the person's highest box."""
    from puppy_motion.behaviors import Targets

    try:
        return Targets.from_scene(scene)
    except ValueError:
        boxes = [b for b in scene.person if hasattr(b, "half")]
        if not boxes:
            raise
        return Targets(person_head=max(boxes, key=lambda b: b.center[2]).center.copy())


def _capability(engine, rules):
    """The key of `rules` for this engine: its id, else the root module of its class."""
    name = engine.get_name()
    if name in rules:
        return name
    root = type(engine).__module__.split(".")[0]
    return root if root in rules else None


def _setup(engine, log):
    """(capability, director, chain, scene) for this goal, or None (logged) to run without Puppy
    Mode. Cancelled, EStopped and ControlTaken propagate."""
    from puppy_motion.director import Director, load_rules
    from puppy_motion.urdf import Chain

    try:
        text = _param(engine, "puppy.person_zone")
        if not isinstance(text, str) or not text.strip():
            log("puppy mode off: no person zone")
            return None
        capability = _capability(engine, load_rules())
        if capability is None:
            log(f"puppy mode off: {engine.get_name()!r} has no rules")
            return None
        try:
            scene = _scene(text.strip())
        except ValueError as error:
            log(f"puppy mode off: {error}")
            return None
        chain = Chain.from_urdf(engine.arm.robot_description())
        director = Director(
            capability,
            chain,
            scene,
            _targets(scene),
            preset=_param(engine, "puppy.preset"),
            lock_on=_param(engine, "puppy.lock_on"),
        )
    except PASS:
        raise
    except Exception as error:  # noqa: BLE001  (bad preset, no URDF, ...: the task runs as it was)
        log(f"puppy mode off: {type(error).__name__}: {error}")
        return None
    return capability, director, chain, scene


def run(engine, action_id, params, ctx, original):
    """Puppy Mode on, for one goal: the original execute, with reactions at its boundaries.

    The TaskRequested beats are deferred: they play just before the task's first move through
    engine.arm, when its own checks have passed (a task that never moves through engine.arm, like
    box_opening, never plays them). The Succeeded beats play after `original` returns, only if the task made a move call through the
    proxy arm or its payload reports `executed` True (a dry run moves nothing); on a
    recoverable EngineError, once the task has moved, the Failed(True) beats play and the same error is re-raised. Anything
    else propagates with no beats. The ending beats never change the task's result: a Cancelled,
    EStopped, ControlTaken or any other exception from them is logged and the payload is returned
    (or the original error re-raised) unchanged. During `original`, engine.arm is a PuppyArm and ctx
    a PuppyCtx; both are restored in a finally."""
    log = engine.get_logger().info
    try:
        from puppy_mode_ros.player import Player
        from puppy_mode_ros.proxies import PuppyArm, PuppyCtx
        from puppy_motion.director import Failed, Succeeded, TaskRequested
    except ImportError as error:  # puppy_motion or the arm library is not installed
        log(f"puppy mode off: {error}")
        return original(engine, action_id, params, ctx)
    setup = _setup(engine, log)
    if setup is None:
        return original(engine, action_id, params, ctx)
    capability, director, chain, scene = setup
    real_arm = engine.arm
    try:  # Puppy Mode must never fail a goal that would otherwise run: any error here runs the task as it was
        player = Player(
            real_arm, ctx, director, chain, scene, capability=capability, held=held_source(engine), log=log
        )
        log(f"puppy mode on: {capability}, preset {director.preset}")
        player.defer(TaskRequested())
    except PASS:
        raise
    except Exception as error:  # noqa: BLE001
        log(f"puppy mode off: {type(error).__name__}: {error}")
        return original(engine, action_id, params, ctx)
    engine.arm = PuppyArm(real_arm, player)

    def ending(event):
        try:
            player.emit(event)
        except BaseException as error:  # noqa: BLE001  (the task's own result stands)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            log(f"puppy: {type(error).__name__} during the {type(event).__name__} beats; the task's own result stands")

    try:
        payload = original(engine, action_id, params, PuppyCtx(ctx, player.emit))
    except EngineError as error:
        engine.arm = real_arm
        if error.recoverable and player.moved:  # a goal refused before any move moves nothing
            ending(Failed(True))
        raise error
    finally:
        engine.arm = real_arm
    if player.moved or (isinstance(payload, dict) and payload.get("executed") is True):
        ending(Succeeded())
    else:  # a dry run: nothing moved the arm, so nothing celebrates
        log("puppy: no Succeeded beats: the task moved nothing and did not report executed")
    return payload


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
