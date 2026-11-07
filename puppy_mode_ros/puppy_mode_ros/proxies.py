"""The two proxies a Puppy Mode action wraps around its ctx and its arm.

PuppyCtx(ctx, director_feed): `progress()` forwards to the real ctx first, then feeds a Phase event
to `director_feed(event)`. Every other attribute is the real ctx's.

PuppyArm(arm, player): the move calls (go_to_*, execute_joint_trajectory, move_joints,
move_cartesian) feed BeforeMove to the player, forward UNCHANGED to the real arm, and feed AfterMove
after a normal return. They also record the call's control mode and gains on the player, which the
beats copy (and `director.set_control` gets, guarded): `control_mode=` / `gains=` of the go_to_* calls and execute_joint_trajectory, and for
move_joints / move_cartesian `impedance=` (a dict means IMPEDANCE with those gains, none means
POSITION). `stream()` marks a stream open until its block exits, and no beat plays meanwhile.
Every other attribute is the real arm's.

The TaskRequested beat is deferred (Player.defer): it plays just before the first move call, so a
task that refuses a goal before moving, or never moves through the arm, plays none.

Neither proxy changes what a call returns or raises, or the arguments it forwards. A move that
fails (False, a non-zero driver code, any exception) disables the player and emits no AfterMove.
"""

import contextlib
import functools
import inspect

import numpy as np

from capability_engine.arm import GAIN_KEYS, IMPEDANCE, POSITION
from capability_engine.driver_state import EStopped

from puppy_motion.director import AfterMove, BeforeMove, Phase  # noqa: F401  (BeforeMove: see Player)

MOVE_CALLS = frozenset(
    {
        "go_to_ee_pose",
        "go_to_joint_config",
        "go_to_preset",
        "go_to_ee_poses",
        "go_to_joint_configs",
        "execute_joint_trajectory",
        "move_joints",
        "move_cartesian",
    }
)


class PuppyCtx:
    def __init__(self, ctx, director_feed):
        self._ctx = ctx
        self._feed = director_feed

    def progress(self, phase, pct, message="", pausable=True):
        result = self._ctx.progress(phase, pct, message, pausable)  # forward first
        self._feed(Phase(phase, message))
        return result

    def __getattr__(self, name):
        if name == "_ctx":  # not set yet (copy, pickle): do not recurse
            raise AttributeError(name)
        return getattr(self._ctx, name)


def failed(result):
    """Whether a move call's result says the move failed: False, or a driver result with a code."""
    if result is False:
        return True
    code = getattr(result, "error_code", 0)
    return isinstance(code, (int, np.integer)) and code != 0


def mode_of(call, arm_method, args, kwargs):
    """(control_mode, gains) a move call asks for."""
    try:
        given = dict(inspect.signature(arm_method).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        given = dict(kwargs)
    if call.startswith("move_"):
        impedance = given.get("impedance")
        if not impedance:
            return POSITION, None
        try:
            return IMPEDANCE, {k: impedance[k] for k in GAIN_KEYS}
        except (KeyError, TypeError):
            return IMPEDANCE, None
    return given.get("control_mode", POSITION), given.get("gains")


class PuppyArm:
    def __init__(self, arm, player):
        self._arm = arm
        self._player = player

    def _wrap(self, call, method):
        player = self._player

        @functools.wraps(method)
        def move(*args, **kwargs):
            # An e-stop met by a beat is the task call's to report, under its own contract (the
            # _rammp methods return False): disable and forward the call.
            try:
                player.before_move()
            except EStopped as e:
                player.fail_move(f"e-stopped before the move: {e}")
            mode, gains = mode_of(call, method, args, kwargs)
            player.move_started()
            try:
                result = method(*args, **kwargs)
            except BaseException as e:
                player.fail_move(f"{type(e).__name__}: {e}")
                raise
            finally:
                player.move_ended()
            if failed(result):  # False, or a driver code: possible contact; no AfterMove, no beats
                player.fail_move(f"{call} returned {result!r}")
                return result
            player.record(mode, gains)
            self._set_control(mode, gains)
            try:
                player.after_move()
            except EStopped as e:
                player.fail_move(f"e-stopped after the move: {e}")
            return result

        return move

    def _set_control(self, mode, gains):
        """Tell the director the move's mode and gains (it keeps a longer settle after a beat while
        the arm holds something under soft impedance). It must never raise into the task's move: a
        director without set_control, or one that raises, is logged and the move carries on."""
        player = self._player
        try:
            player.director.set_control(mode, gains)
        except Exception as e:
            try:
                player.log(f"puppy: director.set_control failed ({type(e).__name__}: {e})")
            except Exception:
                pass

    @contextlib.contextmanager
    def stream(self, *args, **kwargs):
        self._player.stream_opened()
        try:
            with self._arm.stream(*args, **kwargs) as stream:
                yield stream
        except BaseException as e:  # any exit but a normal one (the stream failed, or the task's code
            # raised inside the block): no beats after it
            self._player.fail_move(f"stream: {type(e).__name__}: {e}")
            raise
        finally:
            self._player.stream_closed()

    def __getattr__(self, name):
        if name in ("_arm", "_player"):
            raise AttributeError(name)
        attr = getattr(self._arm, name)
        if name in MOVE_CALLS:
            return self._wrap(name, attr)
        return attr
