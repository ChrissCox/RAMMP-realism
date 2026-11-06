"""The two proxies a Puppy Mode action wraps around its ctx and its arm.

PuppyCtx(ctx, director_feed): `progress()` forwards to the real ctx first, then feeds a Phase event
to `director_feed(event)`. Every other attribute is the real ctx's.

PuppyArm(arm, player): the move calls (go_to_*, execute_joint_trajectory, move_joints,
move_cartesian) feed BeforeMove to the player, forward UNCHANGED to the real arm, and feed AfterMove
after a normal return. They also record the call's control mode and gains on the player, which the
beats copy: `control_mode=` / `gains=` of the go_to_* calls and execute_joint_trajectory, and for
move_joints / move_cartesian `impedance=` (a dict means IMPEDANCE with those gains, none means
POSITION). `stream()` marks a stream open until its block exits, and no beat plays meanwhile.
Every other attribute is the real arm's.

Neither proxy changes what a call returns or raises, or the arguments it forwards.
"""

import contextlib
import functools
import inspect

from capability_engine.arm import GAIN_KEYS, IMPEDANCE, POSITION

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
            player.before_move()
            mode, gains = mode_of(call, method, args, kwargs)
            player.move_started()
            try:
                result = method(*args, **kwargs)
            finally:
                player.move_ended()
            player.record(mode, gains)
            player.after_move()
            return result

        return move

    @contextlib.contextmanager
    def stream(self, *args, **kwargs):
        self._player.stream_opened()
        try:
            with self._arm.stream(*args, **kwargs) as stream:
                yield stream
        finally:
            self._player.stream_closed()

    def __getattr__(self, name):
        if name in ("_arm", "_player"):
            raise AttributeError(name)
        attr = getattr(self._arm, name)
        if name in MOVE_CALLS:
            return self._wrap(name, attr)
        return attr
