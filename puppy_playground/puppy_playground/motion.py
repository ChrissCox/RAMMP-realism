"""express, combo and auto: the playground's expressive actions, all through pipeline.play.

Beats come from a cache keyed by (pose, name, size, held, seed). A beat is composed once for the
stage pose (lazily, on first use, with nothing commanded meanwhile) and reused while the arm is
within the pipeline's start tolerance of that pose; at any other posture it is composed live at the
measured posture before the pipeline's checks. Auto mode composes its next beat on a worker thread
during the pause. EStopped / ControlTaken end the action; other refusals are logged and auto goes on."""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from capability_engine.engine import EngineError
from rammp_arm_interfaces.msg import GripperState
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from puppy_motion.behaviors import Targets
from puppy_motion.idle import EXPRESSIONS, LOOK_TARGET, IdleDirector

from . import pipeline, playlog
from puppy_motion.flatbot import chair_scene

AT_STAGE_RAD = 0.02  # express / combo start within this of the stage pose
SIZE_MIN, SIZE_MAX, SIZE_DEFAULT = 0.25, 1.0, 0.25
HELD_LOW, HELD_HIGH = 0.05, 0.93
GRIPPER_STALE_S = 1.0
STEP_S = 0.1  # auto sleeps its pauses in steps this long
PREFETCH_AFTER_S = 0.5  # auto starts composing its next beat this long after a beat ends
EXPRESS_SEED = 0


def clean_size(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SIZE_DEFAULT
    if not np.isfinite(v):
        return SIZE_DEFAULT
    return min(SIZE_MAX, max(SIZE_MIN, v))


class GripperWatch:
    """Held from /gripper_state: mid-range, no message yet, or a message older than 1 s all count
    as held (the muted beats)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._pos = None
        self._t = None

    def update(self, position):
        with self._lock:
            self._pos, self._t = float(position), self._clock()

    def has_state(self):
        with self._lock:
            return self._t is not None

    def held(self):
        with self._lock:
            pos, t = self._pos, self._t
        if t is None or self._clock() - t > GRIPPER_STALE_S or not np.isfinite(pos):
            return True
        return HELD_LOW < pos < HELD_HIGH


class BeatSource:
    """Beats for express / combo, cached for the stage pose."""

    def __init__(self, chain, scene, stage_q):
        self.chain, self.scene, self.stage_q = chain, scene, np.asarray(stage_q, dtype=float)
        self._cache = {}
        self._lock = threading.Lock()

    def _compose(self, name, size, q, held, seed):
        director = IdleDirector(self.chain, self.scene, Targets(person_head=LOOK_TARGET), size, seed)
        return director.request(name, q, held, 0.0)

    def beat(self, name, size, q, held, seed=EXPRESS_SEED):
        """The beat for `name` at the measured q: the cached stage beat when q is at the stage pose
        (to the pipeline's shift tolerance; the pipeline shifts and re-checks it), else composed live at q."""
        if np.abs(pipeline.shortest_delta(self.chain, q, self.stage_q)).max() < pipeline.SHIFT_TOL_RAD:
            key = ("stage", name, size, bool(held), seed)
            with self._lock:
                if key not in self._cache:
                    self._cache[key] = self._compose(name, size, self.stage_q, held, seed)
                return self._cache[key]
        return self._compose(name, size, q, held, seed)


def _watch(engine):
    w = GripperWatch()
    engine.create_action_subscription(
        GripperState, "/gripper_state", lambda m: w.update(m.position), qos_profile_sensor_data
    )
    return w


def _wait_first(ctx, w):
    """Wait (briefly) for a first gripper message so a fresh action does not start out 'stale'."""
    deadline = time.monotonic() + GRIPPER_STALE_S
    while not w.has_state() and time.monotonic() < deadline:
        ctx.checkpoint()
        time.sleep(0.02)


def _setup(engine):
    _, chain = engine._urdf()
    return chain, chair_scene(), engine._stage_q(chain)


def _require_stage(engine, chain, stage_q):
    q = engine._still_q()
    if np.abs(pipeline.shortest_delta(chain, q, stage_q)).max() > AT_STAGE_RAD:
        raise EngineError("not_at_stage", "the arm is not at the stage pose", recoverable=True)


def _names(names):
    bad = [n for n in names if n not in EXPRESSIONS]
    if bad:
        raise EngineError(
            "unknown_expression", f"unknown expression(s): {', '.join(map(str, bad))}", recoverable=True
        )


def express(engine, ctx, rec):
    p = rec["params"]
    return _play_names(engine, ctx, rec, [p.get("name")], clean_size(p.get("size")))


def combo(engine, ctx, rec):
    p = rec["params"]
    names = p.get("names") or []
    if not isinstance(names, list) or not names:
        raise EngineError("bad_params", "combo needs a non-empty list of names", recoverable=True)
    return _play_names(engine, ctx, rec, names, clean_size(p.get("size")))


def _play_names(engine, ctx, rec, names, size):
    _names(names)
    chain, scene, stage_q = _setup(engine)
    source = BeatSource(chain, scene, stage_q)
    watch = _watch(engine)
    _wait_first(ctx, watch)
    rec["motions"] = []
    for i, name in enumerate(names):
        if i:
            ctx.checkpoint()
        _require_stage(engine, chain, stage_q)
        pipeline.play(
            engine.arm, ctx, chain, scene, lambda q, h, n=name: source.beat(n, size, q, h),
            watch.held(), log=rec["motions"].append, name=name,
        )
    return {"played": list(names)}


class _Requests:
    def __init__(self):
        self.names = deque(maxlen=16)

    def __call__(self, msg):
        self.names.append(str(msg.data).strip())


def _take_request(requests, log):
    while requests.names:
        name = requests.names.popleft()
        if name in EXPRESSIONS:
            return name
        log.warning(f"auto: ignoring unknown request {name!r}")
    return None


def auto(engine, ctx, rec):
    p = rec["params"]
    size = clean_size(p.get("size"))
    seed = int(p.get("seed", 1))
    chain, scene, _ = _setup(engine)
    director = IdleDirector(chain, scene, Targets(person_head=LOOK_TARGET), size, seed)
    watch = _watch(engine)
    requests = _Requests()
    engine.create_action_subscription(String, "~/request", requests, 10)
    _wait_first(ctx, watch)
    rec["motions"] = 0
    log = engine.get_logger()

    def write(m):
        playlog.write(engine.log_path, {**m, "action": "auto"}, log.warning)

    pool = ThreadPoolExecutor(max_workers=1)  # the only thread that touches the director

    def submit_next():
        q = engine._still_q()
        h = watch.held()
        return h, pool.submit(director.next, q, h, time.monotonic())

    def play_beat(beat, composed_held, name):
        def builder(q, held):
            return beat if beat is not None and held == composed_held else None

        rec["motions"] += 1
        try:
            pipeline.play(engine.arm, ctx, chain, scene, builder, watch.held(), log=write, name=name)
        except EngineError as e:
            if not e.recoverable:
                raise
            log.info(f"auto: skipped {name}: {e.error_code}")

    try:
        pending = None
        while True:
            ctx.checkpoint()
            try:
                if pending is None:
                    pending = submit_next()
                composed_held, fut = pending
                pending = None
                beat, pause = fut.result()
                if beat is not None:
                    play_beat(beat, composed_held, beat.name)
            except EngineError as e:
                if not e.recoverable:
                    raise
                log.info(f"auto: {e.error_code}: {e}")
                pause = 1.0
            elapsed, request = 0.0, None
            while elapsed < pause and request is None:
                ctx.checkpoint()
                if pending is None and elapsed >= PREFETCH_AFTER_S:
                    try:
                        pending = submit_next()
                    except EngineError:
                        pass
                request = _take_request(requests, log)
                if request is None:
                    time.sleep(STEP_S)
                    elapsed += STEP_S
            if request is None:
                request = _take_request(requests, log)
            if request is not None:
                if pending is not None:
                    pending[1].result()  # the worker is done with the director
                    pending = None
                try:
                    q = engine._still_q()
                    h = watch.held()
                    beat = pool.submit(director.request, request, q, h, time.monotonic()).result()
                    play_beat(beat, h, request)
                except EngineError as e:
                    if not e.recoverable:
                        raise
                    log.info(f"auto: request {request}: {e.error_code}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
