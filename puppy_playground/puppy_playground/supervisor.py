"""Auto-mode supervisor: a pure state machine (no ROS) deciding when the playground may send motion.

Events (toggle, engine_status, estopped, goal_done) return a list of commands; tick returns one
command or None. Commands: SendAuto(size, seed) and Cancel(). Missing information counts as busy:
another engine is idle only if it said "idle" within STALE_S; no /control_status yet means
e-stopped; nothing is sent before discovered() + GRACE_S. Auto resumes only when every engine is
idle, the arm and the "quiet" clock (reset by any busy signal, e-stop or the auto goal ending) have
both been still for STILL_S.
"""

import enum
from dataclasses import dataclass

OWN_ID = "puppy_playground"
STILL_S = 3.0
STALE_S = 2.5
GRACE_S = 2.0
RETRY_GAP_S = 1.0
SIZE_MIN, SIZE_MAX = 0.25, 1.0
BUSY_STATES = ("executing", "paused")


class State(enum.Enum):
    OFF = "off"
    PLAYING = "playing"
    WAITING_FOR_TASK = "waiting_for_task"
    WAITING_IDLE = "waiting_idle"
    ESTOPPED = "estopped"


@dataclass(frozen=True)
class SendAuto:
    size: float
    seed: int


@dataclass(frozen=True)
class Cancel:
    pass


class Supervisor:
    def __init__(self, clock, size=0.25, seed=1):
        self._clock = clock
        self.state = State.OFF
        self.size = size
        self._seed = seed
        self._engines = {}  # id -> (state or None, heard_at)
        self._still = 0.0
        self._estop = True  # until /control_status says otherwise
        self._last_send = None
        self._quiet_since = clock()
        self._discovered_at = None

    # -- inputs ---------------------------------------------------------------

    def discovered(self):
        """One discovery pass over the status topics has finished."""
        if self._discovered_at is None:
            self._discovered_at = self._clock()

    def engine_discovered(self, engine_id):
        if engine_id != OWN_ID:
            self._engines.setdefault(engine_id, (None, None))

    def forget(self, engine_id):
        self._engines.pop(engine_id, None)

    def toggle(self, on, size=None):
        if size is not None:
            self.size = min(SIZE_MAX, max(SIZE_MIN, float(size)))
        if not on:
            return self._enter(State.OFF)
        if self.state is not State.OFF:
            return []
        return self._enter(self._waiting_state(self._clock()))

    def engine_status(self, engine_id, state):
        now = self._clock()
        if engine_id == OWN_ID:
            return []
        self._engines[engine_id] = (state, now)
        if state == "idle":
            return []
        if state in BUSY_STATES:
            self._quiet_since = now
        if self.state in (State.PLAYING, State.WAITING_IDLE):
            return self._enter(State.WAITING_FOR_TASK)
        return []

    def arm_still_for(self, seconds):
        self._still = seconds

    def estopped(self, value):
        self._estop = bool(value)
        self._quiet_since = self._clock()
        if self._estop:
            if self.state is not State.OFF:
                return self._enter(State.ESTOPPED)
        elif self.state is State.ESTOPPED:
            return self._enter(self._waiting_state(self._clock()))
        return []

    def goal_done(self):
        """The auto goal ended by itself: nothing to cancel."""
        self._quiet_since = self._clock()
        if self.state is State.PLAYING:
            self.state = State.WAITING_IDLE
        return []

    def tick(self, now=None):
        now = self._clock() if now is None else now
        busy = self._task_busy(now)
        if busy or self._estop:
            self._quiet_since = now
        if self._estop and self.state not in (State.OFF, State.ESTOPPED):
            return Cancel() if self._enter(State.ESTOPPED) else None
        if self.state is State.PLAYING:
            if busy:
                self._enter(State.WAITING_FOR_TASK)
                return Cancel()
            return None
        if self.state is State.WAITING_FOR_TASK:
            if busy:
                return None
            self.state = State.WAITING_IDLE
        if self.state is State.WAITING_IDLE:
            if busy:
                self.state = State.WAITING_FOR_TASK
                return None
            if self._discovered_at is None or now - self._discovered_at < GRACE_S:
                return None
            if self._still < STILL_S or now - self._quiet_since < STILL_S:
                return None
            if self._last_send is not None and now - self._last_send < RETRY_GAP_S:
                return None
            self._last_send = now
            self._seed += 1
            self.state = State.PLAYING
            return SendAuto(self.size, self._seed)
        return None

    # -- internals ------------------------------------------------------------

    def _task_busy(self, now):
        for state, heard in self._engines.values():
            if state != "idle" or heard is None or now - heard > STALE_S:
                return True
        return False

    def _waiting_state(self, now):
        if self._estop:
            return State.ESTOPPED
        return State.WAITING_FOR_TASK if self._task_busy(now) else State.WAITING_IDLE

    def _enter(self, state):
        """Change state; cancel if the playground's own goal was running."""
        was_playing = self.state is State.PLAYING
        self.state = state
        return [Cancel()] if was_playing else []
