"""Auto-mode supervisor: a pure state machine (no ROS) deciding when the playground may send motion.

Events (toggle, engine_status, estopped, goal_done) return a list of commands; tick returns one
command or None. Commands: SendAuto(size, seed) and Cancel(). Any other engine reporting
executing (or paused) cancels the playground at once; auto resumes only when every engine is idle,
the arm has been still for STILL_S, and the e-stop is clear.
"""

import enum
from dataclasses import dataclass

OWN_ID = "puppy_playground"
STILL_S = 3.0
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
        self._engines = {}
        self._still = 0.0
        self._estop = False
        self._last_send = None

    # -- inputs ---------------------------------------------------------------

    def toggle(self, on, size=None):
        if size is not None:
            self.size = min(SIZE_MAX, max(SIZE_MIN, float(size)))
        if not on:
            return self._enter(State.OFF)
        if self.state is not State.OFF:
            return []
        return self._enter(self._waiting_state())

    def engine_status(self, engine_id, state):
        self._engines[engine_id] = state
        if engine_id == OWN_ID or state not in BUSY_STATES:
            return []
        if self.state in (State.PLAYING, State.WAITING_IDLE):
            return self._enter(State.WAITING_FOR_TASK)
        return []

    def arm_still_for(self, seconds):
        self._still = seconds

    def estopped(self, value):
        self._estop = bool(value)
        if self._estop:
            if self.state is not State.OFF:
                return self._enter(State.ESTOPPED)
        elif self.state is State.ESTOPPED:
            return self._enter(self._waiting_state())
        return []

    def goal_done(self):
        if self.state is State.PLAYING:
            return self._enter(State.WAITING_IDLE)
        return []

    def tick(self, now=None):
        now = self._clock() if now is None else now
        if self._estop and self.state not in (State.OFF, State.ESTOPPED):
            return Cancel() if self._enter(State.ESTOPPED) else None
        if self.state is State.PLAYING:
            if self._task_busy():
                self._enter(State.WAITING_FOR_TASK)
                return Cancel()
            return None
        if self.state is State.WAITING_FOR_TASK:
            if self._task_busy():
                return None
            self.state = State.WAITING_IDLE
        if self.state is State.WAITING_IDLE:
            if self._task_busy():
                self.state = State.WAITING_FOR_TASK
                return None
            if self._still < STILL_S:
                return None
            if self._last_send is not None and now - self._last_send < RETRY_GAP_S:
                return None
            self._last_send = now
            self._seed += 1
            self.state = State.PLAYING
            return SendAuto(self.size, self._seed)
        return None

    # -- internals ------------------------------------------------------------

    def _task_busy(self):
        return any(s in BUSY_STATES for e, s in self._engines.items() if e != OWN_ID)

    def _waiting_state(self):
        if self._estop:
            return State.ESTOPPED
        return State.WAITING_FOR_TASK if self._task_busy() else State.WAITING_IDLE

    def _enter(self, state):
        """Change state; cancel if the playground's own goal was running."""
        was_playing = self.state is State.PLAYING
        self.state = state
        return [Cancel()] if was_playing else []
