"""The beat player: the one place a Puppy Mode beat reaches the arm.

An event comes in (Player.emit, fed by the proxies), the director answers with beats, and each beat
is played only if, at that moment:
  - no stream is open and no move call of the task is in flight;
  - the arm's joint state is fresh, every |joint velocity| < STILL_RAD_S, and the measured q is
    within START_TOL_RAD (wrap-aware, every joint) of where the beat starts;
  - check_beat passes on EXACTLY the points that are sent, against the measured q (under the
    stricter of the beat's zone and the zone the geometry gives at q).
Just before the send the arm is read again (fresh, still, within START_TOL_RAD of the checked q)
and the goal goes out at once. The beat is sent as one execute_joint_trajectory goal with
preemption QUEUE, over the arm joints only (the grip offsets of a beat are dropped on the robot;
logged once per action). After a beat the player waits (bounded) for the arm to settle before the
task gets control back; before composing a beat for an AfterMove that a rule could answer, it waits
for the arm to settle too, so the beat is composed at the settled posture.

Control mode: the task's last move call's control_mode and gains (recorded by PuppyArm). Before the
task has recorded any move (box_opening moves the arm in a subprocess, never through self.arm), and
for a NEAR_USER beat whatever the task used, it is IMPEDANCE with the recorded gains (else
"default").

A task move that fails (False, a non-zero driver code, an exception) disables the player (PuppyArm
calls fail_move()): nothing plays after that, not even the failure reaction.

Everything per-beat runs inside a never-raise guard: Cancelled, EStopped and ControlTaken
propagate; any other exception disables the player and is logged, so the task's own call goes on.
A refused goal (execute_joint_trajectory returning False, or raising any other ArmError), an
unreadable or stale joint state, a driver tolerance violation (likely contact) or any other failed
goal also set `disabled`: no more beats this action, the task goes on. A goal the driver ended
PREEMPTED raises ControlTaken; HALTED raises EStopped when /control_status says e-stopped, else
ControlTaken.

Not here (Task 13 wires them): the caps preset passed to check_beat.
"""

import logging
import time

import numpy as np
from capability_engine.arm import HALTED, IMPEDANCE, QUEUE
from capability_engine.driver_state import ArmError, ControlTaken, EStopped
from capability_engine.engine import Cancelled
from rammp_arm import ERRORS

from puppy_motion.beat import zone_at
from puppy_motion.director import AfterMove, BeforeMove, Phase
from puppy_motion.joints import align_turns, shortest_delta
from puppy_motion.track import Zone
from puppy_motion.validate import check_beat
from puppy_motion.zones import zone_of

STILL_RAD_S = 0.02  # every |joint velocity| must be below this for a beat to play
START_TOL_RAD = 0.002  # the measured q must be this close to the beat's start (every joint)
HALT_STATUS_WAIT_S = 0.5  # after a HALTED goal, how long /control_status may take to show an e-stop
DEFAULT_GAINS = "default"
FRESH_S = 0.1  # the reading just before the send may be this old at most
SETTLE_S = 1.0  # how long to wait for the arm to come to rest
SETTLE_READS = 3  # consecutive still readings that count as at rest
SETTLE_POLL_S = 0.02
WARM_WAIT_S = 2.0  # how long the first beat waits for the trajectory action server
PASS = (Cancelled, EStopped, ControlTaken)  # the only exceptions that leave the player

logger = logging.getLogger("puppy_mode_ros")


class Player:
    def __init__(self, arm, ctx, director, chain, scene, *, capability=None, held=False, log=None):
        """`arm`: the REAL arm (never a PuppyArm); `ctx`: the real ctx; `director`: anything with
        on_event(event, q_now, held) -> [Beat]. `held`: whether the container rides in the gripper,
        a bool or a zero-argument callable. `capability`: for the zone table; the director's own
        when not given."""
        self.arm, self.ctx, self.director = arm, ctx, director
        self.chain, self.scene = chain, scene
        self.capability = capability if capability is not None else getattr(director, "capability", "")
        self.held = held
        self.log = log or logger.info
        self.disabled = False
        self.control_mode = None  # the task's last move call's control_mode (None: none yet)
        self.gains = None
        self._streams = 0
        self._moving = 0
        self._phase = ("", "")
        self._grip_logged = False
        self._warmed = False
        self._pending_phase = None  # the latest Phase that arrived when the player could not take it

    # -- what the proxies report ---------------------------------------------------------------

    def record(self, control_mode, gains):
        self.control_mode, self.gains = control_mode, gains

    def stream_opened(self):
        self._streams += 1

    def stream_closed(self):
        self._streams = max(0, self._streams - 1)

    @property
    def streaming(self):
        return self._streams > 0

    def move_started(self):
        self._moving += 1

    def move_ended(self):
        self._moving = max(0, self._moving - 1)

    def fail_move(self, why):
        """The task's own move failed (possible contact): no more beats this action."""
        self._disable(f"the task's move failed ({why})")

    def before_move(self):
        entry = zone_of(self.capability, *self._phase)
        self.emit(BeforeMove(bool(entry.precise_end)))

    def after_move(self):
        self.emit(AfterMove())

    # -- events --------------------------------------------------------------------------------

    def emit(self, event):
        """Feed one director event; play the beats it answers with. Raises nothing but Cancelled,
        EStopped and ControlTaken."""
        try:
            self._emit(event)
        except PASS:
            raise
        except Exception as e:  # noqa: BLE001
            self._disable(f"{type(e).__name__}: {e}")

    def _emit(self, event):
        if isinstance(event, Phase):
            self._phase = (event.name, event.message)
        if self.disabled:
            return
        if self.streaming or self._moving:
            if isinstance(event, Phase):  # replayed to the director at the next safe point
                self._pending_phase = event
            return
        if isinstance(event, AfterMove) and self._could_fire(event):
            self._settle()  # compose at the settled posture; if it never settles, the beat is skipped
            if self.disabled:
                return
        read = self._read()
        if read is None:
            return
        q = read[0]
        if q is None:
            self.log("puppy: unreadable joint positions: no beats for this event")
            return
        held = self._held()
        pending, self._pending_phase = self._pending_phase, None
        if pending is not None:  # keeps the director's phase (its PRECISION gate) current
            self.director.on_event(pending, q, held)
        self.play(self.director.on_event(event, q, held))

    def _could_fire(self, event):
        """Whether a rule of the director's preset could answer `event` (True when unknown)."""
        rules = getattr(self.director, "rules", None)
        if rules is None:
            return True
        keep = getattr(self.director, "min_keep", 0)
        name = type(event).__name__
        return any(r.event == name and r.keep >= keep for r in rules)

    def _held(self):
        return bool(self.held() if callable(self.held) else self.held)

    # -- measuring -----------------------------------------------------------------------------

    def _read(self):
        """(q, max |dq|, age_s) of the arm now; q is None when it is not seven finite numbers. None
        (and disabled) when the state cannot be read. EStopped / ControlTaken propagate."""
        try:
            js = self.arm.joint_states()
        except (EStopped, ControlTaken):
            raise
        except ArmError as e:  # StateTimeout, a stale state, not in an action ...
            self._disable(f"cannot read the arm ({type(e).__name__}: {e})")
            return None
        age = float(getattr(js, "age_s", 0.0) or 0.0)
        try:
            q = np.array(js.position, dtype=float)
            v = np.array(js.velocity, dtype=float)
        except (TypeError, ValueError):
            return None, float("inf"), age
        if q.shape != (7,) or not np.isfinite(q).all():
            q = None
        # a NaN or missing velocity makes the speed NaN, which fails every "still" comparison
        speed = float(np.abs(v).max()) if v.shape == (7,) else float("nan")
        return q, speed, age

    def _settle(self):
        """Wait up to SETTLE_S for SETTLE_READS still readings in a row, checkpointing between
        them. True if the arm is at rest."""
        deadline = time.monotonic() + SETTLE_S
        still = 0
        while True:
            self.ctx.checkpoint()
            read = self._read()
            if read is None:
                return False
            q, speed, _ = read
            still = still + 1 if (q is not None and speed < STILL_RAD_S) else 0
            if still >= SETTLE_READS:
                return True
            if time.monotonic() >= deadline:
                self.log("puppy: the arm did not settle")
                return False
            time.sleep(SETTLE_POLL_S)

    def _warm(self):
        """Make the trajectory action client and wait for its server, so the first send does not
        wait on discovery. Best effort."""
        if self._warmed:
            return
        self._warmed = True
        try:
            self.arm._wait_for_server(self.arm._action("execute_joint_trajectory"), self.ctx, WARM_WAIT_S)
        except PASS:
            raise
        except Exception:  # noqa: BLE001  (a stub arm, or no server yet: the send itself reports it)
            pass

    # -- playing -------------------------------------------------------------------------------

    def play(self, beats):
        """Play `beats` in order, each only if it is safe right now (see the module doc)."""
        for beat in beats:
            if self.disabled or self.streaming:
                return
            self.ctx.checkpoint()
            if self._play_one(beat) is None:  # refused: the rest of the beats are dropped too
                return

    def _play_one(self, beat):
        """True if sent and finished, False if skipped, None if the player is now disabled."""
        try:
            return self._play_beat(beat)
        except PASS:
            raise
        except Exception as e:  # noqa: BLE001  (a bad beat or held must never stop the task)
            return self._disable(f"{type(e).__name__}: {e}")

    def _play_beat(self, beat):
        held = self._held()
        self._warm()
        read = self._read()
        if read is None:
            return None
        q, speed, _ = read
        if q is None:
            self.log(f"puppy: skip {beat.name}: the measured q is not finite")
            return False
        if not (speed < STILL_RAD_S):
            self.log(f"puppy: skip {beat.name}: the arm is moving ({speed:.3f} rad/s)")
            return False
        gap = float(np.abs(shortest_delta(self.chain, q, beat.points[0][1])).max())
        if not (gap <= START_TOL_RAD):  # also False for NaN
            self.log(f"puppy: skip {beat.name}: the arm is {gap:.4f} rad from where it starts")
            return False
        points = align_turns(self.chain, q, beat.points)  # the driver reports joints wrapped
        offset = q - np.asarray(points[0][1], dtype=float)  # < START_TOL_RAD: start exactly at q
        points = [(t, np.asarray(p, dtype=float) + offset, dq) for t, p, dq in points]
        points[0] = (points[0][0], q.copy(), points[0][2])
        zone = zone_at(self.chain, self.scene, q, held, beat.zone)  # the stricter of the two
        found = check_beat(points, q, self.chain, self.scene, zone, held=held)
        if found:
            self.log(f"puppy: skip {beat.name}: " + "; ".join(f"{v.rule}: {v.detail}" for v in found))
            return False
        mode, gains = self._mode(zone)
        if not self._grip_logged:
            self._grip_logged = True
            self.log("puppy: beats move the arm joints only; any grip offsets are dropped on the robot")
        # a fresh reading right before the send: the arm is where the beat was checked
        self.ctx.checkpoint()
        read = self._read()
        if read is None:
            return None
        q2, speed2, age = read
        if q2 is None or not (speed2 < STILL_RAD_S) or not (age <= FRESH_S):
            self.log(f"puppy: skip {beat.name}: the arm moved or its state is old at the send")
            return False
        if not (float(np.abs(shortest_delta(self.chain, q, q2)).max()) <= START_TOL_RAD):
            self.log(f"puppy: skip {beat.name}: the arm moved while the beat was checked")
            return False
        sent = self._send(beat, points, mode, gains)
        if sent:
            self._settle()  # the task gets the arm back at rest; if not, the next beat is skipped
        return sent

    def _mode(self, zone):
        mode, gains = self.control_mode, self.gains
        if mode is None:  # the task has moved nothing through self.arm yet
            return IMPEDANCE, DEFAULT_GAINS
        if zone is Zone.NEAR_USER:
            return IMPEDANCE, (DEFAULT_GAINS if gains is None else gains)
        return mode, gains

    def _send(self, beat, points, mode, gains):
        try:
            res = self.arm.execute_joint_trajectory(
                self.chain.joint_names, points, control_mode=mode, gains=gains, preemption=QUEUE
            )
        except (EStopped, ControlTaken):
            raise
        except ArmError as e:  # Refused, a missing action server, ...
            return self._disable(f"the arm refused {beat.name}: {e}")
        if res is False:
            last = str(getattr(self.arm, "last_error", "") or "no reason given")
            return self._disable(f"the driver refused {beat.name}: {last}")
        code = int(res.error_code)
        if code == 0:
            return True
        name = ERRORS.get(code, res.error_string)
        if name == "PREEMPTED":
            raise ControlTaken(f"the beat {beat.name} was preempted: control was taken")
        if name in ("PATH_TOLERANCE_VIOLATED", "GOAL_TOLERANCE_VIOLATED"):
            return self._disable(f"the arm stopped {beat.name} early ({name}): possible contact")
        if code == HALTED:
            if self._estopped():
                raise EStopped(f"the arm e-stopped during {beat.name}")
            raise ControlTaken(f"the driver halted {beat.name}: control was taken")
        return self._disable(f"the driver ended {beat.name}: {name} {res.error_string}")

    def _disable(self, why):
        """Disable the player for the rest of the action; returns None (the 'disabled' result)."""
        self.disabled = True
        self.log(f"puppy: {why}; no more beats this action")
        return None

    def _estopped(self):
        deadline = time.monotonic() + HALT_STATUS_WAIT_S
        while True:
            try:
                if self.arm.control_status().estopped:
                    return True
            except ArmError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
