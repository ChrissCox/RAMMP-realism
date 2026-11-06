from puppy_playground.supervisor import Cancel, SendAuto, State, Supervisor


class Rig:
    """A supervisor on a fake clock. step() advances time in 0.1 s ticks, republishing the idle
    engines every 0.5 s (capability_engine does 1 Hz) and the arm's stillness; returns the commands."""

    def __init__(self, on=True, engines=("capability_a",), estop=False, discovered=True):
        self.t = 100.0
        self.sup = Supervisor(lambda: self.t)
        self.idle = set(engines)
        self.arm_still = True
        self.still_s = 0.0
        self.sup.estopped(estop)
        if discovered:
            for e in engines:
                self.sup.engine_discovered(e)
            self.sup.discovered()
        self.publish()
        if on:
            self.sup.toggle(True)

    def publish(self):
        for e in self.idle:
            self.sup.engine_status(e, "idle")

    def step(self, seconds):
        out = []
        for _ in range(round(seconds * 10)):
            self.t += 0.1
            self.still_s = self.still_s + 0.1 if self.arm_still else 0.0
            self.sup.arm_still_for(self.still_s)
            if round(self.t * 10) % 5 == 0:
                self.publish()
            cmd = self.sup.tick(self.t)
            if cmd is not None:
                out.append(cmd)
        return out

    def play(self):
        self.still_s = 99.0
        out = self.step(8)
        assert self.sup.state is State.PLAYING, out
        return out

    def task(self, engine="rammp_task", state="executing"):
        self.idle.discard(engine)
        return self.sup.engine_status(engine, state)


def sends(cmds):
    return [c for c in cmds if isinstance(c, SendAuto)]


def test_starts_off_and_never_sends():
    r = Rig(on=False)
    assert r.step(10) == []
    assert r.sup.state is State.OFF


def test_plays_when_idle_still_and_quiet():
    r = Rig()
    r.still_s = 99
    out = r.step(8)
    assert len(sends(out)) == 1 and sends(out)[0].size == 0.25
    assert r.sup.state is State.PLAYING
    assert r.step(2) == []  # one send only


def test_toggle_size_is_used_and_clamped():
    r = Rig(on=False)
    r.sup.toggle(True, size=0.6)
    r.still_s = 99
    assert sends(r.step(8))[0].size == 0.6
    r2 = Rig(on=False)
    r2.sup.toggle(True, size=5.0)
    r2.still_s = 99
    assert sends(r2.step(8))[0].size == 1.0


def test_task_start_cancels_within_one_tick_and_waits():
    r = Rig()
    r.play()
    assert r.task() == [Cancel()]
    assert r.sup.state is State.WAITING_FOR_TASK
    for _ in range(50):  # the task keeps reporting executing; the arm is perfectly still
        r.sup.engine_status("rammp_task", "executing")
        assert r.step(0.1) == []
    assert r.sup.state is State.WAITING_FOR_TASK


def test_a_paused_task_also_blocks_and_cancels():
    r = Rig()
    r.play()
    assert r.task(state="paused") == [Cancel()]
    assert r.sup.state is State.WAITING_FOR_TASK


def test_own_engine_status_is_ignored():
    r = Rig()
    r.play()
    assert r.sup.engine_status("puppy_playground", "executing") == []
    assert r.sup.state is State.PLAYING


def test_resumes_only_after_all_idle_and_three_quiet_seconds():
    r = Rig()
    r.play()
    r.task()
    r.step(1)
    r.sup.engine_status("rammp_task", "idle")
    r.idle.add("rammp_task")
    assert sends(r.step(2.8)) == []  # the arm was still all along, but the task only just ended
    assert len(sends(r.step(1.0))) == 1


def test_still_arm_throughout_a_task_then_idle_waits_three_seconds():
    r = Rig()
    r.play()
    r.task()
    for _ in range(20):
        r.sup.engine_status("rammp_task", "executing")
        r.step(0.1)
    r.sup.engine_status("rammp_task", "idle")
    r.idle.add("rammp_task")
    assert sends(r.step(2.8)) == []
    assert len(sends(r.step(0.5))) == 1


def test_moving_arm_blocks_even_when_everything_is_idle():
    r = Rig()
    r.arm_still = False
    assert sends(r.step(10)) == []
    r.arm_still = True
    assert sends(r.step(2.9)) == []
    assert len(sends(r.step(0.5))) == 1


def test_every_engine_must_be_idle():
    r = Rig(on=False, engines=("a", "b"))
    r.idle.discard("b")
    r.sup.engine_status("b", "executing")
    r.sup.toggle(True)
    assert r.sup.state is State.WAITING_FOR_TASK
    r.still_s = 99
    for _ in range(100):
        r.sup.engine_status("b", "executing")
        assert sends(r.step(0.1)) == []
    r.sup.engine_status("b", "idle")
    r.idle.add("b")
    assert len(sends(r.step(4))) == 1


def test_unknown_or_empty_state_is_busy():
    for state in ("", "weird", None):
        r = Rig(on=False)
        r.idle.discard("capability_a")
        r.sup.engine_status("capability_a", state)
        r.sup.toggle(True)
        r.still_s = 99
        assert sends(r.step(10)) == [], state


def test_a_discovered_engine_never_heard_is_busy():
    r = Rig(on=False, engines=("a",))
    r.sup.engine_discovered("ghost")
    r.sup.toggle(True)
    r.still_s = 99
    assert sends(r.step(10)) == []
    r.sup.forget("ghost")
    assert len(sends(r.step(4))) == 1


def test_a_stale_engine_status_is_busy_and_cancels():
    r = Rig()
    r.play()
    r.idle.clear()  # capability_a stops publishing
    out = r.step(3)
    assert Cancel() in out
    assert r.sup.state is State.WAITING_FOR_TASK
    assert sends(r.step(20)) == []


def test_startup_grace_after_discovery():
    r = Rig(discovered=False)
    r.still_s = 99
    assert sends(r.step(10)) == []  # discovery never ran
    r.sup.discovered()
    assert sends(r.step(1.5)) == []
    assert len(sends(r.step(1.0))) == 1


def test_no_control_status_yet_counts_as_estopped():
    sup = Supervisor(lambda: 0.0)
    sup.toggle(True)
    assert sup.state is State.OFF and sup.off_reason == "estopped"


def test_estop_while_playing_cancels_and_turns_auto_off_and_release_does_not_restart():
    r = Rig()
    r.play()
    assert r.sup.estopped(True) == [Cancel()]
    assert r.sup.state is State.OFF and r.sup.off_reason == "estopped"
    assert sends(r.step(10)) == []  # arm still the whole time
    r.sup.estopped(False)
    assert r.sup.state is State.OFF
    assert sends(r.step(60)) == []  # nothing is ever sent after the release
    r.sup.toggle(True)  # the user re-enables auto
    assert r.sup.off_reason is None
    assert len(sends(r.step(10))) == 1


def test_estop_between_beats_turns_auto_off_and_nothing_is_sent_after_release():
    r = Rig()
    r.still_s = 0.0
    r.arm_still = False
    r.step(2)  # waiting, never played
    assert r.sup.state is State.WAITING_IDLE
    assert r.sup.estopped(True) == []
    assert r.sup.state is State.OFF and r.sup.off_reason == "estopped"
    r.sup.estopped(False)
    r.arm_still = True
    assert sends(r.step(60)) == []


def test_estop_cleared_while_a_task_runs_stays_off():
    r = Rig()
    r.play()
    r.sup.estopped(True)
    r.task()
    r.sup.estopped(False)
    assert r.sup.state is State.OFF


def test_toggle_on_while_estopped_is_refused_and_nothing_starts_on_release():
    r = Rig(on=False, estop=True)
    assert r.sup.toggle(True) == []
    assert r.sup.state is State.OFF and r.sup.off_reason == "estopped"
    r.still_s = 99
    assert sends(r.step(10)) == []
    r.sup.estopped(False)
    assert r.sup.state is State.OFF
    assert sends(r.step(60)) == []
    r.sup.toggle(True)  # a press after the release turns it on
    assert r.sup.off_reason is None and sends(r.step(10))


def test_estop_while_off_stays_off():
    r = Rig(on=False)
    r.sup.estopped(True)
    r.sup.estopped(False)
    assert r.sup.state is State.OFF


def test_toggle_off_cancels_a_playing_auto():
    r = Rig()
    r.play()
    assert r.sup.toggle(False) == [Cancel()]
    assert r.sup.state is State.OFF
    assert r.step(10) == []


def test_toggle_off_when_not_playing_sends_nothing():
    r = Rig()
    assert r.sup.toggle(False) == []
    assert r.sup.state is State.OFF


def test_goal_done_does_not_cancel_the_ended_goal_and_turns_auto_off():
    r = Rig()
    r.play()
    assert r.sup.goal_done() == []
    assert r.sup.state is State.OFF and r.sup.off_reason == "ended"
    assert r.step(30) == []


def test_goal_done_is_ignored_when_not_playing():
    r = Rig()
    r.play()
    r.task()
    r.sup.goal_done()
    assert r.sup.state is State.WAITING_FOR_TASK


def test_playing_tick_recancels_if_estop_appears_unreported():
    r = Rig()
    r.play()
    r.sup._estop = True  # a missed edge must still be caught by tick
    assert Cancel() in r.step(0.2)
    assert r.sup.state is State.OFF and r.sup.off_reason == "estopped"


def test_seeds_change_between_sends():
    r = Rig()
    r.still_s = 99
    a = sends(r.step(8))[0]
    r.sup.goal_done()
    r.sup.toggle(True)
    b = sends(r.step(5))[0]
    assert a.seed != b.seed


def test_any_auto_goal_that_ends_by_itself_turns_auto_off_with_its_code():
    for code, reason in (("drifted", "drifted"), ("no_model", "no_model"), ("exception", "exception"),
                         ("plan_only", "plan_only"), ("estopped", "estopped"), ("contact", "contact"),
                         (None, "ended"), ("", "ended")):
        r = Rig()
        r.play()
        assert r.sup.goal_done(code) == []
        assert r.sup.state is State.OFF and r.sup.off_reason == reason
        assert r.step(60) == []
        r.sup.toggle(True)  # toggling on clears the reason and auto resumes
        assert r.sup.off_reason is None and r.sup.state is not State.OFF
        assert sends(r.step(10))


def test_a_goal_the_supervisor_cancelled_for_a_task_is_ignored_and_auto_resumes_after_the_task():
    r = Rig()
    r.play()
    assert r.task() == [Cancel()]
    r.sup.goal_done()  # the cancelled result arrives
    assert r.sup.state is State.WAITING_FOR_TASK and r.sup.off_reason is None
    r.idle.add("rammp_task")
    r.sup.engine_status("rammp_task", "idle")
    assert sends(r.step(10))


def test_a_cancelled_result_never_turns_auto_off():
    r = Rig()
    r.play()
    assert r.sup.goal_done("", cancelled=True) == []
    assert r.sup.state is State.WAITING_IDLE and r.sup.off_reason is None
    assert sends(r.step(10))


def test_retry_later_waits_and_resends_without_turning_auto_off():
    r = Rig()
    r.play()
    assert r.sup.retry_later() == []
    assert r.sup.state is State.WAITING_IDLE and r.sup.off_reason is None
    assert sends(r.step(10))
