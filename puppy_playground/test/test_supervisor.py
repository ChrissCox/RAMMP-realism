import pytest

from puppy_playground.supervisor import Cancel, SendAuto, State, Supervisor


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


def make(on=True, still=10.0):
    clk = Clock()
    sup = Supervisor(clk)
    sup.engine_status("capability_a", "idle")
    sup.arm_still_for(still)
    if on:
        sup.toggle(True)
    return sup, clk


def test_starts_off_and_never_sends():
    sup, clk = make(on=False)
    assert sup.state is State.OFF
    assert sup.tick(clk.advance(5)) is None


def test_goes_playing_when_idle_and_still():
    sup, clk = make()
    assert sup.state is State.WAITING_IDLE
    cmd = sup.tick(clk.advance(0.1))
    assert isinstance(cmd, SendAuto) and cmd.size == 0.25
    assert sup.state is State.PLAYING
    assert sup.tick(clk.advance(0.1)) is None  # one send only


def test_toggle_size_is_used_and_clamped():
    sup, clk = make(on=False)
    sup.toggle(True, size=0.6)
    assert sup.tick(clk.advance(0.1)).size == 0.6
    sup2, clk2 = make(on=False)
    sup2.toggle(True, size=5.0)
    assert sup2.tick(clk2.advance(0.1)).size == 1.0


def test_task_start_cancels_within_one_tick_and_waits():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    assert sup.state is State.PLAYING
    out = sup.engine_status("rammp_task", "executing")
    assert out == [Cancel()]
    assert sup.state is State.WAITING_FOR_TASK
    # nothing is sent while the task runs, however still the arm is
    sup.arm_still_for(60)
    for _ in range(50):
        assert sup.tick(clk.advance(0.1)) is None
    assert sup.state is State.WAITING_FOR_TASK


def test_a_paused_task_also_blocks_and_cancels():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    sup.engine_status("rammp_task", "paused")
    assert sup.state is State.WAITING_FOR_TASK


def test_own_engine_status_is_ignored():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    sup.engine_status("puppy_playground", "executing")
    assert sup.state is State.PLAYING


def test_resumes_only_after_all_idle_and_three_still_seconds():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    sup.engine_status("rammp_task", "executing")
    sup.engine_status("other", "idle")
    sup.engine_status("rammp_task", "idle")
    sup.arm_still_for(0.0)  # arm just moved
    assert sup.tick(clk.advance(0.1)) is None
    assert sup.state is State.WAITING_IDLE
    sup.arm_still_for(2.9)
    assert sup.tick(clk.advance(0.1)) is None
    sup.engine_status("other", "executing")  # another task begins meanwhile
    sup.arm_still_for(3.5)
    assert sup.tick(clk.advance(0.1)) is None
    assert sup.state is State.WAITING_FOR_TASK
    sup.engine_status("other", "idle")
    sup.arm_still_for(3.0)
    assert isinstance(sup.tick(clk.advance(1.0)), SendAuto)
    assert sup.state is State.PLAYING


def test_every_engine_must_be_idle():
    sup, clk = make(on=False)
    sup.engine_status("a", "idle")
    sup.engine_status("b", "executing")
    sup.toggle(True)
    assert sup.state is State.WAITING_FOR_TASK
    assert sup.tick(clk.advance(1)) is None
    sup.engine_status("b", "idle")
    assert isinstance(sup.tick(clk.advance(0.1)), SendAuto)


def test_estop_cancels_and_holds_until_cleared_and_idle():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    out = sup.estopped(True)
    assert out == [Cancel()]
    assert sup.state is State.ESTOPPED
    sup.arm_still_for(60)
    for _ in range(20):
        assert sup.tick(clk.advance(0.1)) is None
    assert sup.state is State.ESTOPPED
    sup.estopped(False)
    assert sup.state is State.WAITING_IDLE
    sup.arm_still_for(1.0)
    assert sup.tick(clk.advance(0.1)) is None
    sup.arm_still_for(3.0)
    assert isinstance(sup.tick(clk.advance(0.1)), SendAuto)


def test_estop_cleared_while_a_task_runs_waits_for_the_task():
    sup, clk = make()
    sup.estopped(True)
    sup.engine_status("t", "executing")
    sup.estopped(False)
    assert sup.state is State.WAITING_FOR_TASK


def test_toggle_on_while_estopped_stays_estopped():
    sup, clk = make(on=False)
    sup.estopped(True)
    sup.toggle(True)
    assert sup.state is State.ESTOPPED
    assert sup.tick(clk.advance(5)) is None


def test_estop_while_off_stays_off():
    sup, clk = make(on=False)
    sup.estopped(True)
    sup.estopped(False)
    assert sup.state is State.OFF


def test_toggle_off_cancels_a_playing_auto():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    out = sup.toggle(False)
    assert out == [Cancel()]
    assert sup.state is State.OFF
    assert sup.tick(clk.advance(5)) is None


def test_toggle_off_when_not_playing_sends_nothing():
    sup, clk = make()
    out = sup.toggle(False)
    assert not out
    assert sup.state is State.OFF


def test_goal_done_re_waits_and_resends_after_the_retry_gap():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    sup.goal_done()
    assert sup.state is State.WAITING_IDLE
    assert sup.tick(clk.advance(0.1)) is None  # not a hot loop
    assert isinstance(sup.tick(clk.advance(1.0)), SendAuto)


def test_goal_done_is_ignored_when_not_playing():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    sup.engine_status("t", "executing")
    sup.goal_done()
    assert sup.state is State.WAITING_FOR_TASK


def test_playing_tick_recancels_if_unsafe_condition_appears_unreported():
    sup, clk = make()
    sup.tick(clk.advance(0.1))
    sup._estop = True  # a missed edge must still be caught by tick
    out = sup.tick(clk.advance(0.1))
    assert out == Cancel()
    assert sup.state is State.ESTOPPED


def test_seeds_change_between_sends():
    sup, clk = make()
    a = sup.tick(clk.advance(0.1))
    sup.goal_done()
    b = sup.tick(clk.advance(2))
    assert a.seed != b.seed
