"""The reference clips: registered, they pass every safety check, and idle_alive does what it says."""

import numpy as np
import pytest

from puppy_sim import checks, render, runs
from puppy_sim.clips import registry
from puppy_sim.timeline import PRECISION

REFERENCE_CLIPS = ("replay_plain", "idle_alive", "vocab_reel", "replay_puppy")


def test_registry_has_the_reference_clips():
    clips = registry()
    for name in REFERENCE_CLIPS:
        assert name in clips and clips[name].camera in render.CAMERAS


@pytest.mark.parametrize("name", REFERENCE_CLIPS)
def test_reference_clip_passes_every_check(name):
    for clip, label, _, results, _ in checks.check_named(name):
        failed = [(c.name, c.shown, c.detail) for c in results if not c.ok]
        assert not failed, (clip, label, failed)


def test_idle_alive_glances_tilts_and_looks(kin, sc):
    track = registry()["idle_alive"].build()
    eye0, R0 = kin.fk(track.q[0], "eye")

    def at(t):
        return kin.fk(track.q[int(round(t / 0.02))], "eye")

    def angle(a, b):
        return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))

    _, R = at(2.4)  # glancing at the user: turned toward them
    to_user = sc.person_head - eye0
    assert angle(R[:, 2], R0[:, 2]) > 12.0
    assert R[:, 2] @ to_user > R0[:, 2] @ to_user
    _, R = at(3.2)  # the curious tilt rolls the head
    assert angle(R[:, 1], R0[:, 1]) > 12.0
    eye, R = at(6.4)  # looking at the container
    d = sc.placement.center - eye
    assert angle(R[:, 2], d / np.linalg.norm(d)) < 10.0
    assert np.abs(track.q[-1] - np.array(runs.HOME_JOINTS)).max() < 1e-3  # settled back home
    assert {b.name for b in track.beats} == {"HELLO", "CURIOUS", "INTERESTED"}


def test_checks_catch_a_doctored_track(kin):
    track = registry()["idle_alive"].build()
    track.zone[100:110] = PRECISION  # pretend these samples were precision: they carry offsets
    track.tau[200:] += 0.5  # and a jump forward in functional time
    failed = {c.name for c in checks.check_track(track, kin) if not c.ok}
    assert {"precision", "dtau/dt"} <= failed


def test_check_command_exits_nonzero_on_a_violation(monkeypatch, kin):
    from puppy_sim import __main__ as cli

    track = registry()["idle_alive"].build()
    track.zone[100:110] = PRECISION
    monkeypatch.setattr(
        checks, "check_named", lambda name: [(name, "puppy", track.duration, checks.check_track(track, kin), [])]
    )
    assert cli.main(["check", "idle_alive"]) == 1
