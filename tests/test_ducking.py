"""priority.md P1-3: per-application volume ducking while Jarvis listens.

``VolumeDucker._sessions`` is the only seam that touches Windows COM, so every test here
replaces it with fake sessions. Nothing in this file changes the machine's real volume.
"""

from __future__ import annotations

import os
import time

import pytest

from jarvis.config import DuckingConfig
from jarvis.ducking import VolumeDucker

OTHER_PID = os.getpid() + 1


class FakeVolume:
    """Stand-in for ``ISimpleAudioVolume``."""

    def __init__(self, level: float = 1.0, *, fail: bool = False) -> None:
        self.level = level
        self.history: list[float] = []
        self.fail = fail

    def GetMasterVolume(self) -> float:  # noqa: N802 - mirrors the COM interface
        if self.fail:
            raise OSError("session đã chết")
        return self.level

    def SetMasterVolume(self, value: float, _context: object) -> None:  # noqa: N802
        if self.fail:
            raise OSError("session đã chết")
        self.level = value
        self.history.append(value)


def make_ducker(
    sessions: list[tuple[FakeVolume, str, int]], **overrides: object
) -> VolumeDucker:
    config_kwargs: dict[str, object] = {"enabled": True, "level": 0.15, "restore_ms": 0}
    config_kwargs.update(overrides)
    ducker = VolumeDucker(DuckingConfig(**config_kwargs))  # type: ignore[arg-type]
    ducker._sessions = lambda: list(sessions)  # type: ignore[assignment] # noqa: SLF001
    return ducker


def wait_until_restored(volume: FakeVolume, original: float, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if volume.level == pytest.approx(original):
            return
        time.sleep(0.02)
    raise AssertionError(f"âm lượng chưa phục hồi: {volume.level} != {original}")


# -- basic behaviour -------------------------------------------------------------------
def test_acquire_lowers_other_apps() -> None:
    music = FakeVolume(1.0)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)])

    ducker.acquire()
    assert music.level == pytest.approx(0.15)
    assert ducker.active

    ducker.release()
    assert music.level == pytest.approx(1.0)
    assert not ducker.active


def test_our_own_process_is_never_ducked() -> None:
    """Jarvis's TTS shares the speakers; ducking ourselves defeats the purpose."""
    own = FakeVolume(1.0)
    ducker = make_ducker([(own, "python.exe", os.getpid())])

    ducker.acquire()
    assert own.level == pytest.approx(1.0)
    assert own.history == []


def test_process_allowlist_limits_what_is_touched() -> None:
    chrome = FakeVolume(1.0)
    game = FakeVolume(1.0)
    ducker = make_ducker(
        [(chrome, "chrome.exe", OTHER_PID), (game, "game.exe", OTHER_PID + 1)],
        processes=["chrome.exe"],
    )

    ducker.acquire()
    assert chrome.level == pytest.approx(0.15)
    assert game.level == pytest.approx(1.0)


def test_exclusions_win_over_the_default_catch_all() -> None:
    call = FakeVolume(1.0)
    ducker = make_ducker([(call, "teams.exe", OTHER_PID)], exclude_processes=["teams.exe"])

    ducker.acquire()
    assert call.level == pytest.approx(1.0)


def test_sessions_already_quiet_are_left_alone() -> None:
    """Nothing to restore later, so do not pretend 0.15 was the user's choice."""
    quiet = FakeVolume(0.10)
    ducker = make_ducker([(quiet, "chrome.exe", OTHER_PID)])

    ducker.acquire()
    assert quiet.history == []
    ducker.release()
    assert quiet.level == pytest.approx(0.10)


# -- reference counting ----------------------------------------------------------------
def test_nested_holds_only_restore_once_released_by_everyone() -> None:
    """Listening and speaking overlap; the music must not pop back up in between."""
    music = FakeVolume(0.8)
    ducker = make_ducker([(music, "spotify.exe", OTHER_PID)])

    ducker.acquire()
    ducker.acquire()
    ducker.release()
    assert music.level == pytest.approx(0.15), "vẫn còn một holder"

    ducker.release()
    assert music.level == pytest.approx(0.8)


def test_extra_releases_are_harmless() -> None:
    music = FakeVolume(1.0)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)])
    ducker.release()
    ducker.release()
    assert music.level == pytest.approx(1.0)


def test_original_level_survives_a_second_duck_cycle() -> None:
    music = FakeVolume(0.6)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)])

    for _ in range(3):
        ducker.acquire()
        assert music.level == pytest.approx(0.15)
        ducker.release()
        assert music.level == pytest.approx(0.6)


def test_context_manager_ducks_only_when_active() -> None:
    music = FakeVolume(1.0)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)])

    with ducker.ducked(active=False):
        assert music.level == pytest.approx(1.0)

    with ducker.ducked(active=True):
        assert music.level == pytest.approx(0.15)
    assert music.level == pytest.approx(1.0)


# -- fading ----------------------------------------------------------------------------
def test_fade_up_restores_asynchronously() -> None:
    """The ramp must not block the pipeline: release returns before the fade finishes."""
    music = FakeVolume(1.0)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)], restore_ms=200)

    ducker.acquire()
    started = time.monotonic()
    ducker.release()
    assert time.monotonic() - started < 0.15, "release() không được chặn luồng gọi"

    wait_until_restored(music, 1.0)
    assert len(music.history) > 2, "phải có nhiều bước tăng dần, không nhảy một nhịp"


def test_a_new_duck_abandons_an_in_flight_fade() -> None:
    music = FakeVolume(1.0)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)], restore_ms=600)

    ducker.acquire()
    ducker.release()  # starts a slow fade
    time.sleep(0.1)
    ducker.acquire()  # barge in while it is still fading

    time.sleep(0.3)
    assert music.level == pytest.approx(0.15), "duck mới phải thắng bản fade đang chạy"


# -- robustness ------------------------------------------------------------------------
def test_restore_all_puts_everything_back() -> None:
    music = FakeVolume(0.9)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)], restore_ms=2000)

    ducker.acquire()
    ducker.restore_all()
    assert music.level == pytest.approx(0.9)
    assert not ducker.active


def test_disabled_config_does_nothing() -> None:
    music = FakeVolume(1.0)
    ducker = make_ducker([(music, "chrome.exe", OTHER_PID)], enabled=False)

    assert not ducker.enabled
    ducker.acquire()
    assert music.history == []


def test_missing_pycaw_disables_ducking_instead_of_raising() -> None:
    ducker = VolumeDucker(DuckingConfig(enabled=True))

    def no_pycaw() -> list[tuple[FakeVolume, str, int]]:
        raise ImportError("no pycaw")

    ducker._sessions = no_pycaw  # type: ignore[assignment] # noqa: SLF001
    ducker.acquire()  # must not raise
    assert not ducker.enabled
    ducker.release()


def test_a_session_that_dies_mid_call_is_skipped() -> None:
    dead = FakeVolume(1.0, fail=True)
    alive = FakeVolume(1.0)
    ducker = make_ducker(
        [(dead, "gone.exe", OTHER_PID), (alive, "chrome.exe", OTHER_PID + 1)]
    )

    ducker.acquire()
    assert alive.level == pytest.approx(0.15)
    ducker.release()
    assert alive.level == pytest.approx(1.0)


def test_unexpected_enumeration_error_turns_ducking_off() -> None:
    ducker = VolumeDucker(DuckingConfig(enabled=True))

    def boom() -> list[tuple[FakeVolume, str, int]]:
        raise OSError("COM đang lỗi")

    ducker._sessions = boom  # type: ignore[assignment] # noqa: SLF001
    ducker.acquire()
    assert not ducker.enabled
