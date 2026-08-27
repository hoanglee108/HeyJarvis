"""The date/time tool: Vietnamese, speech-shaped, and read from the real clock."""

from __future__ import annotations

from datetime import datetime

import pytest

from jarvis.config import ClockToolConfig
from jarvis.tools.clock import Clock, ClockToolError


def clock_at(moment: datetime, *, enabled: bool = True) -> Clock:
    return Clock(ClockToolConfig(enabled=enabled), now=lambda: moment)


# -- weekday names ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("day", "weekday"),
    [
        (24, "Thứ Hai"),
        (25, "Thứ Ba"),
        (26, "Thứ Tư"),
        (27, "Thứ Năm"),
        (28, "Thứ Sáu"),
        (29, "Thứ Bảy"),
        (30, "Chủ Nhật"),
    ],
)
def test_every_weekday_has_a_vietnamese_name(day: int, weekday: str) -> None:
    # 2026-08-24 is a Monday.
    assert clock_at(datetime(2026, 8, day, 9, 0)).date_phrase().startswith(weekday)


# -- speech shaping --------------------------------------------------------------------
def test_time_is_spelled_out_for_the_voice_not_as_a_clock_face() -> None:
    assert clock_at(datetime(2026, 8, 25, 14, 32)).time_phrase() == "14 giờ 32 phút"


def test_a_whole_hour_reads_naturally() -> None:
    assert clock_at(datetime(2026, 8, 25, 9, 0)).time_phrase() == "9 giờ đúng"


def test_describe_covers_both_time_and_date() -> None:
    text = clock_at(datetime(2026, 8, 25, 14, 32)).describe()
    assert text == "Bây giờ là 14 giờ 32 phút, Thứ Ba, ngày 25 tháng 8 năm 2026."


def test_describe_contains_no_markup_for_the_tts_voice() -> None:
    text = clock_at(datetime(2026, 8, 25, 14, 32)).describe()
    assert not any(char in text for char in "<>*#_/:")


# -- configuration ---------------------------------------------------------------------
def test_disabled_clock_refuses_to_answer() -> None:
    clock = clock_at(datetime(2026, 8, 25, 14, 32), enabled=False)
    assert clock.enabled is False
    with pytest.raises(ClockToolError, match="đang bị tắt"):
        clock.describe()


def test_default_clock_reads_the_real_system_time() -> None:
    """No injected clock: the tool must reflect the machine, not a constant."""
    before = datetime.now()
    reading = Clock(ClockToolConfig()).now()
    after = datetime.now()
    assert before <= reading <= after
