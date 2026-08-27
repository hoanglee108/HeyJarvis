"""Local date and time, read from the machine clock.

This used to be a PowerShell entry in the shell whitelist. A dedicated Python tool is
better on every axis that matters here:

* no ``subprocess`` spawn, so the answer is instant instead of ~200-400 ms;
* no dependency on PowerShell's locale for Vietnamese weekday names;
* the output is written for a text-to-speech voice ("14 giờ 32 phút") rather than for
  a screen ("14:32").

It also removes the last reason for the model to reach for ``run_system_command`` on a
question it gets asked constantly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from ..config import ClockToolConfig
from ..logging_setup import get_logger

log = get_logger("jarvis.tools.clock")

#: ``datetime.weekday()`` is 0=Monday.
WEEKDAYS_VI = (
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
)


class ClockToolError(RuntimeError):
    """The clock tool is disabled."""


class Clock:
    """Formats the current local date/time for a Vietnamese voice."""

    def __init__(
        self,
        config: ClockToolConfig,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        # Injectable so tests can pin "now" without freezing the system clock.
        self._now = now or datetime.now

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def now(self) -> datetime:
        return self._now()

    def time_phrase(self, moment: datetime | None = None) -> str:
        moment = moment or self.now()
        if moment.minute == 0:
            return f"{moment.hour} giờ đúng"
        return f"{moment.hour} giờ {moment.minute} phút"

    def date_phrase(self, moment: datetime | None = None) -> str:
        moment = moment or self.now()
        weekday = WEEKDAYS_VI[moment.weekday()]
        return f"{weekday}, ngày {moment.day} tháng {moment.month} năm {moment.year}"

    def describe(self) -> str:
        """One spoken sentence covering both time and date."""
        if not self._config.enabled:
            raise ClockToolError("Công cụ xem ngày giờ đang bị tắt trong config.")
        moment = self.now()
        text = f"Bây giờ là {self.time_phrase(moment)}, {self.date_phrase(moment)}."
        log.info("Ngày giờ hiện tại: %s", text)
        return text
