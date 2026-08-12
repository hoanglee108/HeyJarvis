"""mvp.md Task 6: wake-word triggering and debounce."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.config import WakeWordConfig
from jarvis.wakeword import WakeWordDetector, WakeWordError

FRAME = np.zeros(1280, dtype=np.int16)


class ScriptedDetector(WakeWordDetector):
    """Replaces the ONNX model with a scripted score sequence."""

    def __init__(self, config: WakeWordConfig, scores: list[float]) -> None:
        super().__init__(config)
        self._scores = list(scores)
        self._calls = 0

    def score(self, frame: np.ndarray) -> float:  # noqa: ARG002
        value = self._scores[min(self._calls, len(self._scores) - 1)]
        self._calls += 1
        self._last_score = value
        return value


def test_fires_when_score_crosses_threshold() -> None:
    detector = ScriptedDetector(WakeWordConfig(threshold=0.5), [0.1, 0.3, 0.9])
    assert detector.process(FRAME, now=0.0) is False
    assert detector.process(FRAME, now=0.1) is False
    assert detector.process(FRAME, now=0.2) is True
    assert detector.last_score == pytest.approx(0.9)


def test_refractory_period_suppresses_double_triggers() -> None:
    config = WakeWordConfig(threshold=0.5, refractory_seconds=2.0)
    detector = ScriptedDetector(config, [0.9])

    assert detector.process(FRAME, now=100.0) is True
    assert detector.process(FRAME, now=100.5) is False, "vẫn trong thời gian chờ"
    assert detector.process(FRAME, now=101.9) is False
    assert detector.process(FRAME, now=102.1) is True, "hết thời gian chờ thì được bắn lại"


def test_threshold_is_configurable() -> None:
    strict = ScriptedDetector(WakeWordConfig(threshold=0.9), [0.7])
    assert strict.process(FRAME, now=0.0) is False

    loose = ScriptedDetector(WakeWordConfig(threshold=0.3), [0.7])
    assert loose.process(FRAME, now=0.0) is True


def test_mark_triggered_starts_the_refractory_window() -> None:
    detector = ScriptedDetector(WakeWordConfig(threshold=0.5, refractory_seconds=5.0), [0.99])
    detector.mark_triggered(now=10.0)
    assert detector.process(FRAME, now=12.0) is False
    assert detector.process(FRAME, now=16.0) is True


def test_unknown_model_name_raises_clear_error() -> None:
    detector = WakeWordDetector(WakeWordConfig(model="khong_ton_tai_dau"))
    with pytest.raises(WakeWordError, match="Không tìm thấy model wake word"):
        detector.load()
