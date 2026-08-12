"""mvp.md Task 6: silence detection / end-of-speech logic (no real microphone)."""

from __future__ import annotations

import numpy as np

from jarvis.audio import FRAME_SAMPLES, calibrate_noise_floor, record_utterance, rms
from jarvis.config import AudioConfig

from .conftest import FakeMic, silence_frame, speech_frame


def make_config(**overrides: object) -> AudioConfig:
    defaults: dict[str, object] = {
        "silence_timeout_ms": 240,  # 3 frames
        "min_record_seconds": 0.16,  # 2 frames
        "max_record_seconds": 2.0,
        "start_timeout_seconds": 0.4,  # 5 frames
        "silence_rms_threshold": 0.05,
        "pre_roll_ms": 0,
        "beep_on_listen": False,
    }
    defaults.update(overrides)
    return AudioConfig(**defaults)  # type: ignore[arg-type]


def test_rms_scales_int16_to_unit_range() -> None:
    assert rms(np.zeros(FRAME_SAMPLES, dtype=np.int16)) == 0.0
    loud = np.full(FRAME_SAMPLES, 16384, dtype=np.int16)
    assert 0.45 < rms(loud) < 0.55
    assert rms(np.zeros(0, dtype=np.int16)) == 0.0


def test_recording_stops_after_configured_silence() -> None:
    frames = [speech_frame()] * 5 + [silence_frame()] * 6
    mic = FakeMic(frames)
    result = record_utterance(mic, make_config(), speech_threshold=0.05)

    assert result.reason == "silence"
    assert result.has_speech
    # 5 speech frames + the 3 silence frames that proved the end of speech.
    assert result.samples.size == 8 * FRAME_SAMPLES
    assert mic.index == 8, "phải dừng đọc mic ngay khi phát hiện im lặng"


def test_recording_reports_no_speech_when_nobody_talks() -> None:
    mic = FakeMic([silence_frame()] * 20)
    result = record_utterance(mic, make_config(), speech_threshold=0.05)
    assert result.reason == "no_speech"
    assert not result.has_speech


def test_recording_is_capped_by_max_duration() -> None:
    mic = FakeMic([speech_frame()] * 200)
    config = make_config(max_record_seconds=0.8)  # 10 frames
    result = record_utterance(mic, config, speech_threshold=0.05)
    assert result.reason == "max_duration"
    assert result.duration <= 0.8 + 1e-6


def test_short_blip_does_not_end_the_turn_early() -> None:
    """One quiet frame mid-sentence must not cut the recording."""
    frames = (
        [speech_frame()] * 3
        + [silence_frame()]
        + [speech_frame()] * 3
        + [silence_frame()] * 4
    )
    mic = FakeMic(frames)
    result = record_utterance(mic, make_config(), speech_threshold=0.05)
    assert result.reason == "silence"
    assert result.samples.size == 10 * FRAME_SAMPLES


def test_mic_timeout_is_reported() -> None:
    mic = FakeMic([speech_frame()] * 3)  # then read() returns None
    result = record_utterance(mic, make_config(), speech_threshold=0.05)
    assert result.reason == "mic_timeout"


def test_pre_roll_is_prepended() -> None:
    frames = [speech_frame()] * 3 + [silence_frame()] * 4
    mic = FakeMic(frames)
    pre_roll = np.zeros(800, dtype=np.int16)
    result = record_utterance(mic, make_config(), speech_threshold=0.05, pre_roll=pre_roll)
    assert result.samples.size == 800 + 6 * FRAME_SAMPLES


def test_calibration_uses_configured_threshold_when_given() -> None:
    config = make_config(silence_rms_threshold=0.033)
    assert calibrate_noise_floor(FakeMic([]), config) == 0.033


def test_calibration_derives_threshold_from_ambient_noise() -> None:
    config = make_config(
        silence_rms_threshold=None,
        calibration_seconds=0.4,
        calibration_multiplier=3.0,
        calibration_floor_minimum=0.001,
    )
    mic = FakeMic([silence_frame(level=0.01)] * 10)
    threshold = calibrate_noise_floor(mic, config)
    assert threshold > 0.001
    assert threshold < 0.2


def test_calibration_respects_the_floor_minimum() -> None:
    config = make_config(
        silence_rms_threshold=None,
        calibration_seconds=0.4,
        calibration_floor_minimum=0.02,
    )
    mic = FakeMic([np.zeros(FRAME_SAMPLES, dtype=np.int16)] * 10)
    assert calibrate_noise_floor(mic, config) == 0.02
