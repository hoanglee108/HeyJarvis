"""priority.md P1-2: the speech gate must tell a voice from music.

The energy gate is the thing being replaced, so it is tested here as the *baseline*
that fails: proving it treats a loud non-speech signal as speech is what justifies the
neural gate existing at all.

Silero tests are skipped when openWakeWord's bundled model is unavailable, since it is
an optional artefact of an installed package rather than something in this repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio import FRAME_SAMPLES, record_utterance
from jarvis.config import AudioConfig
from jarvis.vad import (
    EnergyGate,
    SileroGate,
    VadError,
    build_speech_gate,
    frame_rms,
)

from .conftest import FakeMic, silence_frame, speech_frame

SAMPLE_RATE = 16000


def make_config(**overrides: object) -> AudioConfig:
    defaults: dict[str, object] = {
        "silence_timeout_ms": 240,  # 3 frames
        "min_record_seconds": 0.16,  # 2 frames
        "max_record_seconds": 2.0,
        "start_timeout_seconds": 0.4,
        "pre_roll_ms": 0,
        "beep_on_listen": False,
    }
    defaults.update(overrides)
    return AudioConfig(**defaults)  # type: ignore[arg-type]


def music_frame(index: int = 0, level: float = 0.2) -> np.ndarray:
    """A loud harmonic frame: musical in character, definitely not speech."""
    start = index * FRAME_SAMPLES
    t = (np.arange(FRAME_SAMPLES, dtype=np.float32) + start) / SAMPLE_RATE
    tone = sum(np.sin(2 * np.pi * freq * t) for freq in (220.0, 277.2, 329.6, 440.0))
    tone = tone / 4.0 * level
    return np.clip(tone * 32768, -32768, 32767).astype(np.int16)


def silero_or_skip(**kwargs: object) -> SileroGate:
    gate = SileroGate(**kwargs)  # type: ignore[arg-type]
    try:
        gate.load()
    except VadError as exc:  # pragma: no cover - depends on the installed package
        pytest.skip(f"Silero VAD không khả dụng: {exc}")
    return gate


# -- frame_rms -------------------------------------------------------------------------
def test_frame_rms_scales_int16_to_unit_range() -> None:
    assert frame_rms(np.zeros(FRAME_SAMPLES, dtype=np.int16)) == 0.0
    loud = np.full(FRAME_SAMPLES, 16384, dtype=np.int16)
    assert 0.45 < frame_rms(loud) < 0.55
    assert frame_rms(np.zeros(0, dtype=np.int16)) == 0.0


def test_frame_rms_accepts_float_input_without_rescaling() -> None:
    assert frame_rms(np.full(64, 0.5, dtype=np.float32)) == pytest.approx(0.5, abs=1e-6)


# -- energy gate: the baseline that fails ----------------------------------------------
def test_energy_gate_follows_the_threshold() -> None:
    gate = EnergyGate(0.05)
    assert gate.is_speech(speech_frame())
    assert not gate.is_speech(silence_frame())
    assert gate.last_level == pytest.approx(frame_rms(silence_frame()))


def test_energy_gate_cannot_tell_music_from_speech() -> None:
    """The bug P1-2 exists to fix: loud music reads as speech at every frame."""
    gate = EnergyGate(0.05)
    assert all(gate.is_speech(music_frame(i)) for i in range(20))


def test_energy_gate_never_ends_a_recording_while_music_plays() -> None:
    mic = FakeMic([music_frame(i) for i in range(40)])
    result = record_utterance(mic, make_config(), gate=EnergyGate(0.05))
    assert result.reason == "max_duration", "cổng năng lượng không bao giờ thấy im lặng"


# -- silero gate ------------------------------------------------------------------------
def test_silero_scores_music_lower_than_speech() -> None:
    gate = silero_or_skip(threshold=0.5)
    music_scores = [gate.probability(music_frame(i)) for i in range(12)]
    gate.reset()
    speech_scores = [gate.probability(speech_frame()) for _ in range(12)]
    assert max(music_scores) < 0.5, f"nhạc bị coi là giọng nói: {max(music_scores):.3f}"
    assert max(music_scores) < max(speech_scores) or max(speech_scores) < 0.5


def test_silero_lets_a_recording_end_while_music_plays() -> None:
    """The whole point: music alone must read as silence so the turn can finish."""
    gate = silero_or_skip(threshold=0.5)
    frames = [speech_frame()] * 8 + [music_frame(i) for i in range(8)]
    result = record_utterance(FakeMic(frames), make_config(), gate=gate)
    assert result.reason in ("silence", "no_speech")


def test_silero_min_rms_vetoes_quiet_audio() -> None:
    gate = silero_or_skip(threshold=0.01, min_rms=0.5)
    # A very high veto rejects everything regardless of the model's opinion.
    assert not gate.is_speech(speech_frame(level=0.05))


def test_silero_reset_clears_lstm_state() -> None:
    gate = silero_or_skip()
    gate.probability(speech_frame())
    gate.reset()
    assert gate.last_probability == 0.0


def test_silero_handles_frames_that_are_not_a_multiple_of_the_chunk() -> None:
    gate = silero_or_skip()
    score = gate.probability(np.zeros(700, dtype=np.int16))
    assert 0.0 <= score <= 1.0


def test_silero_describe_mentions_the_threshold() -> None:
    gate = SileroGate(threshold=0.42, min_rms=0.01)
    assert "0.42" in gate.describe()


# -- factory ----------------------------------------------------------------------------
def test_build_speech_gate_defaults_to_energy() -> None:
    gate = build_speech_gate(make_config(vad_backend="energy"), energy_threshold=0.07)
    assert isinstance(gate, EnergyGate)
    assert gate.threshold == pytest.approx(0.07)


def test_build_speech_gate_returns_silero_when_asked() -> None:
    silero_or_skip()  # skip early if the model is missing
    gate = build_speech_gate(
        make_config(vad_backend="silero", vad_threshold=0.6), energy_threshold=0.07
    )
    assert isinstance(gate, SileroGate)
    assert gate.threshold == pytest.approx(0.6)


def test_build_speech_gate_falls_back_when_silero_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken install must degrade to the old gate, not kill the assistant."""

    def boom(self: SileroGate) -> None:
        raise VadError("giả lập thiếu model")

    monkeypatch.setattr(SileroGate, "load", boom)
    gate = build_speech_gate(make_config(vad_backend="silero"), energy_threshold=0.09)
    assert isinstance(gate, EnergyGate)
    assert gate.threshold == pytest.approx(0.09)


# -- record_utterance plumbing ----------------------------------------------------------
def test_record_utterance_requires_a_gate_or_a_threshold() -> None:
    with pytest.raises(ValueError):
        record_utterance(FakeMic([]), make_config())


def test_record_utterance_still_accepts_a_bare_threshold() -> None:
    """The old keyword keeps working, so existing callers and tests are unaffected."""
    frames = [speech_frame()] * 5 + [silence_frame()] * 6
    result = record_utterance(FakeMic(frames), make_config(), speech_threshold=0.05)
    assert result.reason == "silence"


def test_record_utterance_reports_peak_from_the_gate() -> None:
    frames = [speech_frame()] * 5 + [silence_frame()] * 6
    result = record_utterance(FakeMic(frames), make_config(), gate=EnergyGate(0.05))
    assert result.peak_rms > 0.1
