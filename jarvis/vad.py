"""Speech gates: the per-frame decision "is a human talking right now?".

Why this module exists (priority.md P1-2)
----------------------------------------
The original gate was pure energy: RMS above a calibrated threshold. That works in a
quiet room and collapses the moment music plays through the speakers, because RMS
cannot tell a voice from a guitar:

* music sits *above* the threshold continuously, so :func:`jarvis.audio.record_utterance`
  sees speech in every frame, never observes ``silence_timeout_ms`` of quiet, and runs
  until ``max_record_seconds`` - feeding 30 s of music to the STT model;
* :func:`jarvis.audio.calibrate_noise_floor` runs once at startup, so calibrating while
  music plays raises the threshold to roughly twice the music level and afterwards even
  shouting counts as silence.

A small neural VAD does distinguish the two. Silero VAD ships *inside* openWakeWord
(``resources/models/silero_vad.onnx``, MIT licensed), which is already a dependency, so
this costs no new package and roughly 0.2 ms of CPU per 80 ms frame.

The gate is a stateful object rather than a function on purpose: Silero is an LSTM and
carries hidden state across frames, so the pipeline must be able to reset it whenever it
drains the microphone.

This module deliberately does not import :mod:`jarvis.audio` - the dependency runs the
other way, and ``INT16_MAX``/:func:`frame_rms` live here so both can share them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .config import AudioConfig
from .logging_setup import get_logger

log = get_logger("jarvis.vad")

INT16_MAX = 32768.0

#: Silero is fed 40 ms sub-chunks (640 samples @ 16 kHz). An 80 ms capture frame divides
#: evenly into two of them. This mirrors openWakeWord's own default.
SILERO_CHUNK_SAMPLES = 640


class VadError(RuntimeError):
    """The neural VAD could not be loaded or failed to run."""


def frame_rms(frame: np.ndarray) -> float:
    """Root-mean-square of an int16 or float frame, normalised to 0..1."""
    if frame.size == 0:
        return 0.0
    data = frame.astype(np.float32)
    if np.issubdtype(frame.dtype, np.integer):
        data /= INT16_MAX
    return float(np.sqrt(np.mean(np.square(data))))


@runtime_checkable
class SpeechGate(Protocol):
    """Frame-by-frame speech decision used by :func:`jarvis.audio.record_utterance`."""

    #: RMS of the most recent frame; kept for logging and ``RecordingResult.peak_rms``.
    last_level: float

    def is_speech(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None: ...

    def describe(self) -> str: ...


class EnergyGate:
    """The original behaviour: a frame is speech when its RMS clears a threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.last_level = 0.0

    def is_speech(self, frame: np.ndarray) -> bool:
        self.last_level = frame_rms(frame)
        return self.last_level >= self.threshold

    def reset(self) -> None:  # no state to clear
        return None

    def describe(self) -> str:
        return f"energy(threshold={self.threshold:.4f})"


class SileroGate:
    """Silero VAD, so music is not mistaken for a voice.

    ``min_rms`` is an optional cheap veto on top of the model. Silero answers "is this
    speech?", not "is this speech *near the microphone*?", so sung vocals coming out of
    the speakers still score high - that is what the model was trained to do. Requiring
    a minimum level as well keeps quiet playback from opening the gate while a real
    speaker half a metre away clears it easily. Set it to 0 to trust the model alone.

    This is a mitigation, not a fix; ducking (priority.md P1-3) is what actually deals
    with vocals.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        *,
        min_rms: float = 0.0,
        chunk_samples: int = SILERO_CHUNK_SAMPLES,
        num_threads: int = 1,
    ) -> None:
        self.threshold = float(threshold)
        self.min_rms = float(min_rms)
        self._chunk = int(chunk_samples)
        self._num_threads = int(num_threads)
        self._model = None
        self.last_level = 0.0
        self.last_probability = 0.0

    # -- loading -----------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from openwakeword.vad import VAD
        except ImportError as exc:  # pragma: no cover - openwakeword is a hard dep
            raise VadError(
                "Chưa cài openwakeword nên không dùng được Silero VAD: "
                "pip install -r requirements.txt"
            ) from exc
        try:
            model = VAD(n_threads=self._num_threads)
        except Exception as exc:
            raise VadError(f"Không nạp được Silero VAD: {exc}") from exc
        self._model = model
        log.info(
            "Silero VAD sẵn sàng (ngưỡng %.2f, min_rms %.4f)", self.threshold, self.min_rms
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- inference ---------------------------------------------------------------
    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one frame (0..1)."""
        self.load()
        assert self._model is not None
        samples = np.asarray(frame).reshape(-1)
        if samples.dtype != np.int16:
            if np.issubdtype(samples.dtype, np.floating):
                samples = np.clip(samples * INT16_MAX, -INT16_MAX, INT16_MAX - 1).astype(np.int16)
            else:
                samples = samples.astype(np.int16)
        if samples.size == 0:
            return 0.0
        remainder = samples.size % self._chunk
        if remainder:
            # Pad so every sub-chunk handed to the model has the expected length.
            samples = np.concatenate(
                [samples, np.zeros(self._chunk - remainder, dtype=np.int16)]
            )
        try:
            score = float(self._model.predict(samples, frame_size=self._chunk))
        except Exception as exc:
            raise VadError(f"Lỗi khi chạy Silero VAD: {exc}") from exc
        self.last_probability = score
        return score

    def is_speech(self, frame: np.ndarray) -> bool:
        self.last_level = frame_rms(frame)
        # The model is fed even when the level veto fails, so its LSTM state stays
        # continuous with the audio timeline.
        speech = self.probability(frame) >= self.threshold
        if self.min_rms > 0.0 and self.last_level < self.min_rms:
            return False
        return speech

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()
        self.last_probability = 0.0

    def describe(self) -> str:
        return f"silero(threshold={self.threshold:.2f}, min_rms={self.min_rms:.4f})"


def build_speech_gate(config: AudioConfig, *, energy_threshold: float) -> SpeechGate:
    """Create the gate named by ``config.vad_backend``.

    Falls back to :class:`EnergyGate` - with a warning, never an exception - when the
    neural model cannot be loaded, so a broken install degrades to the old behaviour
    instead of a dead assistant.
    """
    if config.vad_backend == "silero":
        gate = SileroGate(
            threshold=config.vad_threshold,
            min_rms=config.vad_min_rms,
            num_threads=config.vad_num_threads,
        )
        try:
            gate.load()
            return gate
        except VadError as exc:
            log.warning("Không dùng được Silero VAD (%s); quay lại ngưỡng năng lượng.", exc)
    return EnergyGate(energy_threshold)
