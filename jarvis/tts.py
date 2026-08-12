"""Vietnamese text-to-speech via sherpa-onnx (VITS / Piper voices).

Piper voices need the bundled ``espeak-ng-data`` directory for phonemisation, which
is why :class:`TtsConfig` carries ``data_dir``.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np

from .audio import Speaker, write_wav
from .config import TtsConfig
from .logging_setup import get_logger

log = get_logger("jarvis.tts")

#: Markdown / decorative characters that must never be read out loud.
_STRIP_PATTERN = re.compile(r"[*_`#>|~\[\]{}<>]+")
_EMOJI_PATTERN = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "\u2190-\u21ff" "]+",
    flags=re.UNICODE,
)


class TtsError(RuntimeError):
    """Voice model missing or synthesis failed."""


class SynthesisResult:
    __slots__ = ("samples", "sample_rate")

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate if self.sample_rate else 0.0


def clean_for_speech(text: str) -> str:
    """Strip markdown, emoji and bullet artefacts so the voice sounds natural."""
    cleaned = _EMOJI_PATTERN.sub(" ", text or "")
    cleaned = _STRIP_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"^\s*[-•‣▪]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*\n\s*", ". ", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" .;:,-").strip()


class TextToSpeech:
    """Lazy-loading wrapper around ``sherpa_onnx.OfflineTts`` plus playback."""

    def __init__(self, config: TtsConfig, speaker: Speaker) -> None:
        self._config = config
        self._speaker = speaker
        self._tts = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------------
    def _require_files(self) -> None:
        missing: list[Path] = []
        for path in (self._config.model, self._config.tokens):
            if not Path(path).is_file():
                missing.append(Path(path))
        if self._config.data_dir is not None and not Path(self._config.data_dir).is_dir():
            missing.append(Path(self._config.data_dir))
        if missing:
            listing = "\n  - ".join(str(path) for path in missing)
            raise TtsError(
                "Thiếu file model TTS:\n  - "
                + listing
                + "\nChạy: python scripts/download_models.py"
            )

    def load(self) -> None:
        if self._tts is not None:
            return
        with self._lock:
            if self._tts is not None:
                return
            self._require_files()
            try:
                import sherpa_onnx
            except ImportError as exc:  # pragma: no cover
                raise TtsError("Chưa cài sherpa-onnx: pip install -r requirements.txt") from exc

            started = time.perf_counter()
            try:
                vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(self._config.model),
                    tokens=str(self._config.tokens),
                    lexicon=str(self._config.lexicon) if self._config.lexicon else "",
                    data_dir=str(self._config.data_dir) if self._config.data_dir else "",
                    dict_dir=str(self._config.dict_dir) if self._config.dict_dir else "",
                )
                model_config = sherpa_onnx.OfflineTtsModelConfig(
                    vits=vits,
                    num_threads=self._config.num_threads,
                    provider=self._config.provider,
                    debug=self._config.debug,
                )
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=model_config,
                    max_num_sentences=self._config.max_num_sentences,
                )
                if not tts_config.validate():
                    raise TtsError("Cấu hình TTS không hợp lệ (sherpa-onnx validate() = False)")
                self._tts = sherpa_onnx.OfflineTts(tts_config)
            except TtsError:
                raise
            except Exception as exc:
                raise TtsError(f"Không nạp được model TTS: {exc}") from exc

            log.info(
                "Đã nạp TTS %s (%d speaker, %d Hz) trong %.2fs",
                Path(self._config.model).name,
                self._tts.num_speakers,
                self._tts.sample_rate,
                time.perf_counter() - started,
            )

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    @property
    def sample_rate(self) -> int:
        self.load()
        assert self._tts is not None
        return int(self._tts.sample_rate)

    # -- synthesis ---------------------------------------------------------------
    def synthesize(self, text: str) -> SynthesisResult:
        self.load()
        assert self._tts is not None
        spoken = clean_for_speech(text)
        if not spoken:
            return SynthesisResult(np.zeros(0, dtype=np.float32), int(self._tts.sample_rate))

        speaker_id = self._config.speaker_id
        if speaker_id >= self._tts.num_speakers:
            log.warning(
                "speaker_id=%d vượt quá số speaker (%d), dùng 0",
                speaker_id,
                self._tts.num_speakers,
            )
            speaker_id = 0

        started = time.perf_counter()
        with self._lock:
            try:
                audio = self._tts.generate(spoken, sid=speaker_id, speed=self._config.speed)
            except Exception as exc:
                raise TtsError(f"Lỗi khi tổng hợp giọng nói: {exc}") from exc
        samples = np.asarray(audio.samples, dtype=np.float32)
        result = SynthesisResult(samples, int(audio.sample_rate))
        log.info(
            "TTS %d ký tự -> %.2fs audio trong %.2fs",
            len(spoken),
            result.duration,
            time.perf_counter() - started,
        )
        return result

    def speak(self, text: str) -> SynthesisResult:
        """Synthesize and play through the configured output device."""
        result = self.synthesize(text)
        if result.samples.size:
            self._speaker.play(result.samples, result.sample_rate)
        return result

    def save(self, text: str, path: str | Path) -> Path:
        result = self.synthesize(text)
        if result.samples.size == 0:
            raise TtsError("Không có gì để tổng hợp (text rỗng sau khi làm sạch).")
        return write_wav(path, result.samples, result.sample_rate)
