"""Vietnamese speech-to-text via sherpa-onnx + hynt/Zipformer-30M-RNNT-6000h.

The checkpoint is an *offline* (non-streaming) RNN-T transducer whose BPE vocabulary
is uppercase, so :meth:`SpeechToText.transcribe` normalises the casing before the
text reaches the LLM.
"""

from __future__ import annotations

import threading
import time
import unicodedata
from pathlib import Path

import numpy as np

from .audio import read_wav, to_float32
from .config import SttConfig
from .logging_setup import get_logger

log = get_logger("jarvis.stt")


class SttError(RuntimeError):
    """Model files missing or the recogniser failed."""


class SpeechToText:
    """Lazy-loading wrapper around ``sherpa_onnx.OfflineRecognizer``.

    The ONNX sessions are created once and reused; ``transcribe`` is guarded by a
    lock because sherpa-onnx recognisers are not safe to share across threads.
    """

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._recognizer = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------------
    def _require_files(self) -> None:
        missing = [
            path
            for path in (
                self._config.encoder,
                self._config.decoder,
                self._config.joiner,
                self._config.tokens,
            )
            if not Path(path).is_file()
        ]
        if missing:
            listing = "\n  - ".join(str(path) for path in missing)
            raise SttError(
                "Thiếu file model STT:\n  - "
                + listing
                + "\nChạy: python scripts/download_models.py"
            )

    def load(self) -> None:
        """Create the recogniser (idempotent)."""
        if self._recognizer is not None:
            return
        with self._lock:
            if self._recognizer is not None:
                return
            self._require_files()
            try:
                import sherpa_onnx
            except ImportError as exc:  # pragma: no cover
                raise SttError("Chưa cài sherpa-onnx: pip install -r requirements.txt") from exc

            started = time.perf_counter()
            try:
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=str(self._config.encoder),
                    decoder=str(self._config.decoder),
                    joiner=str(self._config.joiner),
                    tokens=str(self._config.tokens),
                    num_threads=self._config.num_threads,
                    sample_rate=self._config.sample_rate,
                    feature_dim=self._config.feature_dim,
                    decoding_method=self._config.decoding_method,
                    max_active_paths=self._config.max_active_paths,
                    provider=self._config.provider,
                    debug=self._config.debug,
                )
            except Exception as exc:
                raise SttError(f"Không nạp được model STT: {exc}") from exc
            log.info(
                "Đã nạp STT Zipformer (%s, %d threads) trong %.2fs",
                self._config.provider,
                self._config.num_threads,
                time.perf_counter() - started,
            )

    @property
    def loaded(self) -> bool:
        return self._recognizer is not None

    # -- inference ---------------------------------------------------------------
    def transcribe(self, samples: np.ndarray, sample_rate: int | None = None) -> str:
        """Transcribe mono audio (float32 in ``[-1,1]`` or int16) to Vietnamese text."""
        self.load()
        audio = to_float32(samples)
        rate = sample_rate or self._config.sample_rate
        if audio.size == 0:
            return ""

        started = time.perf_counter()
        with self._lock:
            assert self._recognizer is not None  # narrowed by load()
            stream = self._recognizer.create_stream()
            stream.accept_waveform(rate, audio)
            self._recognizer.decode_stream(stream)
            raw = stream.result.text
        elapsed = time.perf_counter() - started

        text = self.normalise(raw, lowercase=self._config.lowercase_output)
        duration = audio.size / rate
        log.info(
            "STT %.2fs audio -> %.2fs (RTF %.2f): %r",
            duration,
            elapsed,
            elapsed / duration if duration else 0.0,
            text,
        )
        return text

    def transcribe_file(self, path: str | Path) -> str:
        samples, rate = read_wav(path, target_rate=self._config.sample_rate)
        return self.transcribe(samples, rate)

    # -- text post-processing ----------------------------------------------------
    @staticmethod
    def normalise(text: str, *, lowercase: bool = True) -> str:
        """Tidy raw transducer output: NFC, collapse spaces, optional lowercase."""
        cleaned = unicodedata.normalize("NFC", text or "").strip()
        cleaned = " ".join(cleaned.split())
        if lowercase:
            cleaned = cleaned.lower()
        return cleaned
