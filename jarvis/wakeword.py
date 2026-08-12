"""Wake-word detection with openWakeWord (mvp.md Task 6).

The pretrained ``hey_jarvis`` model runs on 80 ms int16 frames at 16 kHz. On Windows
there is no ``tflite-runtime`` wheel, so we always use the ONNX variant.

:class:`WakeWordDetector` is deliberately frame-oriented (``process(frame) -> bool``)
rather than owning a thread: the pipeline already has the microphone loop, and this
shape makes the debounce/refractory logic trivial to unit-test with synthetic scores.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import WakeWordConfig
from .logging_setup import get_logger

log = get_logger("jarvis.wakeword")


class WakeWordError(RuntimeError):
    """openWakeWord missing, model not downloaded, or inference failed."""


class WakeWordDetector:
    """Streaming wake-word detector with a refractory period after each trigger."""

    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._model = None
        self._keys: list[str] = []
        #: ``None`` until the first trigger, so a fresh detector is never inside its
        #: own refractory window (which would swallow the very first "Hey Jarvis").
        self._last_trigger: float | None = None
        self._last_score = 0.0

    # -- loading -----------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:  # pragma: no cover
            raise WakeWordError(
                "Chưa cài openwakeword: pip install -r requirements.txt"
            ) from exc

        spec = self._config.model
        model_arg = spec
        if spec in openwakeword.MODELS:
            onnx_path = Path(openwakeword.MODELS[spec]["model_path"].replace(".tflite", ".onnx"))
            if not onnx_path.is_file():
                raise WakeWordError(
                    f"Chưa tải model wake word ONNX ({onnx_path.name}). "
                    "Chạy: python scripts/download_models.py --only wakeword"
                )
        elif Path(spec).is_file():
            model_arg = str(Path(spec).resolve())
        else:
            raise WakeWordError(
                f"Không tìm thấy model wake word {spec!r}. Dùng một trong "
                f"{sorted(openwakeword.MODELS)} hoặc đường dẫn tới file .onnx."
            )

        try:
            self._model = Model(
                wakeword_models=[model_arg],
                inference_framework=self._config.inference_framework,
                enable_speex_noise_suppression=self._config.enable_speex_noise_suppression,
                vad_threshold=self._config.vad_threshold,
            )
        except Exception as exc:
            raise WakeWordError(f"Không nạp được model wake word: {exc}") from exc

        self._keys = list(self._model.models.keys())
        if not self._keys:  # pragma: no cover - defensive
            raise WakeWordError("openWakeWord không nạp được model nào.")
        log.info("Wake word sẵn sàng: %s (ngưỡng %.2f)", ", ".join(self._keys), self._config.threshold)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def last_score(self) -> float:
        return self._last_score

    def reset(self) -> None:
        """Clear the internal audio/prediction buffers (after a conversation turn)."""
        if self._model is not None:
            self._model.reset()
        self._last_score = 0.0

    # -- inference ---------------------------------------------------------------
    def score(self, frame: np.ndarray) -> float:
        """Highest wake-word probability for this frame."""
        self.load()
        assert self._model is not None
        samples = np.asarray(frame)
        if samples.dtype != np.int16:
            if np.issubdtype(samples.dtype, np.floating):
                samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
            else:
                samples = samples.astype(np.int16)

        try:
            predictions = self._model.predict(samples.reshape(-1))
        except Exception as exc:
            raise WakeWordError(f"Lỗi khi chạy wake word: {exc}") from exc

        best = 0.0
        for key in self._keys:
            best = max(best, float(predictions.get(key, 0.0)))
        self._last_score = best
        return best

    def process(self, frame: np.ndarray, *, now: float | None = None) -> bool:
        """Feed one frame; ``True`` means the wake word just fired.

        A trigger is suppressed while inside ``refractory_seconds`` of the previous
        one so a single "hey Jarvis" cannot start two conversations.
        """
        current_time = time.monotonic() if now is None else now
        confidence = self.score(frame)
        if confidence < self._config.threshold:
            return False
        if (
            self._last_trigger is not None
            and current_time - self._last_trigger < self._config.refractory_seconds
        ):
            log.debug("Bỏ qua trigger trong thời gian chờ (score=%.3f)", confidence)
            return False

        self._last_trigger = current_time
        log.info("Đã nghe wake word (score=%.3f)", confidence)
        return True

    def mark_triggered(self, *, now: float | None = None) -> None:
        """Restart the refractory window, e.g. after a manual (push-to-talk) trigger."""
        self._last_trigger = time.monotonic() if now is None else now
