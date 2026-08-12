"""Shared test fixtures and fakes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.audio import FRAME_SAMPLES  # noqa: E402
from jarvis.config import JarvisConfig, load_config, missing_model_files  # noqa: E402

MODELS_AVAILABLE_REASON = "Model chưa tải: chạy python scripts/download_models.py"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_config() -> JarvisConfig:
    """The repository's own ``config.yaml``, fully validated."""
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def models_ready(real_config: JarvisConfig) -> bool:
    return not missing_model_files(real_config)


@pytest.fixture
def requires_models(models_ready: bool) -> None:
    if not models_ready:
        pytest.skip(MODELS_AVAILABLE_REASON)


# --------------------------------------------------------------------------------------
# audio fakes
# --------------------------------------------------------------------------------------
def speech_frame(level: float = 0.2, samples: int = FRAME_SAMPLES) -> np.ndarray:
    """A loud-ish noise frame that the energy gate treats as speech."""
    rng = np.random.default_rng(1234)
    data = rng.normal(0.0, level, samples)
    return np.clip(data * 32768, -32768, 32767).astype(np.int16)


def silence_frame(level: float = 0.0005, samples: int = FRAME_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(99)
    data = rng.normal(0.0, level, samples)
    return np.clip(data * 32768, -32768, 32767).astype(np.int16)


class FakeMic:
    """Replays a scripted list of frames through the :class:`MicStream` interface."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.index = 0
        self.drained = 0
        self.running = True

    def read(self, timeout: float = 1.0) -> np.ndarray | None:  # noqa: ARG002
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def recent(self, milliseconds: int) -> np.ndarray:
        if milliseconds <= 0 or self.index == 0:
            return np.zeros(0, dtype=np.int16)
        return self.frames[max(0, self.index - 1)]

    def drain(self) -> None:
        self.drained += 1

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
