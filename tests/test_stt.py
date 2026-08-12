"""mvp.md Task 3: Zipformer transcription.

The integration test is self-contained: it synthesises Vietnamese speech with the
local TTS voice and feeds it back through the recogniser, so no committed audio
fixture is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.audio import Speaker, write_wav
from jarvis.config import JarvisConfig
from jarvis.stt import SpeechToText, SttError
from jarvis.tts import TextToSpeech


# -- normalisation --------------------------------------------------------------------
def test_uppercase_output_is_lowercased() -> None:
    assert SpeechToText.normalise("XIN CHÀO JARVIS") == "xin chào jarvis"


def test_lowercasing_can_be_disabled() -> None:
    assert SpeechToText.normalise("XIN CHÀO", lowercase=False) == "XIN CHÀO"


def test_whitespace_is_collapsed() -> None:
    assert SpeechToText.normalise("  xin   chào\n bạn  ") == "xin chào bạn"


def test_empty_input_is_safe() -> None:
    assert SpeechToText.normalise("") == ""


def test_output_is_nfc_normalised() -> None:
    decomposed = "xin cha\u0300o"  # 'a' + combining grave accent
    assert SpeechToText.normalise(decomposed) == "xin chào"


# -- inference ------------------------------------------------------------------------
def test_silence_transcribes_to_nothing(real_config: JarvisConfig, requires_models: None) -> None:
    stt = SpeechToText(real_config.stt)
    silence = np.zeros(real_config.stt.sample_rate, dtype=np.float32)
    assert stt.transcribe(silence) == ""


def test_empty_audio_returns_empty_string(
    real_config: JarvisConfig, requires_models: None
) -> None:
    stt = SpeechToText(real_config.stt)
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""


@pytest.mark.parametrize(
    "sentence",
    [
        "Xin chào, hôm nay trời rất đẹp.",
        "Mở thư mục tải về giúp tôi.",
        "Thời tiết Hà Nội hôm nay thế nào.",
    ],
)
def test_round_trip_tts_then_stt(
    real_config: JarvisConfig,
    requires_models: None,
    tmp_path: Path,
    sentence: str,
) -> None:
    """TTS -> WAV -> STT should recover most of the Vietnamese words.

    VITS samples noise on every call, so the audio is not bit-identical between
    runs and an occasional mis-synthesised syllable is expected. We therefore
    assert on word *overlap* rather than an exact match, which still catches a
    genuinely broken recogniser (overlap would collapse toward zero).
    """
    tts = TextToSpeech(real_config.tts, Speaker(real_config.audio))
    audio = tts.synthesize(sentence)
    wav_path = write_wav(tmp_path / "spoken.wav", audio.samples, audio.sample_rate)

    transcript = SpeechToText(real_config.stt).transcribe_file(wav_path)
    assert transcript, "transcript không được rỗng"

    expected = SpeechToText.normalise(sentence.replace(",", "").replace(".", "")).split()
    produced = set(transcript.split())
    overlap = sum(1 for word in expected if word in produced) / len(expected)

    assert overlap >= 0.7, f"chỉ khớp {overlap:.0%} từ: {transcript!r} vs {sentence!r}"


def test_missing_model_file_reports_download_command(real_config: JarvisConfig) -> None:
    broken = real_config.stt.model_copy(update={"encoder": Path("khong/ton/tai.onnx")})
    with pytest.raises(SttError, match="download_models.py"):
        SpeechToText(broken).load()


def test_transcribe_missing_wav_raises(real_config: JarvisConfig, requires_models: None) -> None:
    from jarvis.audio import AudioError

    with pytest.raises(AudioError):
        SpeechToText(real_config.stt).transcribe_file("khong_co_file_nay.wav")
