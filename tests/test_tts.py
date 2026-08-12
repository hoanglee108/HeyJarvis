"""mvp.md Task 4: text cleanup plus a real synthesis smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.audio import Speaker
from jarvis.config import JarvisConfig
from jarvis.tts import TextToSpeech, TtsError, clean_for_speech


# -- text cleanup ---------------------------------------------------------------------
def test_markdown_is_removed() -> None:
    assert clean_for_speech("**Xin chào** _bạn_") == "Xin chào bạn"


def test_bullets_become_sentences() -> None:
    cleaned = clean_for_speech("- Việc một\n- Việc hai")
    assert "-" not in cleaned
    assert "Việc một" in cleaned and "Việc hai" in cleaned


def test_emoji_are_removed() -> None:
    assert "🎉" not in clean_for_speech("Xong rồi 🎉")


def test_code_fences_and_backticks_are_removed() -> None:
    assert "`" not in clean_for_speech("Chạy `pip install` nhé")


def test_empty_input_stays_empty() -> None:
    assert clean_for_speech("   \n  ") == ""
    assert clean_for_speech("") == ""


def test_vietnamese_diacritics_are_preserved() -> None:
    text = "Hôm nay trời Hà Nội rất đẹp"
    assert clean_for_speech(text) == text


# -- synthesis (needs the downloaded voice) --------------------------------------------
def test_synthesize_produces_audio(
    real_config: JarvisConfig, requires_models: None, tmp_path: Path
) -> None:
    tts = TextToSpeech(real_config.tts, Speaker(real_config.audio))
    result = tts.synthesize("Xin chào, tôi là Jarvis.")

    assert result.samples.size > 0
    assert result.sample_rate >= 16000
    assert 0.5 < result.duration < 15.0, f"độ dài audio bất thường: {result.duration}s"


def test_save_writes_a_playable_wav(
    real_config: JarvisConfig, requires_models: None, tmp_path: Path
) -> None:
    import soundfile as sf

    tts = TextToSpeech(real_config.tts, Speaker(real_config.audio))
    out = tts.save("Một hai ba bốn năm.", tmp_path / "out.wav")

    assert out.is_file()
    assert out.stat().st_size > 1000
    info = sf.info(str(out))
    assert info.channels == 1
    assert info.duration > 0.3


def test_empty_text_yields_no_audio(real_config: JarvisConfig, requires_models: None) -> None:
    tts = TextToSpeech(real_config.tts, Speaker(real_config.audio))
    assert tts.synthesize("   ").samples.size == 0


def test_saving_empty_text_raises(
    real_config: JarvisConfig, requires_models: None, tmp_path: Path
) -> None:
    tts = TextToSpeech(real_config.tts, Speaker(real_config.audio))
    with pytest.raises(TtsError):
        tts.save("***", tmp_path / "nope.wav")


def test_missing_model_file_reports_download_command(real_config: JarvisConfig) -> None:
    broken = real_config.tts.model_copy(update={"model": Path("khong/ton/tai.onnx")})
    tts = TextToSpeech(broken, Speaker(real_config.audio))
    with pytest.raises(TtsError, match="download_models.py"):
        tts.load()
