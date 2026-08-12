"""mvp.md Tasks 5 & 12: end-to-end orchestration and fault injection.

All heavy components (STT, TTS, LLM) are replaced with fakes, so these tests run
fast and assert the *control flow*: what gets spoken, which state the tray shows,
and that no single failure crashes the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio import AudioError
from jarvis.config import JarvisConfig
from jarvis.llm import LlmError, LlmUnavailableError
from jarvis.pipeline import Pipeline
from jarvis.state import State, StateMachine
from jarvis.stt import SttError
from jarvis.tts import TtsError

from .conftest import FakeMic, silence_frame, speech_frame


class SpeechRecorder:
    """Captures everything the pipeline tries to say."""

    def __init__(self, fail: bool = False) -> None:
        self.said: list[str] = []
        self.fail = fail

    def speak(self, text: str):  # noqa: ANN202
        self.said.append(text)
        if self.fail:
            raise TtsError("loa hỏng")
        return type("R", (), {"samples": np.zeros(10), "sample_rate": 22050, "duration": 0.1})()


@pytest.fixture
def pipeline(real_config: JarvisConfig, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    states: list[State] = []
    machine = StateMachine(State.IDLE)
    machine.subscribe(lambda state, _detail: states.append(state))

    pipe = Pipeline(real_config, machine)
    pipe.states_seen = states  # type: ignore[attr-defined]

    # Never touch real hardware or models.
    pipe.mic = FakeMic([])  # type: ignore[assignment]
    pipe.tts = SpeechRecorder()  # type: ignore[assignment]
    monkeypatch.setattr(pipe.speaker, "beep", lambda *a, **k: None)
    monkeypatch.setattr(pipe.speaker, "play", lambda *a, **k: None)
    pipe._tool_defs = []  # noqa: SLF001 - skip lmstudio tool schema building
    return pipe


# -- happy path -----------------------------------------------------------------------
def test_full_turn_transcribes_then_speaks_the_reply(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "mấy giờ rồi")
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "Bây giờ là 3 giờ chiều.")

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert result.ok
    assert result.transcript == "mấy giờ rồi"
    assert result.reply == "Bây giờ là 3 giờ chiều."
    assert pipeline.tts.said == ["Bây giờ là 3 giờ chiều."]  # type: ignore[attr-defined]


def test_state_cycle_of_one_turn(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "xin chào")
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "Chào bạn.")

    pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    seen = pipeline.states_seen  # type: ignore[attr-defined]
    assert State.THINKING in seen
    assert State.SPEAKING in seen
    assert seen[-1] is State.IDLE


def test_tools_used_are_reported(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "mở downloads")

    def fake_act(_prompt, _tools, **_kwargs):  # noqa: ANN202
        pipeline.tools.last_used.append("run_system_command")
        return "Đã mở thư mục Downloads."

    monkeypatch.setattr(pipeline.llm, "act", fake_act)
    result = pipeline.respond_to_text("mở downloads")
    assert result.tools_used == ["run_system_command"]


# -- degraded paths (Task 12) ----------------------------------------------------------
def test_empty_transcript_asks_the_user_to_repeat(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "")

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert not result.ok
    assert pipeline.tts.said == [pipeline.config.replies.not_understood]  # type: ignore[attr-defined]


def test_lm_studio_offline_speaks_a_friendly_message(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "xin chào")

    def boom(*_args, **_kwargs):
        raise LlmUnavailableError("LM Studio offline")

    monkeypatch.setattr(pipeline.llm, "act", boom)

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert not result.ok
    assert pipeline.tts.said == [pipeline.config.replies.llm_unavailable]  # type: ignore[attr-defined]
    assert pipeline.state.state is State.ERROR


def test_llm_error_falls_back_to_generic_apology(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "xin chào")
    monkeypatch.setattr(
        pipeline.llm, "act", lambda *a, **k: (_ for _ in ()).throw(LlmError("timeout"))
    )

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert not result.ok
    assert pipeline.tts.said == [pipeline.config.replies.generic_error]  # type: ignore[attr-defined]


def test_stt_failure_does_not_crash(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline.stt,
        "transcribe",
        lambda *a, **k: (_ for _ in ()).throw(SttError("model hỏng")),
    )

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert not result.ok
    assert "model hỏng" in (result.error or "")


def test_tts_failure_still_returns_the_reply(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline.tts = SpeechRecorder(fail=True)  # type: ignore[assignment]
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "xin chào")
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "Chào bạn.")

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert result.reply == "Chào bạn."
    assert result.spoken is False, "TTS lỗi thì spoken=False nhưng pipeline vẫn sống"


def test_empty_llm_reply_is_reported(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "")
    result = pipeline.respond_to_text("xin chào")
    assert not result.ok
    assert pipeline.tts.said == [pipeline.config.replies.generic_error]  # type: ignore[attr-defined]


# -- capture ---------------------------------------------------------------------------
def test_capture_returns_samples_when_someone_speaks(pipeline: Pipeline) -> None:
    # 12 speech frames (~1 s) clears min_record_seconds, then enough silence to
    # satisfy silence_timeout_ms from the real config.
    pipeline.mic = FakeMic([speech_frame()] * 12 + [silence_frame()] * 20)  # type: ignore[assignment]
    pipeline._speech_threshold = 0.05  # noqa: SLF001
    samples = pipeline._capture(use_pre_roll=False)  # noqa: SLF001
    assert samples is not None
    assert samples.size > 0


def test_capture_returns_none_and_prompts_when_nobody_speaks(pipeline: Pipeline) -> None:
    pipeline.mic = FakeMic([silence_frame()] * 200)  # type: ignore[assignment]
    pipeline._speech_threshold = 0.05  # noqa: SLF001
    assert pipeline._capture(use_pre_roll=False) is None  # noqa: SLF001
    assert pipeline.tts.said == [pipeline.config.replies.not_understood]  # type: ignore[attr-defined]


def test_capture_reports_mic_failure(pipeline: Pipeline) -> None:
    pipeline.mic = FakeMic([])  # read() returns None immediately  # type: ignore[assignment]
    pipeline._speech_threshold = 0.05  # noqa: SLF001
    assert pipeline._capture(use_pre_roll=False) is None  # noqa: SLF001
    assert pipeline.tts.said == [pipeline.config.replies.mic_error]  # type: ignore[attr-defined]


def test_mic_error_during_capture_is_caught(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise AudioError("mic bị rút")

    monkeypatch.setattr("jarvis.pipeline.record_utterance", boom)
    with pytest.raises(AudioError):
        pipeline._capture(use_pre_roll=False)  # noqa: SLF001 - the loop catches this


def test_request_stop_flags_the_loop(pipeline: Pipeline) -> None:
    assert not pipeline.stopping
    pipeline.request_stop()
    assert pipeline.stopping
