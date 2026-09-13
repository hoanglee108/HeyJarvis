"""mvp.md Tasks 5 & 12: end-to-end orchestration and fault injection.

All heavy components (STT, TTS, LLM) are replaced with fakes, so these tests run
fast and assert the *control flow*: what gets spoken, which state the tray shows,
and that no single failure crashes the pipeline.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from jarvis.audio import AudioError
from jarvis.config import DuckingConfig, JarvisConfig
from jarvis.ducking import VolumeDucker
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


def build_pipeline(config: JarvisConfig, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    states: list[State] = []
    machine = StateMachine(State.IDLE)
    machine.subscribe(lambda state, _detail: states.append(state))

    pipe = Pipeline(config, machine)
    pipe.states_seen = states  # type: ignore[attr-defined]

    # Never touch real hardware or models. Ducking counts as hardware: the real config
    # enables it, and a live VolumeDucker would move the volume of whatever the
    # developer happens to be playing while the suite runs.
    pipe.mic = FakeMic([])  # type: ignore[assignment]
    pipe.tts = SpeechRecorder()  # type: ignore[assignment]
    pipe.ducker = VolumeDucker(DuckingConfig(enabled=False))
    monkeypatch.setattr(pipe.speaker, "beep", lambda *a, **k: None)
    monkeypatch.setattr(pipe.speaker, "play", lambda *a, **k: None)
    pipe._tool_defs = []  # noqa: SLF001 - skip lmstudio tool schema building
    return pipe


@pytest.fixture
def pipeline(real_config: JarvisConfig, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    return build_pipeline(real_config, monkeypatch)


# -- happy path -----------------------------------------------------------------------
def test_full_turn_transcribes_then_speaks_the_reply(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An open-ended request, so this exercises the LLM path rather than the router.
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "kể tôi nghe một câu chuyện")
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "Ngày xưa có một chú mèo.")

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert result.ok
    assert result.transcript == "kể tôi nghe một câu chuyện"
    assert result.reply == "Ngày xưa có một chú mèo."
    assert pipeline.tts.said == ["Ngày xưa có một chú mèo."]  # type: ignore[attr-defined]


def test_state_cycle_of_one_turn(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "xin chào")
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "Chào bạn.")

    pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    seen = pipeline.states_seen  # type: ignore[attr-defined]
    assert State.THINKING in seen
    assert State.SPEAKING in seen
    assert seen[-1] is State.IDLE


def _forbid_llm(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("the intent router should have handled this without the LLM")

    monkeypatch.setattr(pipeline.llm, "act", boom)
    monkeypatch.setattr(pipeline.llm, "chat", boom)


def test_datetime_question_bypasses_llm_to_avoid_hallucination(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small local LLM has no notion of "now" and will confidently invent a date."""
    monkeypatch.setattr(pipeline.stt, "transcribe", lambda *a, **k: "mấy giờ rồi")
    monkeypatch.setattr(pipeline.tools.clock, "describe", lambda: "Bây giờ là 21 giờ 20 phút.")
    _forbid_llm(pipeline, monkeypatch)

    result = pipeline.process_audio(np.zeros(16000, dtype=np.float32), 16000)

    assert result.ok
    assert result.reply == "Bây giờ là 21 giờ 20 phút."
    assert result.tools_used == ["get_current_datetime"]
    assert pipeline.tts.said == ["Bây giờ là 21 giờ 20 phút."]  # type: ignore[attr-defined]


def test_open_app_command_bypasses_the_llm(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.tools.apps, "open", lambda alias: "Đã mở YouTube.")
    _forbid_llm(pipeline, monkeypatch)

    result = pipeline.respond_to_text("mở youtube giúp tôi")

    assert result.reply == "Đã mở YouTube."
    assert result.tools_used == ["open_application"]


def test_play_song_command_bypasses_the_llm_and_keeps_the_full_title(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    played: list[str] = []
    monkeypatch.setattr(
        pipeline.tools.music,
        "play",
        lambda song: played.append(song) or f"Đang phát {song}.",
    )
    _forbid_llm(pipeline, monkeypatch)

    result = pipeline.respond_to_text("phát bài Em của ngày hôm qua")

    assert played == ["Em của ngày hôm qua"]
    assert result.tools_used == ["play_song"]


def test_open_ended_request_still_reaches_the_llm(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.llm, "act", lambda *a, **k: "Tôi đã tìm được thông tin.")
    result = pipeline.respond_to_text("tìm thông tin về LM Studio")
    assert result.reply == "Tôi đã tìm được thông tin."


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
    pipeline.mic = FakeMic([speech_frame()] * 12 + [silence_frame()] * 30)  # type: ignore[assignment]
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


# -- streamed replies (priority.md P1-1) -----------------------------------------------
def streaming_act(reply: str, *, chunk_size: int = 7):
    """Fake ``llm.act`` that dribbles ``reply`` into the sink like a real stream."""

    def act(_prompt, _tools=None, *, sink=None, **_kwargs):  # noqa: ANN202
        if sink is not None:
            for start in range(0, len(reply), chunk_size):
                sink.push(reply[start : start + chunk_size])
        return reply

    return act


def test_streamed_reply_is_spoken_sentence_by_sentence(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = "Bây giờ là chín giờ sáng. Trời hôm nay khá đẹp. Bạn nên ra ngoài đi bộ."
    monkeypatch.setattr(pipeline.llm, "act", streaming_act(reply))

    result = pipeline.respond_to_text("kể tôi nghe một câu chuyện")

    assert result.reply == reply
    assert result.spoken is True
    # Three separate synthesis calls: the first one starts playing while the model is
    # still writing the rest. That is the whole point of P1-1.
    assert pipeline.tts.said == [  # type: ignore[attr-defined]
        "Bây giờ là chín giờ sáng.",
        "Trời hôm nay khá đẹp.",
        "Bạn nên ra ngoài đi bộ.",
    ]


def test_streamed_reply_is_never_spoken_twice(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback to a plain ``say`` must not re-read what was already played."""
    reply = "Tôi đã mở thư mục tải xuống cho bạn rồi nhé."
    monkeypatch.setattr(pipeline.llm, "act", streaming_act(reply))

    pipeline.respond_to_text("mở thư mục tải xuống")

    said = pipeline.tts.said  # type: ignore[attr-defined]
    assert said.count(reply) <= 1
    assert len(said) == 1


def test_streamed_turn_reaches_speaking_then_idle(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.llm, "act", streaming_act("Chào bạn, tôi nghe đây nhé."))

    pipeline.respond_to_text("xin chào")

    seen = pipeline.states_seen  # type: ignore[attr-defined]
    assert State.THINKING in seen
    assert State.SPEAKING in seen
    assert seen[-1] is State.IDLE


def test_reply_with_no_speakable_content_falls_back_to_say(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was streamed to the speaker, so the plain path has to take over."""

    def act(_prompt, _tools=None, *, sink=None, **_kwargs):  # noqa: ANN202
        if sink is not None:
            sink.push("<think>chỉ có suy luận, không có câu trả lời</think>")
        return "Đây mới là câu trả lời thật."

    monkeypatch.setattr(pipeline.llm, "act", act)

    result = pipeline.respond_to_text("xin chào")

    assert result.spoken is True
    assert pipeline.tts.said == ["Đây mới là câu trả lời thật."]  # type: ignore[attr-defined]


def test_streaming_disabled_speaks_the_whole_reply_at_once(
    config_copy: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_copy.llm.stream = False
    pipe = build_pipeline(config_copy, monkeypatch)
    reply = "Câu một ở đây. Câu hai ở đây."
    monkeypatch.setattr(pipe.llm, "act", streaming_act(reply))

    result = pipe.respond_to_text("xin chào")

    assert result.spoken is True
    assert pipe.tts.said == [reply]  # type: ignore[attr-defined]


def test_llm_failure_during_streaming_does_not_leak_the_speech_thread(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = threading.active_count()

    def boom(_prompt, _tools=None, *, sink=None, **_kwargs):  # noqa: ANN202
        if sink is not None:
            sink.push("Một phần câu trả lời")
        raise LlmError("mất kết nối giữa lúc trả lời")

    monkeypatch.setattr(pipeline.llm, "act", boom)

    result = pipeline.respond_to_text("xin chào")

    assert not result.ok
    assert pipeline.tts.said[-1] == pipeline.config.replies.generic_error  # type: ignore[attr-defined]
    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before, "luồng phát giọng nói bị rò"


def test_unexpected_exception_still_closes_the_speech_worker(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = threading.active_count()

    def boom(_prompt, _tools=None, *, sink=None, **_kwargs):  # noqa: ANN202
        raise RuntimeError("lỗi không ai lường trước")

    monkeypatch.setattr(pipeline.llm, "act", boom)

    with pytest.raises(RuntimeError):
        pipeline.respond_to_text("xin chào")

    deadline = time.monotonic() + 5.0
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before


# -- ducking is wired to the right phases (priority.md P1-3) ---------------------------
class DuckSpy:
    """Counts acquire/release without touching the system volume."""

    def __init__(self) -> None:
        self.acquires = 0
        self.releases = 0
        self.restores = 0
        self.enabled = True

    def acquire(self) -> None:
        self.acquires += 1

    def release(self) -> None:
        self.releases += 1

    def restore_all(self) -> None:
        self.restores += 1

    def ducked(self, *, active: bool = True):  # noqa: ANN202
        from contextlib import contextmanager

        @contextmanager
        def scope():  # noqa: ANN202
            if active:
                self.acquire()
            try:
                yield
            finally:
                if active:
                    self.release()

        return scope()


def test_waiting_for_the_wake_word_never_ducks(pipeline: Pipeline) -> None:
    """Standby can last for hours; holding the music down through it is unacceptable."""
    spy = DuckSpy()
    pipeline.ducker = spy  # type: ignore[assignment]
    pipeline.mic = FakeMic([speech_frame()] * 12 + [silence_frame()] * 30)  # type: ignore[assignment]
    pipeline._speech_threshold = 0.05  # noqa: SLF001

    pipeline._capture(use_pre_roll=False, duck=False)  # noqa: SLF001

    assert spy.acquires == 0


def test_capturing_a_command_ducks_and_restores(pipeline: Pipeline) -> None:
    spy = DuckSpy()
    pipeline.ducker = spy  # type: ignore[assignment]
    pipeline.mic = FakeMic([speech_frame()] * 12 + [silence_frame()] * 30)  # type: ignore[assignment]
    pipeline._speech_threshold = 0.05  # noqa: SLF001

    pipeline._capture(use_pre_roll=False, duck=True)  # noqa: SLF001

    assert spy.acquires == 1
    assert spy.releases == 1


def test_close_restores_the_volume(pipeline: Pipeline) -> None:
    spy = DuckSpy()
    pipeline.ducker = spy  # type: ignore[assignment]
    pipeline.close()
    assert spy.restores == 1


def test_streamed_playback_releases_the_duck_exactly_once(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = DuckSpy()
    pipeline.ducker = spy  # type: ignore[assignment]
    monkeypatch.setattr(pipeline.llm, "act", streaming_act("Một câu trả lời đủ dài đây."))

    pipeline.respond_to_text("xin chào")

    assert spy.acquires == 1
    assert spy.releases == 1
