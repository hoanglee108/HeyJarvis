"""End-to-end orchestration: wake word -> STT -> LLM (+tools) -> TTS.

This is the module that mvp.md Task 5 wires up and Tasks 6 and 12 harden. Every
stage is wrapped so a single failure (LM Studio down, mic unplugged, a tool raising)
degrades into a spoken apology instead of killing the process.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import (
    AudioError,
    MicStream,
    Speaker,
    calibrate_noise_floor,
    record_utterance,
    write_wav,
)
from .config import JarvisConfig
from .ducking import VolumeDucker
from .intents import IntentRouter
from .llm import LLMClient, LlmError, LlmUnavailableError, ToolSpec
from .logging_setup import get_logger
from .state import State, StateMachine
from .streaming import StreamingSpeaker
from .stt import SpeechToText, SttError
from .tools import ToolBox
from .tts import TextToSpeech, TtsError
from .vad import SpeechGate, build_speech_gate
from .wakeword import WakeWordDetector, WakeWordError

log = get_logger("jarvis.pipeline")


@dataclass
class TurnResult:
    """Outcome of one user utterance."""

    transcript: str = ""
    reply: str = ""
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    spoken: bool = False
    conversation_ended: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class Pipeline:
    """Owns every component and the conversation loop."""

    def __init__(self, config: JarvisConfig, state: StateMachine | None = None) -> None:
        self.config = config
        self.state = state or StateMachine()
        self.speaker = Speaker(config.audio)
        self.stt = SpeechToText(config.stt)
        self.tts = TextToSpeech(config.tts, self.speaker)
        self.llm = LLMClient(config.llm)
        self.tools = ToolBox(config)
        self.intents = IntentRouter(self.tools)
        self.wake = WakeWordDetector(config.wake_word)
        self.mic = MicStream(config.audio)
        self.ducker = VolumeDucker(config.audio.ducking)

        self._stop_event = threading.Event()
        self._speech_threshold = config.audio.silence_rms_threshold or 0.01
        #: Built in :meth:`_prepare_mic`, once the noise floor is known.
        self._gate: SpeechGate | None = None
        self._tool_defs: list[ToolSpec] | None = None
        #: State captured when streamed playback starts, restored when it ends.
        self._stream_previous_state = State.IDLE
        self._stream_previous_detail: str | None = None
        #: True while a streamed reply holds a duck, so it is released exactly once.
        self._stream_ducked = False

    # ----------------------------------------------------------------------------
    # lifecycle
    # ----------------------------------------------------------------------------
    def warmup(self, *, include_llm: bool = True, include_wake: bool = True) -> None:
        """Load models up front so the first reply is not slowed by cold start."""
        self.state.set(State.STARTING, "đang nạp model")
        self.stt.load()
        self.tts.load()
        if (
            include_wake
            and self.config.wake_word.enabled
            and self.config.wake_word.stt_phrase is None
        ):
            self.wake.load()
        if include_llm:
            try:
                log.info("LM Studio: %s", self.llm.ping())
            except LlmError as exc:
                # Not fatal: the user may start LM Studio after Jarvis.
                log.warning("%s", exc)

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def close(self) -> None:
        self.request_stop()
        # Before anything else: leaving somebody's music at 15% would be a nasty
        # parting gift if we are shutting down mid-turn.
        self.ducker.restore_all()
        self.mic.stop()
        self.llm.close()
        self.state.set(State.STOPPED)

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # ----------------------------------------------------------------------------
    # speaking helpers
    # ----------------------------------------------------------------------------
    def say(self, text: str, *, resume_state: State | None = None) -> bool:
        """Speak ``text``; returns False when synthesis/playback failed.

        ``resume_state`` keeps a hands-free conversation active after playback
        instead of returning to wake-word standby.
        """
        if not text.strip():
            return False
        previous = self.state.state
        previous_detail = self.state.detail
        self.state.set(State.SPEAKING, text[:60])
        try:
            with self.ducker.ducked(active=self.config.audio.ducking.while_speaking):
                self.tts.speak(text)
            return True
        except (TtsError, AudioError) as exc:
            log.error("Không nói được: %s", exc)
            print(f"[Jarvis] {text}")  # last-resort feedback channel
            return False
        finally:
            self._after_speaking(previous, previous_detail, resume_state)

    def _after_speaking(
        self,
        previous: State,
        previous_detail: str | None,
        resume_state: State | None,
    ) -> None:
        """State restoration and microphone cleanup shared by every speaking path."""
        # Keep an ERROR indication visible; otherwise restore the requested
        # conversation state or go back to ordinary wake-word standby.
        if previous is State.ERROR:
            self.state.set(State.ERROR, previous_detail)
        elif resume_state is not None:
            self.state.set(resume_state)
        elif previous is not State.SPEAKING:
            self.state.set(State.IDLE)
        # Discard the audio the microphone picked up from our own voice.
        if self.mic.running:
            self.mic.drain()
            if self.wake.loaded:
                self.wake.reset()
            if self._gate is not None:
                # Silero carries LSTM state; after a gap in the audio timeline it must
                # not keep reasoning from frames that were thrown away.
                self._gate.reset()

    # ----------------------------------------------------------------------------
    # one turn
    # ----------------------------------------------------------------------------
    def respond_to_text(self, text: str, *, keep_listening: bool = False) -> TurnResult:
        """Intent router, else LLM (with tools) -> spoken reply.

        Used by both the voice loop and ``jarvis chat``, so a command answered
        deterministically behaves identically in either entry point.
        """
        result = TurnResult(transcript=text)
        resume_state = State.LISTENING if keep_listening else None
        self.state.set(State.THINKING, text[:60])
        self.tools.reset_usage()

        direct = self.intents.route(text)
        if direct is not None:
            result.reply = direct.reply
            result.tools_used = list(self.tools.last_used)
            result.spoken = self.say(direct.reply, resume_state=resume_state)
            if not keep_listening:
                self.state.set(State.IDLE)
            return result

        if self._tool_defs is None:
            self._tool_defs = self.tools.build_tool_defs()

        speaker = self._start_streaming_speech()
        try:
            try:
                reply = self.llm.act(text, self._tool_defs, sink=speaker)
            except LlmUnavailableError as exc:
                log.error("LM Studio không khả dụng: %s", exc)
                result.error = str(exc)
                self.state.set(State.ERROR, "LM Studio offline")
                result.spoken = self.say(
                    self.config.replies.llm_unavailable, resume_state=resume_state
                )
                return result
            except LlmError as exc:
                log.error("Lỗi LLM: %s", exc)
                result.error = str(exc)
                self.state.set(State.ERROR, "lỗi LLM")
                result.spoken = self.say(
                    self.config.replies.generic_error, resume_state=resume_state
                )
                return result

            result.tools_used = list(self.tools.last_used)
            if not reply:
                result.error = "LLM trả về nội dung rỗng"
                result.spoken = self.say(
                    self.config.replies.generic_error, resume_state=resume_state
                )
                return result

            result.reply = reply
            result.spoken = self._finish_streaming_speech(
                speaker, reply, resume_state=resume_state
            )
            if not keep_listening:
                self.state.set(State.IDLE)
            return result
        finally:
            # Every exit path, including an unplanned exception, must stop the speech
            # worker and give the music back.
            self._close_streaming_speech(speaker)

    # ----------------------------------------------------------------------------
    # streaming speech (priority.md P1-1)
    # ----------------------------------------------------------------------------
    def _start_streaming_speech(self) -> StreamingSpeaker | None:
        """A sink that speaks the answer as it is written, or ``None`` when disabled."""
        if not self.config.llm.stream:
            return None
        speaker = StreamingSpeaker(self.tts, on_first_audio=self._on_stream_first_audio)
        speaker.start()
        return speaker

    def _on_stream_first_audio(self) -> None:
        """Called on the speech worker thread, just before the first sentence plays."""
        self._stream_previous_state = self.state.state
        self._stream_previous_detail = self.state.detail
        self.state.set(State.SPEAKING, "đang trả lời")
        if self.config.audio.ducking.while_speaking:
            self.ducker.acquire()
            self._stream_ducked = True

    def _close_streaming_speech(self, speaker: StreamingSpeaker | None) -> None:
        """Idempotent cleanup: stop the worker and undo the duck exactly once."""
        if speaker is not None:
            speaker.ensure_closed()
        if self._stream_ducked:
            self._stream_ducked = False
            self.ducker.release()

    def _finish_streaming_speech(
        self,
        speaker: StreamingSpeaker | None,
        reply: str,
        *,
        resume_state: State | None,
    ) -> bool:
        """Wait for streamed playback; fall back to a normal ``say`` when it spoke nothing.

        The fallback is only safe *because* nothing was spoken. Re-speaking a reply that
        was already half played would repeat sentences, which is why this branches on
        ``spoken_any`` rather than on whether an error occurred.
        """
        if speaker is None:
            return self.say(reply, resume_state=resume_state)

        previous = self._stream_previous_state
        previous_detail = self._stream_previous_detail
        ok = speaker.finish(timeout=self.config.llm.request_timeout_s)

        if not speaker.spoken_any:
            # No audio at all: a markup-only reply, or synthesis failed before the first
            # sentence, or the server never streamed. Speak it the old way.
            if speaker.error is not None:
                log.warning("Streaming TTS lỗi trước khi phát: %s", speaker.error)
            return self.say(reply, resume_state=resume_state)

        if speaker.error is not None:
            log.error("Dừng giữa câu trả lời: %s", speaker.error)
            print(f"[Jarvis] {reply}")
        self._after_speaking(previous, previous_detail, resume_state)
        return ok

    @staticmethod
    def _normalise_phrase(text: str) -> str:
        """Case-fold text, remove accents/punctuation, and collapse whitespace."""
        decomposed = unicodedata.normalize("NFD", text.casefold())
        accentless = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
        return " ".join(re.sub(r"[^\w]+", " ", accentless).split())

    def _matches_phrase(self, transcript: str, phrases: list[str] | tuple[str, ...]) -> bool:
        normalised = self._normalise_phrase(transcript)
        for phrase in phrases:
            candidate = self._normalise_phrase(phrase)
            if candidate and re.search(rf"(?:^|\s){re.escape(candidate)}(?:$|\s)", normalised):
                return True
        return False

    def _is_conversation_end(self, transcript: str) -> bool:
        return self._matches_phrase(transcript, self.config.wake_word.conversation_end_phrases)

    @property
    def _wake_phrase(self) -> str:
        return self.config.wake_word.stt_phrase or "Hey Jarvis"

    def _waiting_message(self, *, include_exit_hint: bool = False) -> str:
        suffix = " (Ctrl+C để thoát)" if include_exit_hint else ""
        return f"Đang chờ '{self._wake_phrase}'…{suffix}"

    def process_audio(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        *,
        keep_listening: bool = False,
    ) -> TurnResult:
        """Transcribe an utterance and answer it.

        In a hands-free conversation, the configured end phrase bypasses the LLM
        and ends the session; all other turns leave the microphone listening for
        the next one.
        """
        resume_state = State.LISTENING if keep_listening else None
        self.state.set(State.THINKING, "đang nhận dạng")
        try:
            transcript = self.stt.transcribe(samples, sample_rate)
        except SttError as exc:
            log.error("Lỗi STT: %s", exc)
            self.state.set(State.ERROR, "lỗi STT")
            spoken = self.say(self.config.replies.generic_error, resume_state=resume_state)
            return TurnResult(error=str(exc), spoken=spoken)

        if not transcript or len(transcript) < 2:
            log.info("Không nhận được nội dung nào")
            spoken = self.say(self.config.replies.not_understood, resume_state=resume_state)
            if not keep_listening:
                self.state.set(State.IDLE)
            return TurnResult(error="transcript rỗng", spoken=spoken)

        print(f"Bạn: {transcript}")

        if keep_listening and self._is_conversation_end(transcript):
            goodbye = "Tạm biệt."
            log.info("Kết thúc phiên hội thoại qua câu lệnh: %r", transcript)
            result = TurnResult(
                transcript=transcript,
                reply=goodbye,
                spoken=self.say(goodbye),
                conversation_ended=True,
            )
            print(f"Jarvis: {goodbye}")
            return result

        result = self.respond_to_text(transcript, keep_listening=keep_listening)
        if result.reply:
            print(f"Jarvis: {result.reply}")
        return result

    def process_wav(self, path: str | Path) -> TurnResult:
        """Integration-test friendly entry point: a fixed WAV instead of the mic."""
        from .audio import read_wav

        samples, rate = read_wav(path, target_rate=self.config.stt.sample_rate)
        return self.process_audio(samples, rate)

    # ----------------------------------------------------------------------------
    # recording
    # ----------------------------------------------------------------------------
    def _capture(
        self,
        *,
        use_pre_roll: bool,
        prompt_on_no_speech: bool = True,
        duck: bool = False,
    ) -> np.ndarray | None:
        """Record one utterance; ``None`` when nobody spoke.

        A continuous conversation keeps listening silently when the user pauses;
        ordinary wake-word mode retains the existing spoken retry prompt.

        ``duck`` lowers other apps' volume for the duration. It must stay False while
        waiting for the wake word - that phase can run for hours, and holding the music
        down through all of it would be absurd.
        """
        self.state.set(State.LISTENING)
        if self.config.audio.beep_on_listen:
            self.speaker.beep()
            self.mic.drain()

        pre_roll = self.mic.recent(self.config.audio.pre_roll_ms) if use_pre_roll else None
        with self.ducker.ducked(active=duck):
            recording = record_utterance(
                self.mic,
                self.config.audio,
                gate=self._gate,
                speech_threshold=self._speech_threshold,
                pre_roll=pre_roll,
            )
        if recording.reason == "no_speech":
            if prompt_on_no_speech:
                log.info("Không có ai nói, quay lại chờ")
                self.say(self.config.replies.not_understood)
                self.state.set(State.IDLE)
            else:
                log.info("Không có ai nói, vẫn giữ phiên hội thoại")
            return None
        if recording.reason == "mic_timeout":
            log.error("Mic không trả về dữ liệu")
            self.state.set(State.ERROR, "mic timeout")
            self.say(self.config.replies.mic_error)
            return None
        return recording.samples

    def save_last_recording(self, samples: np.ndarray, path: str | Path) -> Path:
        return write_wav(path, samples, self.config.audio.sample_rate)

    # ----------------------------------------------------------------------------
    # loops
    # ----------------------------------------------------------------------------
    def _prepare_mic(self) -> None:
        self.mic.start()
        self._speech_threshold = calibrate_noise_floor(self.mic, self.config.audio)
        self._gate = build_speech_gate(
            self.config.audio, energy_threshold=self._speech_threshold
        )
        log.info("Cổng phát hiện giọng nói: %s", self._gate.describe())

    def run_push_to_talk(self) -> None:
        """Task 5 loop: press Enter, speak, get an answer. No wake word."""
        self.warmup(include_wake=False)
        self._prepare_mic()
        self.state.set(State.IDLE, "nhấn Enter để nói")
        print("\nNhấn Enter rồi nói (Ctrl+C để thoát).")

        while not self.stopping:
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                break
            self.mic.drain()
            try:
                samples = self._capture(use_pre_roll=False, duck=True)
            except AudioError as exc:
                log.error("Lỗi mic: %s", exc)
                self.say(self.config.replies.mic_error)
                continue
            if samples is None:
                print("Nhấn Enter để thử lại.")
                continue
            self.process_audio(samples)
            self.state.set(State.IDLE, "nhấn Enter để nói")
            print("\nNhấn Enter để nói tiếp.")

    def _run_conversation(self) -> None:
        """Keep serving turns after a wake word until the configured end phrase."""
        end_phrases = ", ".join(repr(phrase) for phrase in self.config.wake_word.conversation_end_phrases)
        print(f"Đang nghe. Nói {end_phrases} để quay lại chế độ chờ.")
        use_pre_roll = True

        while not self.stopping:
            try:
                samples = self._capture(
                    use_pre_roll=use_pre_roll,
                    prompt_on_no_speech=False,
                    duck=True,
                )
            except AudioError as exc:
                log.error("Lỗi mic khi ghi âm: %s", exc)
                self.say(self.config.replies.mic_error, resume_state=State.LISTENING)
                self._restart_mic()
                use_pre_roll = False
                continue

            use_pre_roll = False
            if samples is None:
                continue

            result = self.process_audio(samples, keep_listening=True)
            if not result.conversation_ended:
                continue

            self.state.set(State.IDLE)
            self.wake.reset()
            self.mic.drain()
            print(self._waiting_message())
            return

    def _run_stt_phrase_wake(self) -> None:
        """Wait for an STT-recognised wake phrase such as ``Xin chào``.

        This is intentionally separate from openWakeWord: arbitrary text cannot
        be used as an openWakeWord model name without supplying a trained ONNX
        model. The trade-off is that activation happens after the phrase ends.
        """
        phrase = self.config.wake_word.stt_phrase
        assert phrase is not None
        print(f"\n{self._waiting_message(include_exit_hint=True)}")

        while not self.stopping:
            try:
                samples = self._capture(use_pre_roll=False, prompt_on_no_speech=False)
            except AudioError as exc:
                log.error("Lỗi mic khi chờ câu đánh thức: %s", exc)
                self.say(self.config.replies.mic_error)
                self._restart_mic()
                continue

            if samples is None:
                continue

            self.state.set(State.THINKING, "đang nhận dạng câu đánh thức")
            try:
                transcript = self.stt.transcribe(samples)
            except SttError as exc:
                log.error("Lỗi STT khi chờ câu đánh thức: %s", exc)
                self.state.set(State.ERROR, "lỗi STT")
                continue

            if not self._matches_phrase(transcript, [phrase]):
                log.debug("Chưa nghe câu đánh thức: %r", transcript)
                self.state.set(State.IDLE, f"đang chờ '{phrase}'")
                continue

            log.info("Đã nghe câu đánh thức %r: %r", phrase, transcript)
            self.mic.drain()
            self._run_conversation()

    def run_hands_free(self) -> None:
        """Wait for one configured wake phrase, then converse until the end phrase."""
        self.warmup()
        self._prepare_mic()
        self.say(self.config.replies.startup)
        self.state.set(State.IDLE)

        if self.config.wake_word.stt_phrase is not None:
            self._run_stt_phrase_wake()
            return

        print(f"\n{self._waiting_message(include_exit_hint=True)}")
        consecutive_mic_errors = 0
        while not self.stopping:
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                consecutive_mic_errors += 1
                if consecutive_mic_errors == 5:
                    log.error("Mic không có dữ liệu trong 5 giây liên tiếp")
                    self.state.set(State.ERROR, "mic không phản hồi")
                if consecutive_mic_errors >= 15:
                    self.say(self.config.replies.mic_error)
                    self._restart_mic()
                    consecutive_mic_errors = 0
                continue
            consecutive_mic_errors = 0

            try:
                triggered = self.wake.process(frame)
            except WakeWordError as exc:
                log.error("Wake word lỗi: %s", exc)
                self.state.set(State.ERROR, "lỗi wake word")
                time.sleep(1.0)
                continue

            if not triggered:
                continue

            self.wake.reset()
            self.mic.drain()
            self._run_conversation()

    def _restart_mic(self) -> None:
        log.warning("Khởi động lại micro")
        self.mic.stop()
        time.sleep(0.5)
        try:
            self._prepare_mic()
            self.state.set(State.IDLE)
        except AudioError as exc:
            log.error("Không mở lại được mic: %s", exc)
            self.state.set(State.ERROR, "mic lỗi")
            time.sleep(3.0)

    def run(self, *, hands_free: bool | None = None) -> None:
        """Entry point used by the CLI; picks the loop based on config/flags."""
        use_wake = self.config.wake_word.enabled if hands_free is None else hands_free
        try:
            if use_wake:
                self.run_hands_free()
            else:
                self.run_push_to_talk()
        except KeyboardInterrupt:
            print("\nĐang dừng…")
        finally:
            self.close()
