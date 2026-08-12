"""End-to-end orchestration: wake word -> STT -> LLM (+tools) -> TTS.

This is the module that mvp.md Task 5 wires up and Tasks 6 and 12 harden. Every
stage is wrapped so a single failure (LM Studio down, mic unplugged, a tool raising)
degrades into a spoken apology instead of killing the process.
"""

from __future__ import annotations

import threading
import time
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
from .llm import LLMClient, LlmError, LlmUnavailableError
from .logging_setup import get_logger
from .state import State, StateMachine
from .stt import SpeechToText, SttError
from .tools import ToolBox
from .tts import TextToSpeech, TtsError
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
        self.wake = WakeWordDetector(config.wake_word)
        self.mic = MicStream(config.audio)

        self._stop_event = threading.Event()
        self._speech_threshold = config.audio.silence_rms_threshold or 0.01
        self._tool_defs: list[object] | None = None

    # ----------------------------------------------------------------------------
    # lifecycle
    # ----------------------------------------------------------------------------
    def warmup(self, *, include_llm: bool = True, include_wake: bool = True) -> None:
        """Load models up front so the first reply is not slowed by cold start."""
        self.state.set(State.STARTING, "đang nạp model")
        self.stt.load()
        self.tts.load()
        if include_wake and self.config.wake_word.enabled:
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
    def say(self, text: str) -> bool:
        """Speak ``text``; returns False when synthesis/playback failed."""
        if not text.strip():
            return False
        previous = self.state.state
        previous_detail = self.state.detail
        self.state.set(State.SPEAKING, text[:60])
        try:
            self.tts.speak(text)
            return True
        except (TtsError, AudioError) as exc:
            log.error("Không nói được: %s", exc)
            print(f"[Jarvis] {text}")  # last-resort feedback channel
            return False
        finally:
            # Keep an ERROR indication visible; otherwise go back to waiting.
            if previous is State.ERROR:
                self.state.set(State.ERROR, previous_detail)
            elif previous is not State.SPEAKING:
                self.state.set(State.IDLE)
            # Discard the audio the microphone picked up from our own voice.
            if self.mic.running:
                self.mic.drain()
                if self.wake.loaded:
                    self.wake.reset()

    # ----------------------------------------------------------------------------
    # one turn
    # ----------------------------------------------------------------------------
    def respond_to_text(self, text: str) -> TurnResult:
        """LLM (with tools) -> spoken reply. Used by both voice and ``jarvis chat``."""
        result = TurnResult(transcript=text)
        self.state.set(State.THINKING, text[:60])
        self.tools.reset_usage()

        if self._tool_defs is None:
            self._tool_defs = self.tools.build_tool_defs()

        try:
            reply = self.llm.act(text, self._tool_defs)
        except LlmUnavailableError as exc:
            log.error("LM Studio không khả dụng: %s", exc)
            result.error = str(exc)
            self.state.set(State.ERROR, "LM Studio offline")
            result.spoken = self.say(self.config.replies.llm_unavailable)
            return result
        except LlmError as exc:
            log.error("Lỗi LLM: %s", exc)
            result.error = str(exc)
            self.state.set(State.ERROR, "lỗi LLM")
            result.spoken = self.say(self.config.replies.generic_error)
            return result

        result.tools_used = list(self.tools.last_used)
        if not reply:
            result.error = "LLM trả về nội dung rỗng"
            result.spoken = self.say(self.config.replies.generic_error)
            return result

        result.reply = reply
        result.spoken = self.say(reply)
        self.state.set(State.IDLE)
        return result

    def process_audio(self, samples: np.ndarray, sample_rate: int | None = None) -> TurnResult:
        """Transcribe an utterance and answer it."""
        self.state.set(State.THINKING, "đang nhận dạng")
        try:
            transcript = self.stt.transcribe(samples, sample_rate)
        except SttError as exc:
            log.error("Lỗi STT: %s", exc)
            self.state.set(State.ERROR, "lỗi STT")
            spoken = self.say(self.config.replies.generic_error)
            return TurnResult(error=str(exc), spoken=spoken)

        if not transcript or len(transcript) < 2:
            log.info("Không nhận được nội dung nào")
            spoken = self.say(self.config.replies.not_understood)
            self.state.set(State.IDLE)
            return TurnResult(error="transcript rỗng", spoken=spoken)

        print(f"Bạn: {transcript}")
        result = self.respond_to_text(transcript)
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
    def _capture(self, *, use_pre_roll: bool) -> np.ndarray | None:
        """Record one utterance; ``None`` when nobody spoke."""
        self.state.set(State.LISTENING)
        if self.config.audio.beep_on_listen:
            self.speaker.beep()
            self.mic.drain()

        pre_roll = self.mic.recent(self.config.audio.pre_roll_ms) if use_pre_roll else None
        recording = record_utterance(
            self.mic,
            self.config.audio,
            speech_threshold=self._speech_threshold,
            pre_roll=pre_roll,
        )
        if recording.reason == "no_speech":
            log.info("Không có ai nói, quay lại chờ")
            self.say(self.config.replies.not_understood)
            self.state.set(State.IDLE)
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
                samples = self._capture(use_pre_roll=False)
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

    def run_hands_free(self) -> None:
        """Task 6 loop: always listening for 'Hey Jarvis'."""
        self.warmup()
        self._prepare_mic()
        self.say(self.config.replies.startup)
        self.state.set(State.IDLE)
        print("\nĐang chờ 'Hey Jarvis'… (Ctrl+C để thoát)")

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

            try:
                samples = self._capture(use_pre_roll=True)
            except AudioError as exc:
                log.error("Lỗi mic khi ghi âm: %s", exc)
                self.say(self.config.replies.mic_error)
                self._restart_mic()
                continue

            if samples is not None:
                self.process_audio(samples)

            self.state.set(State.IDLE)
            self.wake.reset()
            self.mic.drain()
            print("Đang chờ 'Hey Jarvis'…")

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
