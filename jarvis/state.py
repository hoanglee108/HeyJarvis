"""Assistant state machine shared between the pipeline and the tray icon.

The pipeline only ever calls :meth:`StateMachine.set`; observers (tray icon, logs,
tests) subscribe and react. Transitions are validated so an illegal jump surfaces
as a warning instead of a silently wrong icon.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import Enum

from .logging_setup import get_logger

log = get_logger("jarvis.state")


class State(str, Enum):
    STARTING = "starting"
    IDLE = "idle"           # waiting for the wake word
    LISTENING = "listening"  # recording the user's utterance
    THINKING = "thinking"    # STT + LLM + tools
    SPEAKING = "speaking"    # TTS playback
    ERROR = "error"
    STOPPED = "stopped"


#: Allowed transitions. ERROR and STOPPED are reachable from anywhere.
_ALLOWED: dict[State, set[State]] = {
    State.STARTING: {State.IDLE, State.LISTENING, State.THINKING, State.SPEAKING},
    State.IDLE: {State.LISTENING, State.THINKING},
    State.LISTENING: {State.THINKING, State.SPEAKING, State.IDLE},
    State.THINKING: {State.SPEAKING, State.IDLE, State.LISTENING},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.THINKING},
    State.ERROR: {State.IDLE, State.LISTENING, State.THINKING, State.SPEAKING},
    State.STOPPED: set(),
}

_LABELS: dict[State, str] = {
    State.STARTING: "Đang khởi động…",
    State.IDLE: "Đang chờ lệnh đánh thức",
    State.LISTENING: "Đang nghe…",
    State.THINKING: "Đang xử lý…",
    State.SPEAKING: "Đang nói…",
    State.ERROR: "Lỗi",
    State.STOPPED: "Đã dừng",
}

Observer = Callable[[State, str], None]


def is_valid_transition(current: State, target: State) -> bool:
    if current is target:
        return True
    if target in (State.ERROR, State.STOPPED):
        return True
    return target in _ALLOWED[current]


def label_for(state: State) -> str:
    return _LABELS[state]


class StateMachine:
    """Thread-safe current-state holder with observer notifications."""

    def __init__(self, initial: State = State.STARTING) -> None:
        self._state = initial
        self._detail = ""
        self._lock = threading.RLock()
        self._observers: list[Observer] = []

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    def subscribe(self, observer: Observer) -> None:
        with self._lock:
            self._observers.append(observer)
            state, detail = self._state, self._detail
        self._notify_one(observer, state, detail)

    def set(self, target: State, detail: str = "") -> State:
        """Move to ``target``. Illegal transitions are logged and still applied
        (the UI must never desync from reality), but they are loud in the log."""
        with self._lock:
            current = self._state
            if not is_valid_transition(current, target):
                log.warning("Chuyển trạng thái không hợp lệ: %s -> %s", current.value, target.value)
            self._state = target
            self._detail = detail
            observers = list(self._observers)

        if current is not target:
            log.debug("state %s -> %s %s", current.value, target.value, detail)
        for observer in observers:
            self._notify_one(observer, target, detail)
        return target

    @staticmethod
    def _notify_one(observer: Observer, state: State, detail: str) -> None:
        try:
            observer(state, detail)
        except Exception:  # pragma: no cover - an observer must never break the pipeline
            log.exception("State observer lỗi")
