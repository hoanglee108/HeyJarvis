"""Speak the answer while the model is still writing it (priority.md P1-1).

The latency problem
-------------------
Every stage used to be batch and strictly sequential: record the whole utterance,
transcribe it, wait 5-13 s for the complete LLM reply, synthesise *all* of it, then play
the first sample. The user hears nothing for the sum of all four.

Nothing here makes the model faster. It only removes the waiting: tokens are consumed as
they arrive, cut into sentences, and each sentence is synthesised and played while the
model is still generating the next one. Perceived latency collapses to "time until the
first sentence is complete" while total speaking time is unchanged.

Two details that matter more than they look
-------------------------------------------
* **Reasoning must never be spoken.** LM Studio returns Nemotron's ``<think>`` block in a
  separate ``reasoning_content`` field, so ``delta.content`` is normally clean - but a
  differently configured server, or a block truncated by ``max_tokens``, can still put
  markup in the content stream. :class:`SentenceStream` therefore refuses to emit
  anything while a ``<think>``/``<tool_call>`` tag is open, and scrubs closed ones. It
  errs towards holding text back, because speaking English chain-of-thought out loud is
  much worse than a slightly later first sentence.
* **Sentence boundaries are not just "a dot".** ``3.14`` and ``15.30`` must not be split,
  so a period only ends a sentence when whitespace (or the end of the buffer) follows.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Callable, Protocol

from .llm import strip_reasoning
from .logging_setup import get_logger
from .tts import TextToSpeech, clean_for_speech

log = get_logger("jarvis.streaming")

#: End of a spoken sentence: terminal punctuation followed by whitespace or end of text.
#: The lookahead is what keeps decimals ("3.14") and times ("15.30") in one piece.
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")

#: Markup whose *contents* must never reach the speaker.
_MARKUP_OPEN = re.compile(r"<(think|tool_call|tools|function|parameter)\b", re.IGNORECASE)
#: A tag that is still being typed, e.g. the buffer ends with ``"<thi"``.
_PARTIAL_TAG = re.compile(r"<[a-zA-Z_/]*$")

#: Shorter fragments are merged into the next sentence instead of being spoken alone;
#: one-word utterances make the delivery choppy and cost a full synthesis round-trip.
DEFAULT_MIN_CHARS = 12


class TextSink(Protocol):
    """What :meth:`jarvis.llm.LLMClient.act` needs from a streaming consumer."""

    def push(self, chunk: str) -> None: ...

    def discard_pending(self) -> None: ...


def _has_unclosed_markup(text: str) -> bool:
    for match in _MARKUP_OPEN.finditer(text):
        name = match.group(1)
        if not re.search(rf"</{name}\s*>", text[match.end() :], re.IGNORECASE):
            return True
    return False


class SentenceStream:
    """Turns a token stream into complete, speakable sentences.

    Stateful and not thread-safe by itself: one instance belongs to one response.
    """

    def __init__(self, min_chars: int = DEFAULT_MIN_CHARS) -> None:
        self._min_chars = max(1, min_chars)
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        """Add streamed text; return whichever sentences are now complete."""
        if not chunk:
            return []
        self._pending += chunk
        if _has_unclosed_markup(self._pending) or _PARTIAL_TAG.search(self._pending):
            # Hold everything: we cannot yet tell reasoning from answer.
            return []

        # ``strip_reasoning`` collapses whitespace, which also *removes* a trailing
        # space. Tokens arrive split at arbitrary points ("Trời hôm " + "nay khá"), so
        # losing that space silently glues two words together.
        trailing = " " if self._pending[-1:].isspace() else ""
        cleaned = strip_reasoning(self._pending)
        if not cleaned:
            # The buffer was nothing but markup; drop it so it is not re-scanned.
            self._pending = ""
            return []

        ready: list[str] = []
        cursor = 0
        candidate_start = 0
        for match in _SENTENCE_END.finditer(cleaned):
            end = match.end()
            candidate = cleaned[candidate_start:end].strip()
            if len(candidate) < self._min_chars:
                # Too short to speak on its own; let it grow into the next sentence.
                continue
            ready.append(candidate)
            candidate_start = end
            cursor = end

        self._pending = (cleaned[cursor:] if cursor else cleaned) + trailing
        return [sentence for sentence in ready if clean_for_speech(sentence)]

    def flush(self) -> list[str]:
        """Whatever is left, spoken even if it never got terminal punctuation."""
        remainder = strip_reasoning(self._pending).strip()
        self._pending = ""
        if not remainder or not clean_for_speech(remainder):
            return []
        return [remainder]

    def discard(self) -> None:
        self._pending = ""


class StreamingSpeaker:
    """Synthesises and plays sentences in order on a worker thread.

    Satisfies :class:`TextSink`, so it can be handed straight to
    :meth:`jarvis.llm.LLMClient.act`.
    """

    def __init__(
        self,
        tts: TextToSpeech,
        *,
        on_first_audio: Callable[[], None] | None = None,
        min_chars: int = DEFAULT_MIN_CHARS,
    ) -> None:
        self._tts = tts
        self._on_first_audio = on_first_audio
        self._stream = SentenceStream(min_chars)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._aborted = threading.Event()
        self.spoken_any = False
        self.closed = False
        self.error: Exception | None = None
        self.sentences: list[str] = []

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, name="jarvis-speak", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            sentence = self._queue.get()
            if sentence is None:
                return
            if self._aborted.is_set():
                continue  # keep draining so finish() does not block
            try:
                if not self.spoken_any and self._on_first_audio is not None:
                    self._on_first_audio()
                self._tts.speak(sentence)
                self.spoken_any = True
            except Exception as exc:  # noqa: BLE001 - a failed sentence must not kill the turn
                log.error("Không đọc được câu %r: %s", sentence[:40], exc)
                if self.error is None:
                    self.error = exc
                self._aborted.set()

    # -- TextSink ----------------------------------------------------------------
    def push(self, chunk: str) -> None:
        """Feed streamed model text; complete sentences are queued for playback."""
        if self._aborted.is_set():
            return
        with self._lock:
            sentences = self._stream.feed(chunk)
        for sentence in sentences:
            self.sentences.append(sentence)
            self._queue.put(sentence)

    def discard_pending(self) -> None:
        """Drop text that has not been spoken yet.

        Called when a round turns out to be a tool call: whatever the model had started
        writing belongs to that call, not to the answer.
        """
        with self._lock:
            self._stream.discard()

    # -- completion --------------------------------------------------------------
    def finish(self, timeout: float | None = None) -> bool:
        """Flush the tail, wait for playback, and report whether we spoke cleanly."""
        if self.closed:
            return self.spoken_any and self.error is None
        self.closed = True
        with self._lock:
            remainder = self._stream.flush()
        for sentence in remainder:
            self.sentences.append(sentence)
            self._queue.put(sentence)
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout)
            if self._worker.is_alive():  # pragma: no cover - playback wedged
                log.warning("Luồng phát giọng nói chưa kết thúc sau %.1fs", timeout or 0.0)
                return False
        return self.spoken_any and self.error is None

    def abort(self) -> None:
        """Stop speaking as soon as the current sentence ends."""
        self._aborted.set()
        if self.closed:
            return
        self.closed = True
        with self._lock:
            self._stream.discard()
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    def ensure_closed(self) -> None:
        """Never leave the worker thread parked on an empty queue.

        Any escape route out of the turn - including an exception nobody planned for -
        has to come through here, otherwise every failed turn leaks a thread.
        """
        if not self.closed:
            self.abort()

    @property
    def text(self) -> str:
        return " ".join(self.sentences)
