"""Deterministic intent router that runs *before* the LLM.

Why this exists
---------------
``nvidia/nemotron-3-nano-4b`` is a reasoning model, and measurements against the local
LM Studio server show the cost of that on this hardware:

* one reasoning round takes 5-13 s, and a tool call needs at least two rounds, so
  "mở youtube" through the LLM lands at 10-20 s;
* its reasoning runs in English, and Vietnamese noun phrases get mangled on the way
  back. Observed live: "phát bài Em của ngày hôm qua" produced
  ``play_song(song_name="Em")`` because the model read "của ngày hôm qua" as
  "of yesterday" rather than as part of the title.

Both problems disappear for the handful of commands that are pure pattern matching.
The router answers those from Python in milliseconds, with the user's words preserved
verbatim, and falls through to the LLM for anything genuinely open-ended (chat,
questions, web search summaries).

Matching is done on diacritic-folded *words*, not on the raw string, because STT output
is inconsistent about tone marks. Titles, however, are taken from the original words so
nothing the user said is lost.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .logging_setup import get_logger
from .tools import ToolBox
from .tools.apps import AppToolError, MediaController

log = get_logger("jarvis.intents")


@dataclass(slots=True)
class IntentMatch:
    """A command the router handled without asking the LLM."""

    #: Tool name, mirroring what ``ToolBox.last_used`` would have recorded.
    tool: str
    #: The sentence to speak.
    reply: str


def fold(text: str) -> str:
    """Lowercase and strip Vietnamese diacritics: 'Máy tính' -> 'may tinh'."""
    decomposed = unicodedata.normalize("NFD", (text or "").casefold())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "d")


def _words(text: str) -> list[str]:
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return cleaned.split()


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------
#: Politeness padding that must not end up inside a song title or an app alias.
TRAILING_FILLERS: tuple[tuple[str, ...], ...] = (
    ("giup", "toi"),
    ("giup", "minh"),
    ("giup", "tao"),
    ("gium", "toi"),
    ("gium", "minh"),
    ("ho", "toi"),
    ("ho", "minh"),
    ("cho", "toi"),
    ("cho", "minh"),
    ("duoc", "khong"),
    ("di",),
    ("nhe",),
    ("nha",),
    ("voi",),
    ("luon",),
    ("ngay",),
    ("nao",),
)
LEADING_FILLERS: tuple[tuple[str, ...], ...] = (
    ("jarvis",),
    ("oi",),
    ("hay",),
    ("ban",),
    ("lam", "on"),
)

PLAY_VERBS = ("mo", "phat", "bat", "nghe", "choi", "play")
#: Noun phrases that mark "the rest of this sentence is a song title".
SONG_NOUNS: tuple[tuple[str, ...], ...] = (
    ("bai", "hat"),
    ("ca", "khuc"),
    ("bai",),
    ("nhac",),
)

OPEN_VERBS: tuple[tuple[str, ...], ...] = (
    ("khoi", "dong"),
    ("mo",),
    ("bat",),
    ("vao",),
    ("chay",),
    ("truy", "cap"),
)
#: Optional nouns between the verb and the alias ("mở ứng dụng spotify").
OPEN_NOUNS: tuple[tuple[str, ...], ...] = (
    ("ung", "dung"),
    ("chuong", "trinh"),
    ("trang", "web"),
    ("phan", "mem"),
    ("app",),
    ("trang",),
    ("web",),
)

#: Phrase -> media action. Matched as a whole-word subsequence of the utterance.
MEDIA_PHRASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tang", "am", "luong"), "volume_up"),
    (("tang", "tieng"), "volume_up"),
    (("to", "len"), "volume_up"),
    (("lon", "tieng"), "volume_up"),
    (("giam", "am", "luong"), "volume_down"),
    (("giam", "tieng"), "volume_down"),
    (("nho", "tieng"), "volume_down"),
    (("nho", "lai"), "volume_down"),
    (("tat", "tieng"), "mute"),
    (("bo", "tieng"), "mute"),
    (("im", "lang"), "mute"),
    (("bai", "tiep", "theo"), "next"),
    (("bai", "ke", "tiep"), "next"),
    (("chuyen", "bai"), "next"),
    (("bai", "truoc"), "previous"),
    (("quay", "lai", "bai", "truoc"), "previous"),
    (("tam", "dung"), "play_pause"),
    (("dung", "nhac"), "play_pause"),
    (("dung", "phat"), "play_pause"),
    (("tiep", "tuc", "phat"), "play_pause"),
)

#: Date/time questions. A local model answers these from stale training data and
#: states a confidently wrong date, so they never reach it.
DATETIME_PHRASES: tuple[tuple[str, ...], ...] = (
    ("may", "gio"),
    ("gio", "may"),
    ("ngay", "may"),
    ("thu", "may"),
    ("ngay", "bao", "nhieu"),
    ("hom", "nay", "la", "ngay"),
    ("ngay", "gio", "hien", "tai"),
    ("gio", "hien", "tai"),
    ("thoi", "gian", "hien", "tai"),
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _strip_sequences(
    words: list[str],
    folded: list[str],
    sequences: tuple[tuple[str, ...], ...],
    *,
    from_end: bool,
) -> tuple[list[str], list[str]]:
    """Repeatedly drop any of ``sequences`` from one end of the word list."""
    changed = True
    while changed and words:
        changed = False
        for sequence in sequences:
            size = len(sequence)
            if len(folded) < size:
                continue
            window = tuple(folded[-size:]) if from_end else tuple(folded[:size])
            if window == sequence:
                if from_end:
                    words, folded = words[:-size], folded[:-size]
                else:
                    words, folded = words[size:], folded[size:]
                changed = True
                break
    return words, folded


def _match_prefix(
    folded: list[str], sequences: tuple[tuple[str, ...], ...]
) -> int:
    """Length of the longest sequence in ``sequences`` matching the start, else 0."""
    best = 0
    for sequence in sequences:
        size = len(sequence)
        if size > best and tuple(folded[:size]) == sequence:
            best = size
    return best


def _contains_phrase(folded: list[str], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(tuple(folded[i : i + size]) == phrase for i in range(len(folded) - size + 1))


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------
class IntentRouter:
    """Maps a transcript to a tool call, or to ``None`` when the LLM should handle it."""

    def __init__(self, tools: ToolBox) -> None:
        self._tools = tools

    def route(self, transcript: str) -> IntentMatch | None:
        words = _words(transcript or "")
        if not words:
            return None
        folded = [fold(word) for word in words]
        words, folded = _strip_sequences(words, folded, LEADING_FILLERS, from_end=False)
        if not words:
            return None

        for handler in (
            self._route_song,
            self._route_open_app,
            self._route_media,
            self._route_datetime,
        ):
            match = handler(words, folded)
            if match is not None:
                log.info("Intent %s xử lý trực tiếp (bỏ qua LLM): %r", match.tool, transcript)
                return match
        return None

    # -- song --------------------------------------------------------------------
    def _route_song(self, words: list[str], folded: list[str]) -> IntentMatch | None:
        if not self._tools.music.enabled:
            return None
        if not folded or folded[0] not in PLAY_VERBS:
            return None

        rest_words, rest_folded = words[1:], folded[1:]
        noun_size = _match_prefix(rest_folded, SONG_NOUNS)
        if noun_size == 0:
            return None
        rest_words, rest_folded = rest_words[noun_size:], rest_folded[noun_size:]

        # "của" / "tên là" between the noun and the title.
        rest_words, rest_folded = _strip_sequences(
            rest_words, rest_folded, (("ten", "la"), ("co", "ten", "la")), from_end=False
        )
        rest_words, rest_folded = _strip_sequences(
            rest_words, rest_folded, TRAILING_FILLERS, from_end=True
        )
        title = " ".join(rest_words).strip()
        if not title:
            # "mở nhạc" with no title is an app-launch request, not a song request.
            return None
        return IntentMatch("play_song", self._tools.play_song(song_name=title))

    # -- app ---------------------------------------------------------------------
    def _route_open_app(self, words: list[str], folded: list[str]) -> IntentMatch | None:
        if not self._tools.apps.enabled:
            return None
        verb_size = _match_prefix(folded, OPEN_VERBS)
        if verb_size == 0:
            return None

        rest_words, rest_folded = words[verb_size:], folded[verb_size:]
        noun_size = _match_prefix(rest_folded, OPEN_NOUNS)
        rest_words, rest_folded = rest_words[noun_size:], rest_folded[noun_size:]
        rest_words, rest_folded = _strip_sequences(
            rest_words, rest_folded, TRAILING_FILLERS, from_end=True
        )
        target = " ".join(rest_words).strip()
        if not target:
            return None

        # Only claim the turn when the alias really exists; otherwise the LLM may
        # still have a better idea (a whitelisted folder, a web search, ...).
        try:
            self._tools.apps.resolve(target)
        except AppToolError:
            return None
        return IntentMatch("open_application", self._tools.open_application(app_name=target))

    # -- media -------------------------------------------------------------------
    def _route_media(self, _words: list[str], folded: list[str]) -> IntentMatch | None:
        if not self._tools.media.enabled:
            return None
        for phrase, action in MEDIA_PHRASES:
            if _contains_phrase(folded, phrase):
                if action not in MediaController.ACTIONS:  # pragma: no cover - guard
                    continue
                return IntentMatch("control_media", self._tools.control_media(action=action))
        return None

    # -- date/time ---------------------------------------------------------------
    def _route_datetime(self, _words: list[str], folded: list[str]) -> IntentMatch | None:
        if not self._tools.clock.enabled:
            return None
        if not any(_contains_phrase(folded, phrase) for phrase in DATETIME_PHRASES):
            return None
        return IntentMatch("get_current_datetime", self._tools.get_current_datetime())
