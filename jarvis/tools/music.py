"""Play a song by name.

``control_media`` can only press media keys and ``open_application`` can only start
Spotify, so "mở bài hát <tên>" had no tool that could satisfy it. Without this the
model tends to invent a plausible-sounding title and claim it started playing.

Safety: when the video is resolved through web search, only URLs whose host is in
``MusicToolConfig.allowed_hosts`` are ever opened, so an arbitrary search result
cannot be launched.
"""

from __future__ import annotations

import os
import urllib.parse
import webbrowser

from ..config import MusicToolConfig
from ..logging_setup import get_logger
from .websearch import WebSearch, WebSearchError

log = get_logger("jarvis.tools.music")

YOUTUBE_SEARCH_URL = "https://www.youtube.com/results?search_query={query}"
_WATCH_PATHS = ("/watch",)


class MusicToolError(RuntimeError):
    """The song could not be resolved or opened."""


class MusicPlayer:
    """Resolves a song name to a playable target and opens it."""

    def __init__(self, config: MusicToolConfig, search: WebSearch) -> None:
        self._config = config
        self._search = search

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def provider(self) -> str:
        return self._config.provider

    # -- URL helpers -------------------------------------------------------------
    def _is_allowed_video(self, url: str) -> bool:
        """True only for a video URL on an allowed host."""
        if not url:
            return False
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.netloc.split("@")[-1].split(":")[0].lower()
        if host not in {allowed.lower() for allowed in self._config.allowed_hosts}:
            return False
        if host == "youtu.be":
            return len(parsed.path.strip("/")) > 0
        if parsed.path not in _WATCH_PATHS:
            return False
        return "v" in urllib.parse.parse_qs(parsed.query)

    def _resolve_video_url(self, song: str) -> str | None:
        """Find the first allowed video URL for ``song`` via web search."""
        if not self._search.enabled:
            log.info("search_web đang tắt nên không thể tìm video cho %r", song)
            return None
        try:
            results = self._search.search(f"{song} youtube")
        except WebSearchError as exc:
            log.warning("Không tìm được video cho %r: %s", song, exc)
            return None

        for result in results:
            if self._is_allowed_video(result.url):
                log.info("Chọn video %r cho %r", result.url, song)
                return result.url
        log.info("Không có kết quả YouTube hợp lệ cho %r", song)
        return None

    # -- public API --------------------------------------------------------------
    def play(self, song: str) -> str:
        song = (song or "").strip()
        if not song:
            raise MusicToolError("Chưa có tên bài hát để phát.")
        if not self._config.enabled:
            raise MusicToolError("Công cụ phát nhạc đang bị tắt trong config.")

        if self._config.provider == "spotify":
            return self._open_spotify(song)

        if self._config.provider == "youtube":
            url = self._resolve_video_url(song)
            if url:
                self._open_url(url)
                return f"Đang phát {song} trên YouTube."

        search_url = YOUTUBE_SEARCH_URL.format(query=urllib.parse.quote_plus(song))
        self._open_url(search_url)
        return (
            f"Tôi chưa tìm được đúng video cho {song}, nên đã mở trang kết quả "
            "tìm kiếm YouTube để bạn chọn."
        )

    def _open_url(self, url: str) -> None:
        try:
            if not webbrowser.open(url, new=2):
                raise MusicToolError(f"Không mở được {url}.")
        except MusicToolError:
            raise
        except OSError as exc:
            raise MusicToolError(f"Không mở được trình duyệt: {exc}") from exc

    def _open_spotify(self, song: str) -> str:
        target = f"spotify:search:{urllib.parse.quote(song)}"
        try:
            os.startfile(target)  # noqa: S606 - fixed protocol handler, quoted query
        except OSError as exc:
            raise MusicToolError(
                f"Không mở được Spotify: {exc}. Kiểm tra Spotify đã được cài chưa."
            ) from exc
        return f"Đã mở Spotify và tìm {song}. Bạn bấm phát bài mong muốn nhé."
