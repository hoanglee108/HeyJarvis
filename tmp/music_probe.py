"""Temporary probe: song resolution and host allow-listing (no browser opened)."""

from __future__ import annotations

from jarvis.config import load_config
from jarvis.tools.music import MusicPlayer
from jarvis.tools.websearch import WebSearch

BLOCKED = [
    "https://evil.example.com/watch?v=abc",
    "https://youtube.com.attacker.net/watch?v=abc",
    "file:///C:/Windows/System32/calc.exe",
    "https://www.youtube.com/results?search_query=abc",
    "",
]
ALLOWED = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
]


def main() -> None:
    config = load_config("config.yaml")
    player = MusicPlayer(config.tools.music, WebSearch(config.tools.web_search))

    for url in BLOCKED:
        assert not player._is_allowed_video(url), f"phải bị chặn: {url!r}"
    for url in ALLOWED:
        assert player._is_allowed_video(url), f"phải được phép: {url!r}"
    print("HOST_ALLOWLIST_OK")

    song = "Em của ngày hôm qua"
    print(f"resolved={player._resolve_video_url(song)!r}")


if __name__ == "__main__":
    main()
