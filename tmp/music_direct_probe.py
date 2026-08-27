from __future__ import annotations

from jarvis.config import load_config
from jarvis.tools import ToolBox
from jarvis.tools import music as music_module

OPENED: list[str] = []
music_module.webbrowser.open = lambda url, new=0: (OPENED.append(url), True)[1]


def main() -> None:
    config = load_config("config.yaml")
    box = ToolBox(config)
    with open("tmp/music_direct_result.txt", "w", encoding="utf-8") as fh:
        raw = box._play_song(song_name="Em của ngày hôm qua")
        fh.write(f"raw={raw!r}\nopened={OPENED!r}\n")


if __name__ == "__main__":
    main()
