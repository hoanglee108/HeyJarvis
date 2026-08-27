from __future__ import annotations

from jarvis.config import load_config
from jarvis.llm import LLMClient
from jarvis.tools import ToolBox
from jarvis.tools import music as music_module

OPENED: list[str] = []
music_module.webbrowser.open = lambda url, new=0: (OPENED.append(url), True)[1]

CALLS: list[tuple] = []
_orig_play_song = ToolBox._play_song


def _wrapped(self, *args, **kwargs):
    result = _orig_play_song(self, *args, **kwargs)
    CALLS.append((args, kwargs, result))
    return result


ToolBox._play_song = _wrapped

ROUNDS: list[int] = []


def main() -> None:
    config = load_config("config.yaml")
    box = ToolBox(config)
    client = LLMClient(config.llm)
    try:
        box.reset_usage()
        OPENED.clear()
        CALLS.clear()
        ROUNDS.clear()
        answer = client.act(
            "mở bài hát em của ngày hôm qua",
            box.build_tool_defs(),
            on_round=lambda n: ROUNDS.append(n),
        )
        with open("tmp/music_act_result.txt", "w", encoding="utf-8") as fh:
            fh.write(f"answer={answer!r}\n")
            fh.write(f"rounds={ROUNDS!r}\n")
            fh.write(f"calls={CALLS!r}\n")
            fh.write(f"opened={OPENED!r}\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
