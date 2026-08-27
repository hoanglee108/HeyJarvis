"""Temporary probe: does the model call play_song / open_application reliably?

webbrowser.open and os.startfile are stubbed so nothing is actually launched.
"""

from __future__ import annotations

import os

from jarvis.config import load_config
from jarvis.llm import LLMClient
from jarvis.tools import ToolBox
from jarvis.tools import apps as apps_module
from jarvis.tools import music as music_module

OPENED: list[str] = []

music_module.webbrowser.open = lambda url, new=0: (OPENED.append(f"browser:{url}"), True)[1]
apps_module.webbrowser.open = lambda url, new=0: (OPENED.append(f"browser:{url}"), True)[1]
os.startfile = lambda target: OPENED.append(f"startfile:{target}")  # type: ignore[attr-defined]
apps_module.os.startfile = os.startfile


def main() -> None:
    config = load_config("config.yaml")
    box = ToolBox(config)
    client = LLMClient(config.llm)
    try:
        with open("tmp/act_result.txt", "w", encoding="utf-8") as fh:
            for prompt in (
                "mở bài hát em của ngày hôm qua",
                "mở trình duyệt cho tôi",
                "mở youtube giúp tôi",
            ):
                box.reset_usage()
                OPENED.clear()
                answer = client.act(prompt, box.build_tool_defs())
                fh.write(
                    f"prompt={prompt!r} tools_used={box.last_used} "
                    f"opened={list(OPENED)} answer={answer!r}\n"
                )
    finally:
        client.close()


if __name__ == "__main__":
    main()
