"""Temporary probe: does the model call the existing current_datetime tool?"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

from jarvis.config import load_config
from jarvis.llm import LLMClient
from jarvis.tools import ToolBox


def main() -> None:
    config = load_config("config.yaml")
    box = ToolBox(config)
    client = LLMClient(config.llm)
    try:
        with open("tmp/datetime_result.txt", "w", encoding="utf-8") as fh:
            for prompt in ("mấy giờ rồi", "bây giờ là ngày mấy"):
                box.reset_usage()
                answer = client.act(prompt, box.build_tool_defs())
                fh.write(f"prompt={prompt!r} tools_used={box.last_used} answer={answer!r}\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
