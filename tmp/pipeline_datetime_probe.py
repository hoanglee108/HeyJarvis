from __future__ import annotations

from jarvis.config import load_config
from jarvis.pipeline import Pipeline


def main() -> None:
    config = load_config("config.yaml")
    probe = object.__new__(Pipeline)
    probe.config = config
    probe.tools = __import__("jarvis.tools", fromlist=["ToolBox"]).ToolBox(config)

    with open("tmp/pipeline_datetime_result.txt", "w", encoding="utf-8") as fh:
        for text in ("mấy giờ rồi", "hôm nay là ngày mấy", "thứ mấy hôm nay", "thời tiết hôm nay thế nào"):
            answer = probe._try_answer_datetime_directly(text)
            fh.write(f"text={text!r} datetime_shortcut={answer!r}\n")


if __name__ == "__main__":
    main()
