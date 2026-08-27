"""Command line interface: ``python -m jarvis <command>``.

Each mvp.md task has a matching demo command:

===========  ==============================================================
Task         Command
===========  ==============================================================
1            ``--print-config``, ``doctor``, ``devices``
2            ``chat "Xin chào"``
3            ``transcribe sample.wav``
4            ``speak "Xin chào, tôi là Jarvis"``
5            ``listen`` (push-to-talk)
6/7          ``run`` (wake word + tray)  -- also the default command
8            ``shell open_downloads``
9            ``open spotify`` / ``media volume_up``
10           ``search "thời tiết Hà Nội hôm nay"``
11           ``browse "mở youtube và tìm nhạc lofi"``
===========  ==============================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, JarvisConfig, load_config, missing_model_files
from .console import enable_utf8_output
from .logging_setup import get_logger, setup_logging

log = get_logger("jarvis.cli")


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Trợ lý giọng nói tiếng Việt chạy local (wake word -> STT -> LLM -> TTS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Đường dẫn tới config.yaml")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Ghi đè logging.level trong config",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="In config đã parse & validate rồi thoát",
    )

    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="Chạy trợ lý đầy đầy đủ (wake word + tray icon)")
    run_cmd.add_argument("--no-tray", action="store_true", help="Không hiện tray icon")
    run_cmd.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Dùng Enter thay cho wake word",
    )

    sub.add_parser("listen", help="Vòng lặp push-to-talk (nhấn Enter rồi nói)")
    sub.add_parser("devices", help="Liệt kê thiết bị audio")
    sub.add_parser("doctor", help="Kiểm tra model, audio, LM Studio, tools")

    chat_cmd = sub.add_parser("chat", help="Gửi text cho LLM và in câu trả lời")
    chat_cmd.add_argument("text", nargs="+")
    chat_cmd.add_argument("--no-tools", action="store_true", help="Không cho LLM gọi tool")
    chat_cmd.add_argument("--speak", action="store_true", help="Đọc câu trả lời bằng TTS")

    transcribe_cmd = sub.add_parser("transcribe", help="Nhận dạng một file WAV")
    transcribe_cmd.add_argument("wav")

    speak_cmd = sub.add_parser("speak", help="Đọc một câu bằng giọng tiếng Việt")
    speak_cmd.add_argument("text", nargs="+")
    speak_cmd.add_argument("--out", help="Lưu ra file WAV thay vì phát ra loa")

    record_cmd = sub.add_parser("record", help="Ghi âm ra file WAV (để test STT)")
    record_cmd.add_argument("out")
    record_cmd.add_argument("--seconds", type=float, default=5.0)

    shell_cmd = sub.add_parser("shell", help="Chạy một lệnh trong whitelist")
    shell_cmd.add_argument("name", nargs="?", help="Bỏ trống để xem danh sách")
    shell_cmd.add_argument("argument", nargs="?", default="")

    open_cmd = sub.add_parser("open", help="Mở ứng dụng theo alias")
    open_cmd.add_argument("alias", nargs="?", help="Bỏ trống để xem danh sách")

    media_cmd = sub.add_parser("media", help="Điều khiển nhạc / âm lượng")
    media_cmd.add_argument("action", nargs="?", help="play_pause, next, volume_up…")

    search_cmd = sub.add_parser("search", help="Tìm kiếm web (DuckDuckGo/SearxNG)")
    search_cmd.add_argument("query", nargs="+")

    browse_cmd = sub.add_parser("browse", help="Giao tác vụ web cho browser-use")
    browse_cmd.add_argument("task", nargs="+")

    return parser


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load(args: argparse.Namespace) -> JarvisConfig:
    config = load_config(args.config)
    if args.log_level:
        config.logging.level = args.log_level
    setup_logging(config.logging)
    return config


def _dump_config(config: JarvisConfig) -> str:
    import yaml

    data = config.model_dump(mode="json", exclude={"llm": {"system_prompt"}})
    rendered = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)
    prompt_preview = config.llm.system_prompt.strip().splitlines()
    preview = "\n".join(f"    {line}" for line in prompt_preview[:4])
    return (
        f"# config: {config.source_path}\n"
        f"{rendered}\n"
        f"llm.system_prompt: ({len(config.llm.system_prompt)} ký tự, 4 dòng đầu)\n{preview}\n"
    )


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------
def cmd_devices(_config: JarvisConfig) -> int:
    from .audio import describe_devices

    print(describe_devices())
    return 0


def cmd_doctor(config: JarvisConfig) -> int:
    from .audio import AudioError, describe_devices
    from .llm import LLMClient, LlmError
    from .tools import ToolBox

    problems = 0
    print(f"config           : {config.source_path}")
    print(f"python           : {sys.version.split()[0]}")

    missing = missing_model_files(config)
    if missing:
        problems += 1
        print("model files      : THIẾU")
        for path in missing:
            print(f"  - {path}")
        print("  -> chạy: python scripts/download_models.py")
    else:
        print("model files      : OK (STT + TTS)")

    try:
        import openwakeword

        onnx = Path(
            openwakeword.MODELS.get(config.wake_word.model, {})
            .get("model_path", "")
            .replace(".tflite", ".onnx")
        )
        if config.wake_word.model in openwakeword.MODELS and not onnx.is_file():
            problems += 1
            print(f"wake word        : THIẾU {onnx.name} -> python scripts/download_models.py")
        else:
            print(f"wake word        : OK ({config.wake_word.model})")
    except ImportError:
        problems += 1
        print("wake word        : chưa cài openwakeword")

    try:
        lines = describe_devices().splitlines()
        print(f"audio devices    : {len(lines)} thiết bị")
        for line in lines:
            if "default-in" in line or "default-out" in line:
                print(f"  {line}")
    except AudioError as exc:
        problems += 1
        print(f"audio devices    : LỖI {exc}")

    client = LLMClient(config.llm)
    try:
        print(f"LM Studio        : OK {client.ping()}")
    except LlmError as exc:
        problems += 1
        print(f"LM Studio        : LỖI {exc}")
    finally:
        client.close()

    print("tools:")
    for row in ToolBox(config).summary():
        print(f"  {row}")

    print("\n" + ("Có vấn đề cần xử lý ở trên." if problems else "Mọi thứ đã sẵn sàng."))
    return 1 if problems else 0


def cmd_chat(config: JarvisConfig, args: argparse.Namespace) -> int:
    """Text-mode equivalent of one voice turn.

    It goes through the same intent router as the voice loop, so a command answered
    deterministically there behaves identically here.
    """
    from .intents import IntentRouter
    from .llm import LLMClient, LlmError
    from .tools import ToolBox

    prompt = " ".join(args.text)
    toolbox = ToolBox(config)

    if not args.no_tools:
        direct = IntentRouter(toolbox).route(prompt)
        if direct is not None:
            print(direct.reply)
            return _speak_if_requested(config, args, direct.reply)

    client = LLMClient(config.llm)
    try:
        answer = (
            client.chat(prompt) if args.no_tools else client.act(prompt, toolbox.build_tool_defs())
        )
    except LlmError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(answer or "(LLM không trả về nội dung)")
    return _speak_if_requested(config, args, answer)


def _speak_if_requested(config: JarvisConfig, args: argparse.Namespace, answer: str) -> int:
    if not (args.speak and answer):
        return 0

    from .audio import Speaker
    from .tts import TextToSpeech, TtsError

    try:
        TextToSpeech(config.tts, Speaker(config.audio)).speak(answer)
    except TtsError as exc:
        print(f"Không đọc được: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_transcribe(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .audio import AudioError
    from .stt import SpeechToText, SttError

    try:
        text = SpeechToText(config.stt).transcribe_file(args.wav)
    except (SttError, AudioError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    if not text:
        print("(không nhận được nội dung nào)")
        return 1
    print(text)
    return 0


def cmd_speak(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .audio import AudioError, Speaker
    from .tts import TextToSpeech, TtsError

    text = " ".join(args.text)
    tts = TextToSpeech(config.tts, Speaker(config.audio))
    try:
        if args.out:
            path = tts.save(text, args.out)
            print(f"Đã lưu: {path}")
        else:
            result = tts.speak(text)
            print(f"Đã đọc {result.duration:.2f}s audio.")
    except (TtsError, AudioError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_record(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .audio import AudioError, record_fixed, write_wav

    print(f"Ghi âm {args.seconds:.0f} giây, nói ngay bây giờ…")
    try:
        samples = record_fixed(config.audio, args.seconds)
        path = write_wav(args.out, samples, config.audio.sample_rate)
    except AudioError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    print(f"Đã lưu: {path}")
    return 0


def cmd_shell(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .tools import ShellRunner, ShellToolError

    runner = ShellRunner(config.tools.shell)
    if not args.name:
        print("Các lệnh trong whitelist:")
        print(runner.catalogue() or "  (trống)")
        return 0
    try:
        print(runner.run(args.name, args.argument))
    except ShellToolError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_open(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .tools import AppLauncher, AppToolError

    launcher = AppLauncher(config.tools.apps)
    if not args.alias:
        print("Các alias khả dụng:")
        print(launcher.catalogue() or "  (trống)")
        return 0
    try:
        print(launcher.open(args.alias))
    except AppToolError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_media(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .tools import AppToolError
    from .tools.apps import MediaController

    controller = MediaController(config.tools.media)
    if not args.action:
        print("Hành động hợp lệ: " + ", ".join(MediaController.ACTIONS))
        return 0
    try:
        print(controller.perform(args.action))
    except AppToolError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_search(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .tools import WebSearch, WebSearchError

    try:
        print(WebSearch(config.tools.web_search).search_as_text(" ".join(args.query)))
    except WebSearchError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_browse(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .tools import BrowserAgentTool, BrowserToolError

    tool = BrowserAgentTool(config.tools.browser_use, config.llm)
    try:
        print(tool.run(" ".join(args.task)))
    except BrowserToolError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_listen(config: JarvisConfig) -> int:
    from .pipeline import Pipeline

    with Pipeline(config) as pipeline:
        pipeline.run(hands_free=False)
    return 0


def cmd_run(config: JarvisConfig, args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    hands_free = not args.push_to_talk
    pipeline = Pipeline(config)
    use_tray = config.tray.enabled and not args.no_tray
    if not use_tray:
        try:
            pipeline.run(hands_free=hands_free)
        finally:
            pipeline.close()
        return 0

    from .tray import run_with_tray

    return run_with_tray(pipeline, config.tray, hands_free=hands_free)


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    enable_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _load(args)
    except ConfigError as exc:
        print(f"Lỗi config: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(_dump_config(config))
        return 0

    command = args.command or "run"
    try:
        if command == "run":
            if not hasattr(args, "no_tray"):  # default command without the subparser
                args.no_tray = False
                args.push_to_talk = False
            return cmd_run(config, args)
        if command == "listen":
            return cmd_listen(config)
        if command == "devices":
            return cmd_devices(config)
        if command == "doctor":
            return cmd_doctor(config)
        if command == "chat":
            return cmd_chat(config, args)
        if command == "transcribe":
            return cmd_transcribe(config, args)
        if command == "speak":
            return cmd_speak(config, args)
        if command == "record":
            return cmd_record(config, args)
        if command == "shell":
            return cmd_shell(config, args)
        if command == "open":
            return cmd_open(config, args)
        if command == "media":
            return cmd_media(config, args)
        if command == "search":
            return cmd_search(config, args)
        if command == "browse":
            return cmd_browse(config, args)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
        return 130

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
