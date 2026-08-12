"""Typed, validated configuration for Jarvis.

Everything that a user might want to tweak (model paths, LM Studio endpoint,
whitelisted shell commands, app aliases, audio thresholds) lives in ``config.yaml``
and is validated here with pydantic before any runtime module touches it.

Relative paths inside the config are resolved against the *config file's* directory
so the project stays portable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_CONFIG_FILENAME = "config.yaml"
LOCAL_CONFIG_FILENAME = "config.local.yaml"


class ConfigError(RuntimeError):
    """Raised when the config file is missing, malformed or fails validation."""


class _Base(BaseModel):
    # extra="forbid" turns silent typos ("treshold") into loud errors.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------------------
# Speech to text
# --------------------------------------------------------------------------------------
class SttConfig(_Base):
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path
    provider: Literal["cpu", "cuda", "directml"] = "cpu"
    num_threads: Annotated[int, Field(ge=1, le=32)] = 4
    decoding_method: Literal["greedy_search", "modified_beam_search"] = "greedy_search"
    max_active_paths: Annotated[int, Field(ge=1, le=32)] = 4
    sample_rate: Annotated[int, Field(ge=8000, le=48000)] = 16000
    feature_dim: Annotated[int, Field(ge=1)] = 80
    # The Zipformer checkpoint emits UPPERCASE Vietnamese; normalise for the LLM.
    lowercase_output: bool = True
    debug: bool = False


# --------------------------------------------------------------------------------------
# Text to speech
# --------------------------------------------------------------------------------------
class TtsConfig(_Base):
    model: Path
    tokens: Path
    data_dir: Path | None = None
    lexicon: Path | None = None
    dict_dir: Path | None = None
    provider: Literal["cpu", "cuda", "directml"] = "cpu"
    num_threads: Annotated[int, Field(ge=1, le=32)] = 2
    speaker_id: Annotated[int, Field(ge=0)] = 0
    speed: Annotated[float, Field(gt=0.1, le=3.0)] = 1.0
    max_num_sentences: Annotated[int, Field(ge=1)] = 1
    debug: bool = False


# --------------------------------------------------------------------------------------
# Wake word
# --------------------------------------------------------------------------------------
class WakeWordConfig(_Base):
    enabled: bool = True
    #: openWakeWord pretrained name (``hey_jarvis``) or a path to a custom ``.onnx`` model.
    #: Used only when ``stt_phrase`` is unset.
    model: str = "hey_jarvis"
    #: Recognise this wake phrase with the local STT instead of an openWakeWord model.
    #: This supports phrases such as Vietnamese ``Xin chào`` without a custom ONNX model.
    stt_phrase: str | None = None
    threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    inference_framework: Literal["onnx", "tflite"] = "onnx"
    #: Ignore further detections for this long after a trigger (debounce).
    refractory_seconds: Annotated[float, Field(ge=0.0)] = 2.0
    #: Phrases that return a hands-free conversation to wake-word standby.
    conversation_end_phrases: list[str] = Field(default_factory=lambda: ["goodbye", "good bye"])
    #: openWakeWord expects 80 ms frames of 16 kHz int16 audio.
    frame_samples: Literal[1280] = 1280
    enable_speex_noise_suppression: bool = False
    vad_threshold: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0


# --------------------------------------------------------------------------------------
# Audio capture / playback
# --------------------------------------------------------------------------------------
class AudioConfig(_Base):
    #: ``None`` = system default. Can be an index (int) or a substring of the device name.
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: Annotated[int, Field(ge=8000, le=48000)] = 16000
    #: Hard cap on one utterance.
    max_record_seconds: Annotated[float, Field(gt=0.5, le=120)] = 15.0
    #: Never cut before this much speech has been captured.
    min_record_seconds: Annotated[float, Field(ge=0.0, le=10)] = 0.5
    #: Stop recording after this much continuous silence.
    silence_timeout_ms: Annotated[int, Field(ge=100, le=10_000)] = 900
    #: Give the speaker this long to start talking before giving up.
    start_timeout_seconds: Annotated[float, Field(gt=0.2, le=30)] = 6.0
    #: RMS (0..1 float scale) above which a frame counts as speech.
    #: ``None`` -> auto-calibrated from ambient noise at startup.
    silence_rms_threshold: float | None = None
    #: Ambient-noise calibration window used when ``silence_rms_threshold`` is None.
    calibration_seconds: Annotated[float, Field(ge=0.1, le=10)] = 1.0
    #: speech_threshold = max(noise_floor * multiplier, floor_minimum)
    calibration_multiplier: Annotated[float, Field(ge=1.0, le=20)] = 3.0
    calibration_floor_minimum: Annotated[float, Field(ge=0.0, le=1.0)] = 0.006
    #: Prepend this much pre-trigger audio so the first syllable is not clipped.
    pre_roll_ms: Annotated[int, Field(ge=0, le=2000)] = 300
    #: Play a short beep when Jarvis starts listening.
    beep_on_listen: bool = True


# --------------------------------------------------------------------------------------
# LLM (LM Studio)
# --------------------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """\
Bạn là Jarvis, trợ lý giọng nói cá nhân chạy hoàn toàn trên máy tính Windows của người dùng.

Nguyên tắc trả lời:
- Luôn trả lời bằng tiếng Việt, giọng văn tự nhiên, thân thiện, ngắn gọn.
- Câu trả lời sẽ được đọc thành tiếng, nên KHÔNG dùng markdown, bullet, emoji,
  bảng, code block hay ký tự đặc biệt. Chỉ viết văn xuôi thuần.
- Trả lời tối đa 2-3 câu, trừ khi người dùng yêu cầu chi tiết.
- Văn bản người dùng đến từ nhận dạng giọng nói nên có thể sai chính tả;
  hãy suy luận ý định hợp lý nhất thay vì hỏi lại nhiều lần.

Cách dùng công cụ:
- Khi cần thao tác máy, mở ứng dụng, điều khiển nhạc hay tra cứu thông tin mới,
  hãy gọi công cụ tương ứng thay vì phỏng đoán.
- Chỉ gọi công cụ khi thật sự cần. Sau khi công cụ trả kết quả, hãy tóm tắt lại
  bằng một câu nói tự nhiên cho người dùng nghe.
- Nếu công cụ báo lỗi, hãy nói rõ ngắn gọn là không thực hiện được và vì sao.
"""


class LlmConfig(_Base):
    #: LM Studio server address, e.g. ``localhost:1234`` (no scheme).
    api_host: str = "localhost:1234"
    #: Model key as shown by ``lms ls`` / LM Studio UI.
    model: str = "qwen2.5-3b-instruct"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.4
    max_tokens: Annotated[int, Field(ge=16, le=8192)] = 512
    context_length: Annotated[int, Field(ge=512)] = 4096
    #: Safety valve for the tool-calling loop.
    max_tool_rounds: Annotated[int, Field(ge=1, le=20)] = 6
    request_timeout_s: Annotated[float, Field(gt=1)] = 180.0
    #: Keep the last N turns of conversation in memory (user+assistant pairs).
    history_turns: Annotated[int, Field(ge=0, le=50)] = 6

    @field_validator("api_host")
    @classmethod
    def _strip_scheme(cls, value: str) -> str:
        value = value.strip()
        for prefix in ("http://", "https://"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
        return value.rstrip("/")


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------
class ShellCommandSpec(_Base):
    """A single pre-approved command.

    The LLM can only pick a command *by name*; it never gets to compose a command
    line. An optional free-text argument is allowed only when ``allow_argument`` is
    true and it must match ``argument_pattern``.
    """

    name: str
    description: str
    executable: str
    args: list[str] = Field(default_factory=list)
    allow_argument: bool = False
    argument_pattern: str = r"^[\w\s\-.,:\\/()']{1,120}$"
    #: Where the argument is substituted, e.g. ``["-Path", "{arg}"]``.
    argument_args: list[str] = Field(default_factory=list)
    use_powershell: bool = False
    timeout_s: Annotated[float, Field(gt=0, le=300)] = 20.0

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]{2,48}", value):
            raise ValueError(
                "shell command name must be snake_case ([a-z0-9_], 2-48 chars)"
            )
        return value

    @field_validator("argument_pattern")
    @classmethod
    def _valid_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid argument_pattern regex: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _no_powershell_interpolation(self) -> ShellCommandSpec:
        # Splicing LLM-provided text into a PowerShell -Command string would be a
        # command-injection hole; arguments are only allowed for direct exec calls.
        if self.use_powershell and self.allow_argument:
            raise ValueError(
                f"shell command {self.name!r}: use_powershell không được dùng cùng "
                "allow_argument (nguy cơ command injection)"
            )
        return self


class ShellToolConfig(_Base):
    enabled: bool = True
    whitelist: list[ShellCommandSpec] = Field(default_factory=list)

    @field_validator("whitelist")
    @classmethod
    def _unique_names(cls, value: list[ShellCommandSpec]) -> list[ShellCommandSpec]:
        names = [spec.name for spec in value]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate shell command names: {sorted(duplicates)}")
        return value


class AppAliasSpec(_Base):
    #: ``exe`` -> executable/path launched directly.
    #: ``uri`` -> protocol handler (``spotify:``).
    #: ``url`` -> opened in the default browser.
    #: ``path`` -> file/folder opened with the shell association.
    kind: Literal["exe", "uri", "url", "path"] = "exe"
    target: str
    args: list[str] = Field(default_factory=list)
    description: str = ""


class AppsToolConfig(_Base):
    enabled: bool = True
    aliases: dict[str, AppAliasSpec] = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def _normalise_keys(cls, value: dict[str, AppAliasSpec]) -> dict[str, AppAliasSpec]:
        return {key.strip().lower(): spec for key, spec in value.items()}


class MediaToolConfig(_Base):
    enabled: bool = True
    #: Number of volume steps per "tăng/giảm âm lượng" request.
    volume_step: Annotated[int, Field(ge=1, le=20)] = 4


class WebSearchToolConfig(_Base):
    enabled: bool = True
    backend: Literal["duckduckgo", "searxng"] = "duckduckgo"
    searxng_url: str | None = None
    max_results: Annotated[int, Field(ge=1, le=15)] = 5
    timeout_s: Annotated[float, Field(gt=1, le=120)] = 15.0
    region: str = "vn-vi"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )


class BrowserUseToolConfig(_Base):
    #: Off by default: browser-use is a heavy optional extra (see README).
    enabled: bool = False
    #: Model used to drive browser-use; defaults to the main LM Studio model.
    model: str | None = None
    headless: bool = False
    max_steps: Annotated[int, Field(ge=1, le=60)] = 15
    timeout_s: Annotated[float, Field(gt=5, le=900)] = 180.0
    use_vision: bool = False


class ToolsConfig(_Base):
    shell: ShellToolConfig = Field(default_factory=ShellToolConfig)
    apps: AppsToolConfig = Field(default_factory=AppsToolConfig)
    media: MediaToolConfig = Field(default_factory=MediaToolConfig)
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    browser_use: BrowserUseToolConfig = Field(default_factory=BrowserUseToolConfig)


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------
class TrayConfig(_Base):
    enabled: bool = True
    title: str = "Jarvis"


class LoggingConfig(_Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = Path("logs/jarvis.log")
    max_bytes: Annotated[int, Field(ge=1024)] = 2_000_000
    backup_count: Annotated[int, Field(ge=0, le=20)] = 3


class SpeechRepliesConfig(_Base):
    """Canned spoken lines used for fallbacks (Task 12)."""

    startup: str = "Jarvis đã sẵn sàng."
    not_understood: str = "Xin lỗi, tôi không nghe rõ. Bạn nói lại giúp tôi nhé."
    llm_unavailable: str = (
        "Xin lỗi, tôi không kết nối được tới LM Studio. "
        "Bạn kiểm tra xem server đã bật chưa nhé."
    )
    generic_error: str = "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu này."
    mic_error: str = "Xin lỗi, tôi không truy cập được micro."


class JarvisConfig(_Base):
    stt: SttConfig
    tts: TtsConfig
    llm: LlmConfig = Field(default_factory=LlmConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    tray: TrayConfig = Field(default_factory=TrayConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    replies: SpeechRepliesConfig = Field(default_factory=SpeechRepliesConfig)

    #: Populated by :func:`load_config`; not part of the YAML schema.
    root_dir: Path = Field(default_factory=Path.cwd, exclude=False)
    source_path: Path | None = None


# --------------------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------------------
_PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("stt", "encoder"),
    ("stt", "decoder"),
    ("stt", "joiner"),
    ("stt", "tokens"),
    ("tts", "model"),
    ("tts", "tokens"),
    ("tts", "data_dir"),
    ("tts", "lexicon"),
    ("tts", "dict_dir"),
    ("logging", "file"),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def find_config_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``config.yaml``: explicit arg > env var > project root > cwd."""
    if explicit is not None:
        return Path(_expand(str(explicit))).resolve()

    env_value = os.environ.get("JARVIS_CONFIG")
    if env_value:
        return Path(_expand(env_value)).resolve()

    project_root = Path(__file__).resolve().parent.parent
    for candidate in (project_root / DEFAULT_CONFIG_FILENAME, Path.cwd() / DEFAULT_CONFIG_FILENAME):
        if candidate.is_file():
            return candidate.resolve()
    return (project_root / DEFAULT_CONFIG_FILENAME).resolve()


def load_config(path: str | os.PathLike[str] | None = None) -> JarvisConfig:
    """Read, merge and validate the config file.

    A sibling ``config.local.yaml`` (git-ignored) is deep-merged on top when present,
    so machine-specific tweaks never touch the committed defaults.
    """
    config_path = find_config_path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Không tìm thấy file config: {config_path}\n"
            "Hãy chạy từ thư mục project hoặc truyền --config <path>."
        )

    raw = _read_yaml_mapping(config_path)

    local_path = config_path.with_name(LOCAL_CONFIG_FILENAME)
    if local_path.is_file():
        raw = _deep_merge(raw, _read_yaml_mapping(local_path))

    root_dir = config_path.parent
    raw["root_dir"] = str(root_dir)
    raw["source_path"] = str(config_path)

    try:
        config = JarvisConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Config không hợp lệ ({config_path}):\n{exc}") from exc

    _resolve_paths(config, root_dir)
    return config


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Lỗi cú pháp YAML trong {path}:\n{exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config phải là một mapping YAML, nhận được {type(data).__name__}: {path}")
    return data


def _resolve_paths(config: JarvisConfig, root_dir: Path) -> None:
    """Expand env vars / ``~`` and make every path absolute relative to the project."""
    for section_name, field_name in _PATH_FIELDS:
        section = getattr(config, section_name)
        value = getattr(section, field_name)
        if value is None:
            continue
        expanded = Path(_expand(str(value)))
        if not expanded.is_absolute():
            expanded = (root_dir / expanded).resolve()
        setattr(section, field_name, expanded)


def missing_model_files(config: JarvisConfig) -> list[Path]:
    """Return the model artefacts referenced by the config that do not exist yet."""
    required = [
        config.stt.encoder,
        config.stt.decoder,
        config.stt.joiner,
        config.stt.tokens,
        config.tts.model,
        config.tts.tokens,
    ]
    if config.tts.data_dir is not None:
        required.append(config.tts.data_dir)
    return [path for path in required if not path.exists()]
