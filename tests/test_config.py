"""mvp.md Task 1: config loads, validates, and rejects bad input."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from jarvis.config import ConfigError, JarvisConfig, load_config

MINIMAL = """
stt:
  encoder: models/stt/enc.onnx
  decoder: models/stt/dec.onnx
  joiner: models/stt/join.onnx
  tokens: models/stt/tokens.txt
tts:
  model: models/tts/voice.onnx
  tokens: models/tts/tokens.txt
"""


def write_config(tmp_path: Path, body: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_repository_config_is_valid(real_config: JarvisConfig) -> None:
    assert real_config.stt.encoder.name.endswith(".onnx")
    assert real_config.tts.data_dir is not None
    assert real_config.llm.api_host == "localhost:1234"
    assert real_config.tools.shell.whitelist, "whitelist mẫu không nên trống"


def test_minimal_config_uses_defaults(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL))
    assert config.audio.sample_rate == 16000
    assert config.wake_word.model == "hey_jarvis"
    assert config.tray.enabled is True


def test_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL))
    assert config.stt.encoder == (tmp_path / "models/stt/enc.onnx").resolve()
    assert config.stt.encoder.is_absolute()


def test_missing_required_section_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "tts:\n  model: a.onnx\n  tokens: b.txt\n")
    with pytest.raises(ConfigError, match="không hợp lệ"):
        load_config(path)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    body = MINIMAL.replace("  tokens: models/stt/tokens.txt\n", "", 1)
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(tmp_path, body))
    assert "tokens" in str(excinfo.value)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    body = MINIMAL + "\nwake_word:\n  treshold: 0.5\n"
    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(tmp_path, body))
    assert "treshold" in str(excinfo.value)


def test_out_of_range_value_is_rejected(tmp_path: Path) -> None:
    body = MINIMAL + "\nwake_word:\n  threshold: 1.5\n"
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, body))


def test_api_host_scheme_is_stripped(tmp_path: Path) -> None:
    body = MINIMAL + "\nllm:\n  api_host: http://127.0.0.1:4321/\n"
    config = load_config(write_config(tmp_path, body))
    assert config.llm.api_host == "127.0.0.1:4321"


def shell_whitelist_config(entries: str) -> str:
    """MINIMAL plus a ``tools.shell.whitelist`` block (already at column 0)."""
    return MINIMAL + "\ntools:\n  shell:\n    whitelist:\n" + entries


def test_duplicate_shell_command_names_rejected(tmp_path: Path) -> None:
    body = shell_whitelist_config(
        "      - name: dup\n"
        "        description: a\n"
        "        executable: explorer.exe\n"
        "      - name: dup\n"
        "        description: b\n"
        "        executable: explorer.exe\n"
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write_config(tmp_path, body))


def test_powershell_with_argument_is_rejected(tmp_path: Path) -> None:
    """Splicing LLM text into a PowerShell -Command string must not be possible."""
    body = shell_whitelist_config(
        "      - name: risky\n"
        "        description: injection\n"
        "        executable: powershell.exe\n"
        "        use_powershell: true\n"
        "        allow_argument: true\n"
    )
    with pytest.raises(ConfigError, match="command injection"):
        load_config(write_config(tmp_path, body))


def test_bad_shell_name_is_rejected(tmp_path: Path) -> None:
    body = shell_whitelist_config(
        '      - name: "Open Downloads!"\n'
        "        description: x\n"
        "        executable: explorer.exe\n"
    )
    with pytest.raises(ConfigError, match="snake_case"):
        load_config(write_config(tmp_path, body))


def test_local_override_is_merged(tmp_path: Path) -> None:
    write_config(tmp_path, MINIMAL)
    write_config(
        tmp_path,
        "llm:\n  model: my-local-model\naudio:\n  sample_rate: 16000\n",
        name="config.local.yaml",
    )
    config = load_config(tmp_path / "config.yaml")
    assert config.llm.model == "my-local-model"


def test_invalid_yaml_reports_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("stt: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Không tìm thấy file config"):
        load_config(tmp_path / "nope.yaml")
