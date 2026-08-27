"""The tool layer handed to the LLM: descriptions, JSON schemas, error containment."""

from __future__ import annotations

import pytest

from jarvis.config import JarvisConfig
from jarvis.tools import ToolBox


@pytest.fixture
def toolbox(real_config: JarvisConfig) -> ToolBox:
    return ToolBox(real_config)


# -- descriptions: what the model actually reads ----------------------------------------
def test_shell_description_lists_the_whitelist(toolbox: ToolBox) -> None:
    description = toolbox._shell_description()  # noqa: SLF001
    assert "open_downloads" in description
    assert "ĐÃ ĐƯỢC PHÊ DUYỆT" in description


def test_shell_description_names_the_real_parameter(toolbox: ToolBox) -> None:
    """It used to say ``command_name`` while the handler took ``name``."""
    description = toolbox._shell_description()  # noqa: SLF001
    assert "Tham số name" in description
    assert "command_name" not in description


def test_apps_description_lists_the_aliases(toolbox: ToolBox) -> None:
    assert "spotify" in toolbox._apps_description()  # noqa: SLF001


def test_media_description_lists_valid_actions(toolbox: ToolBox) -> None:
    description = toolbox._media_description()  # noqa: SLF001
    assert "play_pause" in description and "volume_up" in description


def test_music_description_warns_against_truncating_the_title(toolbox: ToolBox) -> None:
    description = toolbox._music_description()  # noqa: SLF001
    assert "Em của ngày hôm qua" in description
    assert "NGUYÊN VĂN" in description or "nguyên văn" in description


def test_clock_description_says_it_takes_no_parameters(toolbox: ToolBox) -> None:
    assert "không có" in toolbox._clock_description()  # noqa: SLF001


# -- schemas ---------------------------------------------------------------------------
def test_tool_definitions_expose_the_expected_names(toolbox: ToolBox) -> None:
    names = {spec.name for spec in toolbox.build_tool_defs()}
    assert {
        "get_current_datetime",
        "open_application",
        "play_song",
        "search_web",
        "control_media",
        "run_system_command",
    } <= names
    # browser_use is disabled by default in config.yaml
    assert "browse_web" not in names


def test_tool_definitions_carry_explicit_parameter_schemas(toolbox: ToolBox) -> None:
    spec = next(s for s in toolbox.build_tool_defs() if s.name == "open_application")
    schema = spec.to_openai_schema()
    parameters = schema["function"]["parameters"]
    assert parameters["properties"]["app_name"]["type"] == "string"
    assert parameters["required"] == ["app_name"]


def test_datetime_tool_takes_no_arguments(toolbox: ToolBox) -> None:
    spec = next(s for s in toolbox.build_tool_defs() if s.name == "get_current_datetime")
    assert spec.to_openai_schema()["function"]["parameters"]["properties"] == {}
    assert spec.required == []


def test_shell_schema_constrains_the_name_to_the_whitelist(toolbox: ToolBox) -> None:
    spec = next(s for s in toolbox.build_tool_defs() if s.name == "run_system_command")
    assert spec.properties["name"]["enum"] == toolbox.shell.names


def test_media_schema_constrains_the_action(toolbox: ToolBox) -> None:
    spec = next(s for s in toolbox.build_tool_defs() if s.name == "control_media")
    assert "volume_up" in spec.properties["action"]["enum"]


def test_disabled_tools_are_not_advertised(config_copy: JarvisConfig) -> None:
    config_copy.tools.clock.enabled = False
    config_copy.tools.music.enabled = False
    names = {spec.name for spec in ToolBox(config_copy).build_tool_defs()}
    assert "get_current_datetime" not in names
    assert "play_song" not in names


# -- error containment: tools return strings, they never raise into the LLM loop --------
def test_rejected_shell_command_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox.run_system_command("rm -rf /")
    assert output.startswith("Lỗi:")
    assert "whitelist" in output


def test_unknown_app_alias_returns_an_error_string(toolbox: ToolBox) -> None:
    assert toolbox.open_application("photoshop_khong_co").startswith("Lỗi:")


def test_invalid_media_action_returns_an_error_string(toolbox: ToolBox) -> None:
    assert toolbox.control_media("tu_huy").startswith("Lỗi:")


def test_empty_search_query_returns_an_error_string(toolbox: ToolBox) -> None:
    assert toolbox.search_web("").startswith("Lỗi:")


def test_missing_app_name_returns_actionable_guidance(toolbox: ToolBox) -> None:
    output = toolbox.open_application()
    assert output.startswith("Lỗi:")
    assert "app_name" in output


def test_missing_song_name_returns_actionable_guidance(toolbox: ToolBox) -> None:
    output = toolbox.play_song()
    assert output.startswith("Lỗi:")
    assert "song_name" in output


def test_invented_field_names_are_tolerated(toolbox: ToolBox, monkeypatch) -> None:  # noqa: ANN001
    """A small model often emits ``app`` or ``alias`` instead of ``app_name``."""
    monkeypatch.setattr(toolbox.apps, "open", lambda alias: f"[app:{alias}]")
    assert toolbox.open_application(alias="youtube") == "[app:youtube]"
    assert toolbox.open_application(app="youtube") == "[app:youtube]"


def test_disabled_browser_tool_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox.browse_web("mở youtube")
    assert output.startswith("Lỗi:")
    assert "browser_use" in output


def test_datetime_tool_returns_a_spoken_sentence(toolbox: ToolBox) -> None:
    output = toolbox.get_current_datetime()
    assert output.startswith("Bây giờ là")
    assert not output.startswith("Lỗi")


# -- bookkeeping -----------------------------------------------------------------------
def test_usage_tracking(toolbox: ToolBox) -> None:
    toolbox.reset_usage()
    toolbox.run_system_command("khong_ton_tai")
    toolbox.search_web("")
    assert toolbox.last_used == ["run_system_command", "search_web"]
    toolbox.reset_usage()
    assert toolbox.last_used == []


def test_summary_reports_every_tool(toolbox: ToolBox) -> None:
    summary = "\n".join(toolbox.summary())
    for name in ("clock", "shell", "apps", "media", "music", "web_search", "browser_use"):
        assert name in summary
