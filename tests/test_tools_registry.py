"""The tool layer handed to the LLM: descriptions, error containment, schemas."""

from __future__ import annotations

import pytest

from jarvis.config import JarvisConfig
from jarvis.tools import ToolBox


@pytest.fixture
def toolbox(real_config: JarvisConfig) -> ToolBox:
    return ToolBox(real_config)


def test_shell_description_lists_the_whitelist(toolbox: ToolBox) -> None:
    description = toolbox._shell_description()  # noqa: SLF001
    assert "open_downloads" in description
    assert "ĐÃ ĐƯỢC PHÊ DUYỆT" in description


def test_apps_description_lists_the_aliases(toolbox: ToolBox) -> None:
    assert "spotify" in toolbox._apps_description()  # noqa: SLF001


def test_media_description_lists_valid_actions(toolbox: ToolBox) -> None:
    description = toolbox._media_description()  # noqa: SLF001
    assert "play_pause" in description and "volume_up" in description


def test_tool_definitions_expose_the_expected_names(toolbox: ToolBox) -> None:
    definitions = toolbox.build_tool_defs()
    names = {getattr(d, "name", None) for d in definitions}
    assert {"run_system_command", "open_application", "control_media", "search_web"} <= names
    # browser_use is disabled by default in config.yaml
    assert "browse_web" not in names


def test_tool_definitions_carry_parameter_schemas(toolbox: ToolBox) -> None:
    definition = next(d for d in toolbox.build_tool_defs() if d.name == "open_application")
    parameters = getattr(definition, "parameters", None)
    assert parameters is not None, "SDK phải suy ra schema tham số từ type hints"


# -- error containment: tools return strings, they never raise into the LLM loop --------
def test_rejected_shell_command_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox._run_system_command("rm -rf /")  # noqa: SLF001
    assert output.startswith("Lỗi:")
    assert "whitelist" in output


def test_unknown_app_alias_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox._open_application("photoshop_khong_co")  # noqa: SLF001
    assert output.startswith("Lỗi:")


def test_invalid_media_action_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox._control_media("tu_huy")  # noqa: SLF001
    assert output.startswith("Lỗi:")


def test_empty_search_query_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox._search_web("")  # noqa: SLF001
    assert output.startswith("Lỗi:")


def test_disabled_browser_tool_returns_an_error_string(toolbox: ToolBox) -> None:
    output = toolbox._browse_web("mở youtube")  # noqa: SLF001
    assert output.startswith("Lỗi:")
    assert "browser_use" in output


def test_usage_tracking(toolbox: ToolBox) -> None:
    toolbox.reset_usage()
    toolbox._run_system_command("khong_ton_tai")  # noqa: SLF001
    toolbox._search_web("")  # noqa: SLF001
    assert toolbox.last_used == ["run_system_command", "search_web"]
    toolbox.reset_usage()
    assert toolbox.last_used == []


def test_summary_reports_every_tool(toolbox: ToolBox) -> None:
    summary = "\n".join(toolbox.summary())
    for name in ("shell", "apps", "media", "web_search", "browser_use"):
        assert name in summary
