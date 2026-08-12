"""mvp.md Task 8: the shell tool must only ever run whitelisted commands."""

from __future__ import annotations

import pytest

from jarvis.config import ShellCommandSpec, ShellToolConfig
from jarvis.tools import ShellRunner, ShellToolError


def make_runner(**overrides: object) -> ShellRunner:
    specs = [
        ShellCommandSpec(
            name="open_downloads",
            description="Mở thư mục Downloads",
            executable="explorer.exe",
            args=["%USERPROFILE%\\Downloads"],
        ),
        ShellCommandSpec(
            name="open_folder",
            description="Mở một thư mục cụ thể",
            executable="explorer.exe",
            allow_argument=True,
            argument_pattern=r"^[A-Za-z]:\\[\w\s\-.\\()]{0,150}$",
            argument_args=["{arg}"],
        ),
        ShellCommandSpec(
            name="current_datetime",
            description="Xem ngày giờ",
            executable="powershell.exe",
            use_powershell=True,
            args=["Get-Date -Format 'yyyy-MM-dd'"],
        ),
    ]
    return ShellRunner(ShellToolConfig(enabled=True, whitelist=specs, **overrides))  # type: ignore[arg-type]


# -- rejection ------------------------------------------------------------------------
def test_command_outside_whitelist_is_rejected() -> None:
    runner = make_runner()
    with pytest.raises(ShellToolError, match="không nằm trong whitelist"):
        runner.run("del /f /s /q C:\\")


def test_rejection_message_lists_allowed_commands() -> None:
    runner = make_runner()
    with pytest.raises(ShellToolError) as excinfo:
        runner.build_argv("format_c")
    assert "open_downloads" in str(excinfo.value)


def test_argument_rejected_when_command_does_not_accept_one() -> None:
    runner = make_runner()
    with pytest.raises(ShellToolError, match="không nhận tham số"):
        runner.build_argv("open_downloads", "C:\\Windows")


def test_argument_must_match_the_pattern() -> None:
    runner = make_runner()
    with pytest.raises(ShellToolError, match="không hợp lệ"):
        runner.build_argv("open_folder", "C:\\Windows & calc.exe")
    with pytest.raises(ShellToolError, match="không hợp lệ"):
        runner.build_argv("open_folder", "../../etc/passwd")


def test_quotes_and_backticks_are_refused() -> None:
    runner = make_runner()
    for hostile in ('C:\\a"b', "C:\\a'b", "C:\\a`b"):
        with pytest.raises(ShellToolError):
            runner.build_argv("open_folder", hostile)


def test_disabled_tool_refuses_to_run() -> None:
    runner = ShellRunner(ShellToolConfig(enabled=False, whitelist=[]))
    with pytest.raises(ShellToolError, match="đang bị tắt"):
        runner.run("open_downloads")
    assert runner.enabled is False


# -- argv construction ----------------------------------------------------------------
def test_env_vars_are_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USERPROFILE", "C:\\Users\\test")
    argv = make_runner().build_argv("open_downloads")
    assert argv == ["explorer.exe", "C:\\Users\\test\\Downloads"]


def test_valid_argument_is_substituted() -> None:
    argv = make_runner().build_argv("open_folder", "D:\\Projects\\jarvis")
    assert argv == ["explorer.exe", "D:\\Projects\\jarvis"]


def test_powershell_commands_run_non_interactively() -> None:
    argv = make_runner().build_argv("current_datetime")
    assert argv[0] == "powershell.exe"
    assert "-NoProfile" in argv and "-NonInteractive" in argv
    assert argv[-1] == "Get-Date -Format 'yyyy-MM-dd'"


def test_name_lookup_is_case_insensitive() -> None:
    assert make_runner().build_argv("OPEN_DOWNLOADS")[0] == "explorer.exe"


def test_catalogue_marks_commands_that_take_an_argument() -> None:
    catalogue = make_runner().catalogue()
    assert "open_folder" in catalogue
    assert "nhận 1 tham số" in catalogue


# -- real execution -------------------------------------------------------------------
def test_whitelisted_powershell_command_actually_runs() -> None:
    """Integration check: one safe read-only command really executes."""
    output = make_runner().run("current_datetime")
    assert output
    assert output[:2].isdigit(), f"kỳ vọng ngày dạng yyyy-MM-dd, nhận {output!r}"


def test_missing_executable_produces_a_clear_error() -> None:
    spec = ShellCommandSpec(
        name="ghost",
        description="không tồn tại",
        executable="chuong_trinh_khong_ton_tai_12345.exe",
    )
    runner = ShellRunner(ShellToolConfig(enabled=True, whitelist=[spec]))
    with pytest.raises(ShellToolError, match="Không tìm thấy chương trình"):
        runner.run("ghost")
