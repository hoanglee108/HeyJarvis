"""Whitelisted system-command tool (mvp.md Task 8).

Security model
--------------
The LLM never composes a command line. It picks a *name* from a whitelist that the
user wrote in ``config.yaml``; we look up the pre-approved executable and argument
vector ourselves and run it with ``shell=False``.

A command may optionally accept one free-text argument (``allow_argument: true``),
which is checked against a regex before substitution. Argument passing is rejected
outright for PowerShell commands (see ``ShellCommandSpec`` validation) because
splicing text into ``-Command`` would reintroduce injection.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ..config import ShellCommandSpec, ShellToolConfig
from ..logging_setup import get_logger

log = get_logger("jarvis.tools.shell")

MAX_OUTPUT_CHARS = 1500
#: Hide the console window that would otherwise flash on screen.
_CREATE_NO_WINDOW = 0x08000000


class ShellToolError(RuntimeError):
    """The requested command is not allowed or could not be executed."""


class ShellRunner:
    """Executes only the commands present in the configured whitelist."""

    def __init__(self, config: ShellToolConfig) -> None:
        self._config = config
        self._commands: dict[str, ShellCommandSpec] = {
            spec.name: spec for spec in config.whitelist
        }

    # -- introspection -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._commands)

    @property
    def names(self) -> list[str]:
        return sorted(self._commands)

    def catalogue(self) -> str:
        """Rendered list of commands, embedded in the tool description for the LLM."""
        lines = []
        for name in self.names:
            spec = self._commands[name]
            suffix = " (nhận 1 tham số)" if spec.allow_argument else ""
            lines.append(f"- {name}: {spec.description}{suffix}")
        return "\n".join(lines)

    # -- execution ---------------------------------------------------------------
    def build_argv(self, name: str, argument: str = "") -> list[str]:
        """Resolve a whitelist entry into a concrete argv (also used by tests)."""
        spec = self._commands.get(name.strip().lower())
        if spec is None:
            raise ShellToolError(
                f"Lệnh {name!r} không nằm trong whitelist. "
                f"Các lệnh được phép: {', '.join(self.names) or '(trống)'}"
            )

        argument = (argument or "").strip()
        if argument and not spec.allow_argument:
            raise ShellToolError(f"Lệnh {spec.name!r} không nhận tham số.")
        if argument:
            if not re.fullmatch(spec.argument_pattern, argument):
                raise ShellToolError(
                    f"Tham số {argument!r} không hợp lệ cho lệnh {spec.name!r} "
                    f"(cần khớp {spec.argument_pattern})."
                )
            if any(char in argument for char in ('"', "'", "`", "\0")):
                raise ShellToolError("Tham số chứa ký tự không được phép.")

        expanded_args = [os.path.expandvars(arg) for arg in spec.args]

        if spec.use_powershell:
            script = " ".join(expanded_args)
            return [
                spec.executable or "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ]

        argv = [spec.executable, *expanded_args]
        if argument:
            if spec.argument_args:
                argv.extend(
                    os.path.expandvars(part).replace("{arg}", argument)
                    for part in spec.argument_args
                )
            else:
                argv.append(argument)
        return argv

    def run(self, name: str, argument: str = "") -> str:
        if not self._config.enabled:
            raise ShellToolError("Công cụ lệnh hệ thống đang bị tắt trong config.")

        normalised = name.strip().lower()
        spec = self._commands.get(normalised)
        argv = self.build_argv(normalised, argument)
        assert spec is not None  # build_argv already validated the name

        log.info("Chạy lệnh whitelist %s: %s", spec.name, argv)
        try:
            completed = subprocess.run(  # noqa: S603 - argv comes from the whitelist
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=spec.timeout_s,
                shell=False,
                creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except FileNotFoundError as exc:
            raise ShellToolError(
                f"Không tìm thấy chương trình {spec.executable!r} cho lệnh {spec.name!r}."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ShellToolError(
                f"Lệnh {spec.name!r} chạy quá {spec.timeout_s:.0f} giây và đã bị dừng."
            ) from exc
        except OSError as exc:
            raise ShellToolError(f"Không chạy được lệnh {spec.name!r}: {exc}") from exc

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        if completed.returncode != 0:
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise ShellToolError(f"Lệnh {spec.name!r} thất bại: {_truncate(detail)}")

        if not stdout:
            return f"Đã thực hiện: {spec.description}"
        return _truncate(stdout)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    collapsed = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + " …(đã cắt ngắn)"


def default_whitelist_path() -> Path:  # pragma: no cover - convenience for docs
    return Path("config.yaml")
