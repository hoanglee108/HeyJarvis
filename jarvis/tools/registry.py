"""Builds the tool list handed to ``LLMClient.act``.

Each entry is a ``lmstudio.ToolFunctionDef`` with a *dynamically generated*
description: the whitelist names and app aliases from ``config.yaml`` are inlined so
a small 3B model can pick the right identifier without guessing.

Tool functions never raise: the SDK feeds their return value straight back to the
model, so a readable Vietnamese error string is far more useful than an exception.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import JarvisConfig
from ..logging_setup import get_logger
from .apps import AppLauncher, AppToolError, MediaController
from .browser import BrowserAgentTool, BrowserToolError
from .shell import ShellRunner, ShellToolError
from .websearch import WebSearch, WebSearchError

log = get_logger("jarvis.tools")


@dataclass
class ToolBox:
    """Owns the concrete tool implementations and exposes them to the LLM."""

    config: JarvisConfig
    shell: ShellRunner = field(init=False)
    apps: AppLauncher = field(init=False)
    media: MediaController = field(init=False)
    web: WebSearch = field(init=False)
    browser: BrowserAgentTool = field(init=False)
    #: Names of tools invoked during the current turn (used for logging/tray detail).
    last_used: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        tools = self.config.tools
        self.shell = ShellRunner(tools.shell)
        self.apps = AppLauncher(tools.apps)
        self.media = MediaController(tools.media)
        self.web = WebSearch(tools.web_search)
        self.browser = BrowserAgentTool(tools.browser_use, self.config.llm)

    # -- bookkeeping -------------------------------------------------------------
    def reset_usage(self) -> None:
        self.last_used.clear()

    def _record(self, name: str) -> None:
        self.last_used.append(name)

    # -- tool implementations ----------------------------------------------------
    def _run_system_command(self, name: str, argument: str = "") -> str:
        """Run a named entry from the pre-approved command whitelist.

        ``name`` intentionally mirrors the concise field convention small local
        models tend to emit. It is still only an identifier looked up by
        :class:`ShellRunner`; it is never interpreted as a shell command.
        """
        self._record("run_system_command")
        try:
            return self.shell.run(name, argument)
        except ShellToolError as exc:
            log.warning("run_system_command bị từ chối: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001 - never break the LLM loop
            log.exception("run_system_command lỗi không mong đợi")
            return f"Lỗi không mong đợi khi chạy lệnh: {exc}"

    def _open_application(self, app_name: str) -> str:
        self._record("open_application")
        try:
            return self.apps.open(app_name)
        except AppToolError as exc:
            log.warning("open_application thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("open_application lỗi không mong đợi")
            return f"Lỗi không mong đợi khi mở ứng dụng: {exc}"

    def _control_media(self, action: str) -> str:
        self._record("control_media")
        try:
            return self.media.perform(action)
        except AppToolError as exc:
            log.warning("control_media thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("control_media lỗi không mong đợi")
            return f"Lỗi không mong đợi khi điều khiển media: {exc}"

    def _search_web(self, query: str) -> str:
        self._record("search_web")
        try:
            return self.web.search_as_text(query)
        except WebSearchError as exc:
            log.warning("search_web thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("search_web lỗi không mong đợi")
            return f"Lỗi không mong đợi khi tìm kiếm: {exc}"

    def _browse_web(self, task: str) -> str:
        self._record("browse_web")
        try:
            return self.browser.run(task)
        except BrowserToolError as exc:
            log.warning("browse_web thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("browse_web lỗi không mong đợi")
            return f"Lỗi không mong đợi khi điều khiển trình duyệt: {exc}"

    # -- descriptions ------------------------------------------------------------
    def _shell_description(self) -> str:
        return (
            "Chạy một lệnh hệ thống Windows ĐÃ ĐƯỢC PHÊ DUYỆT TRƯỚC. "
            "Tham số command_name phải là một trong các tên dưới đây, không được tự nghĩ ra lệnh mới. "
            "Chỉ truyền argument khi mô tả lệnh nói rằng nó nhận tham số.\n"
            "Các lệnh khả dụng:\n" + self.shell.catalogue()
        )

    def _apps_description(self) -> str:
        return (
            "Mở một ứng dụng hoặc trang web đã cấu hình sẵn trên máy người dùng. "
            "Tham số app_name phải là một alias trong danh sách dưới đây.\n"
            "Các alias khả dụng:\n" + self.apps.catalogue()
        )

    def _media_description(self) -> str:
        return (
            "Điều khiển phát nhạc và âm lượng hệ thống bằng phím media. "
            "Tham số action là một trong: " + ", ".join(MediaController.ACTIONS) + "."
        )

    @staticmethod
    def _search_description() -> str:
        return (
            "Tìm kiếm thông tin mới trên Internet (tin tức, thời tiết, giá cả, sự kiện). "
            "Truyền query là câu truy vấn tiếng Việt ngắn gọn. "
            "Trả về danh sách tiêu đề và đoạn trích để bạn tóm tắt lại cho người dùng."
        )

    @staticmethod
    def _browser_description() -> str:
        return (
            "Điều khiển trình duyệt để thực hiện tác vụ web nhiều bước "
            "(mở trang, tìm trong trang, bấm vào kết quả). "
            "Chỉ dùng khi search_web là không đủ, vì công cụ này chậm. "
            "Truyền task là mô tả ngắn gọn tác vụ cần làm."
        )

    # -- assembly ----------------------------------------------------------------
    def build_tool_defs(self) -> list[Any]:
        """Return ``ToolFunctionDef`` objects for every enabled tool."""
        try:
            from lmstudio import ToolFunctionDef
        except ImportError:  # pragma: no cover
            log.warning("Không import được lmstudio.ToolFunctionDef; bỏ qua tool-calling")
            return []

        definitions: list[Any] = []

        def add(fn: Callable[..., Any], name: str, description: str) -> None:
            definitions.append(
                ToolFunctionDef.from_callable(fn, name=name, description=description)
            )

        if self.shell.enabled:
            add(self._run_system_command, "run_system_command", self._shell_description())
        if self.apps.enabled:
            add(self._open_application, "open_application", self._apps_description())
        if self.media.enabled:
            add(self._control_media, "control_media", self._media_description())
        if self.web.enabled:
            add(self._search_web, "search_web", self._search_description())
        if self.browser.enabled:
            add(self._browse_web, "browse_web", self._browser_description())

        log.info(
            "Đã bật %d tool: %s",
            len(definitions),
            ", ".join(getattr(d, "name", "?") for d in definitions),
        )
        return definitions

    def summary(self) -> list[str]:
        """Short status lines for ``jarvis doctor``."""
        rows = [
            f"shell          : {'on' if self.shell.enabled else 'off'} "
            f"({len(self.shell.names)} lệnh whitelist)",
            f"apps           : {'on' if self.apps.enabled else 'off'} "
            f"({len(self.apps.aliases)} alias)",
            f"media          : {'on' if self.media.enabled else 'off'}",
            f"web_search     : {'on' if self.web.enabled else 'off'} "
            f"(backend={self.config.tools.web_search.backend})",
        ]
        if self.browser.enabled:
            rows.append(
                f"browser_use    : on (package {'đã cài' if self.browser.available else 'CHƯA cài'})"
            )
        else:
            rows.append("browser_use    : off")
        return rows
