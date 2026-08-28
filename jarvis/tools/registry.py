"""Builds the tool list handed to ``LLMClient.act``.

Each entry is a :class:`~jarvis.llm.ToolSpec` with an explicit JSON schema and a
*dynamically generated* description: the whitelist names and app aliases from
``config.yaml`` are inlined so a small local model can pick the right identifier
without guessing.

Tool functions never raise: their return value is fed straight back to the model, so a
readable Vietnamese error string is far more useful than an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import JarvisConfig
from ..llm import ToolSpec
from ..logging_setup import get_logger
from .apps import AppLauncher, AppToolError, MediaController
from .browser import BrowserAgentTool, BrowserToolError
from .clock import Clock, ClockToolError
from .music import MusicPlayer, MusicToolError
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
    music: MusicPlayer = field(init=False)
    web: WebSearch = field(init=False)
    clock: Clock = field(init=False)
    browser: BrowserAgentTool = field(init=False)
    #: Names of tools invoked during the current turn (used for logging/tray detail).
    last_used: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        tools = self.config.tools
        self.shell = ShellRunner(tools.shell)
        self.apps = AppLauncher(tools.apps)
        self.media = MediaController(tools.media)
        self.web = WebSearch(tools.web_search)
        self.music = MusicPlayer(tools.music, self.web)
        self.clock = Clock(tools.clock)
        self.browser = BrowserAgentTool(tools.browser_use, self.config.llm)

    # -- bookkeeping -------------------------------------------------------------
    def reset_usage(self) -> None:
        self.last_used.clear()

    def _record(self, name: str) -> None:
        self.last_used.append(name)

    # -- tool implementations ----------------------------------------------------
    def run_system_command(self, name: str = "", argument: str = "", **extra: Any) -> str:
        """Run a named entry from the pre-approved command whitelist.

        ``name`` is only ever an identifier looked up by :class:`ShellRunner`; it is
        never interpreted as a shell command line.
        """
        self._record("run_system_command")
        requested = _first_value(name, extra.get("command_name"), extra.get("command"))
        if not requested:
            return (
                "Lỗi: thiếu tên lệnh. Hãy gọi lại run_system_command với name là một "
                f"trong: {', '.join(self.shell.names) or '(trống)'}."
            )
        try:
            return self.shell.run(requested, argument)
        except ShellToolError as exc:
            log.warning("run_system_command bị từ chối: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001 - never break the LLM loop
            log.exception("run_system_command lỗi không mong đợi")
            return f"Lỗi không mong đợi khi chạy lệnh: {exc}"

    def open_application(self, app_name: str = "", **extra: Any) -> str:
        """Open an app/site by alias.

        ``**extra`` absorbs the alternative field names small models like to invent
        (``app``, ``name``, ``alias``); a missing required field would otherwise abort
        the whole tool call.
        """
        self._record("open_application")
        requested = _first_value(
            app_name, extra.get("app"), extra.get("name"), extra.get("alias")
        )
        if not requested:
            return (
                "Lỗi: thiếu tên ứng dụng. Hãy gọi lại open_application với "
                'app_name là một alias hợp lệ, ví dụ app_name="youtube".'
            )
        try:
            return self.apps.open(requested)
        except AppToolError as exc:
            log.warning("open_application thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("open_application lỗi không mong đợi")
            return f"Lỗi không mong đợi khi mở ứng dụng: {exc}"

    def control_media(self, action: str = "", **extra: Any) -> str:
        """Press a media/volume key."""
        self._record("control_media")
        requested = _first_value(action, extra.get("command"), extra.get("name"))
        if not requested:
            return (
                "Lỗi: thiếu hành động. Chọn một trong: "
                + ", ".join(MediaController.ACTIONS)
                + "."
            )
        try:
            return self.media.perform(requested)
        except AppToolError as exc:
            log.warning("control_media thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("control_media lỗi không mong đợi")
            return f"Lỗi không mong đợi khi điều khiển media: {exc}"

    def play_song(self, song_name: str = "", **extra: Any) -> str:
        """Play a song by name."""
        self._record("play_song")
        log.debug("play_song args: song_name=%r extra=%r", song_name, extra)
        requested = _first_value(
            song_name,
            extra.get("song"),
            extra.get("query"),
            extra.get("title"),
            *(str(value) for value in extra.values() if value),
        )
        if not requested:
            return (
                "Lỗi: thiếu tên bài hát. Hãy gọi lại play_song với "
                'song_name là tên bài hát, ví dụ song_name="Em của ngày hôm qua".'
            )
        try:
            return self.music.play(requested)
        except MusicToolError as exc:
            log.warning("play_song thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("play_song lỗi không mong đợi")
            return f"Lỗi không mong đợi khi phát nhạc: {exc}"

    def search_web(self, query: str = "", **extra: Any) -> str:
        """Search the web and return one block of URL-free material to synthesise."""
        self._record("search_web")
        requested = _first_value(query, extra.get("q"), extra.get("search_query"))
        try:
            return self.web.answer_context(requested)
        except WebSearchError as exc:
            log.warning("search_web thất bại: %s", exc)
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("search_web lỗi không mong đợi")
            return f"Lỗi không mong đợi khi tìm kiếm: {exc}"

    def get_current_datetime(self, **_extra: Any) -> str:
        """Read the machine clock."""
        self._record("get_current_datetime")
        try:
            return self.clock.describe()
        except ClockToolError as exc:
            return f"Lỗi: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("get_current_datetime lỗi không mong đợi")
            return f"Lỗi không mong đợi khi xem ngày giờ: {exc}"

    def browse_web(self, task: str = "", **extra: Any) -> str:
        """Hand a multi-step web task to browser-use."""
        self._record("browse_web")
        requested = _first_value(task, extra.get("query"), extra.get("instruction"))
        try:
            return self.browser.run(requested)
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
            "Tham số name phải là một trong các tên dưới đây, không được tự nghĩ ra lệnh mới. "
            "Chỉ truyền argument khi mô tả lệnh nói rằng nó nhận tham số.\n"
            "Các lệnh khả dụng:\n" + self.shell.catalogue()
        )

    def _apps_description(self) -> str:
        return (
            "Mở một ứng dụng hoặc trang web đã cấu hình sẵn trên máy người dùng. "
            "BẮT BUỘC gọi tool này khi người dùng nói 'mở ...', không được chỉ nói "
            "là đã mở. Tham số app_name phải là một alias trong danh sách dưới đây.\n"
            "Các alias khả dụng:\n" + self.apps.catalogue()
        )

    @staticmethod
    def _media_description() -> str:
        return (
            "Điều khiển phát nhạc và âm lượng hệ thống bằng phím media. "
            "Tham số action là một trong: " + ", ".join(MediaController.ACTIONS) + "."
        )

    def _music_description(self) -> str:
        return (
            "Phát một bài hát cụ thể theo tên. Dùng công cụ này khi người dùng nói "
            "'mở bài hát ...', 'phát bài ...', 'nghe bài ...'. "
            "BẮT BUỘC truyền song_name là TOÀN BỘ tên bài hát đúng nguyên văn người "
            'dùng đã nói, ví dụ song_name="Em của ngày hôm qua" (không được cắt bớt '
            "thành \"Em\"). Không tự sửa hay tự nghĩ ra tên khác. "
            f"Nguồn phát: {self.music.provider}."
        )

    @staticmethod
    def _search_description() -> str:
        return (
            "Tìm kiếm thông tin mới trên Internet (tin tức, thời tiết, giá cả, sự kiện). "
            "Truyền query là câu truy vấn tiếng Việt ngắn gọn. "
            "Công cụ trả về một khối tư liệu đã gộp sẵn từ nhiều nguồn và đã bỏ hết "
            "đường dẫn. Việc của bạn là TỔNG HỢP khối tư liệu đó thành MỘT câu trả lời "
            "tiếng Việt duy nhất, dài 1 đến 2 câu, có con số và mốc thời gian nếu tư "
            "liệu nêu. TUYỆT ĐỐI không liệt kê nhiều kết quả, không đánh số, không đọc "
            "tên trang web hay địa chỉ web, vì người dùng đang nghe bằng tai."
        )

    @staticmethod
    def _clock_description() -> str:
        return (
            "Xem ngày, thứ và giờ hiện tại trên máy người dùng. Công cụ này không có "
            "tham số. BẮT BUỘC gọi nó cho mọi câu hỏi về ngày giờ hiện tại, vì bạn "
            "không tự biết hôm nay là ngày nào."
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
    def build_tool_defs(self) -> list[ToolSpec]:
        """Return a :class:`ToolSpec` for every enabled tool."""
        definitions: list[ToolSpec] = []

        if self.clock.enabled:
            definitions.append(
                ToolSpec(
                    name="get_current_datetime",
                    description=self._clock_description(),
                    handler=self.get_current_datetime,
                )
            )
        if self.apps.enabled:
            definitions.append(
                ToolSpec(
                    name="open_application",
                    description=self._apps_description(),
                    handler=self.open_application,
                    properties={
                        "app_name": {
                            "type": "string",
                            "description": "Alias của ứng dụng, ví dụ youtube, spotify, chrome.",
                        }
                    },
                    required=["app_name"],
                )
            )
        if self.music.enabled:
            definitions.append(
                ToolSpec(
                    name="play_song",
                    description=self._music_description(),
                    handler=self.play_song,
                    properties={
                        "song_name": {
                            "type": "string",
                            "description": (
                                "Toàn bộ tên bài hát, giữ đúng nguyên văn người dùng nói."
                            ),
                        }
                    },
                    required=["song_name"],
                )
            )
        if self.web.enabled:
            definitions.append(
                ToolSpec(
                    name="search_web",
                    description=self._search_description(),
                    handler=self.search_web,
                    properties={
                        "query": {
                            "type": "string",
                            "description": "Câu truy vấn tìm kiếm, tiếng Việt, ngắn gọn.",
                        }
                    },
                    required=["query"],
                )
            )
        if self.media.enabled:
            definitions.append(
                ToolSpec(
                    name="control_media",
                    description=self._media_description(),
                    handler=self.control_media,
                    properties={
                        "action": {
                            "type": "string",
                            "enum": list(MediaController.ACTIONS),
                            "description": "Hành động media cần thực hiện.",
                        }
                    },
                    required=["action"],
                )
            )
        if self.shell.enabled:
            definitions.append(
                ToolSpec(
                    name="run_system_command",
                    description=self._shell_description(),
                    handler=self.run_system_command,
                    properties={
                        "name": {
                            "type": "string",
                            "enum": self.shell.names,
                            "description": "Tên lệnh trong whitelist.",
                        },
                        "argument": {
                            "type": "string",
                            "description": (
                                "Chỉ dùng cho những lệnh được ghi là nhận 1 tham số."
                            ),
                        },
                    },
                    required=["name"],
                )
            )
        if self.browser.enabled:
            definitions.append(
                ToolSpec(
                    name="browse_web",
                    description=self._browser_description(),
                    handler=self.browse_web,
                    properties={
                        "task": {
                            "type": "string",
                            "description": "Mô tả tác vụ web nhiều bước cần thực hiện.",
                        }
                    },
                    required=["task"],
                )
            )

        log.info(
            "Đã bật %d tool: %s",
            len(definitions),
            ", ".join(spec.name for spec in definitions),
        )
        return definitions

    def _search_backend_detail(self) -> str:
        """Extra detail for the ``web_search`` doctor line.

        Each backend has one prerequisite that is invisible until a search runs: an
        optional package for ``ddgs``, a URL for ``searxng``, an API key for ``brave``.
        Reporting it here beats letting the first spoken question be what finds out.
        """
        config = self.config.tools.web_search
        if config.backend == "brave":
            # Never the key itself: this line is printed to the console and pasted into
            # bug reports. And only whether a key was *found*, not that it works - an
            # invalid token is only discovered when a search runs, where Brave answers
            # HTTP 422 for it.
            state = "đã có key" if config.resolved_brave_api_key() else "CHƯA có key"
            return (
                f", {state}, lang={config.brave_search_lang}, country={config.brave_country}"
                f", extra_snippets={'on' if config.brave_extra_snippets else 'off'}"
            )
        if config.backend == "ddgs":
            try:
                import ddgs  # noqa: F401

                installed = "đã cài"
            except Exception:  # noqa: BLE001
                installed = "CHƯA cài"
            return f", engines={config.ddgs_engines}, package {installed}"
        if config.backend == "searxng":
            return f", url={config.searxng_url or 'CHƯA đặt'}"
        return ""

    def summary(self) -> list[str]:
        """Short status lines for ``jarvis doctor``."""
        rows = [
            f"clock          : {'on' if self.clock.enabled else 'off'}",
            f"apps           : {'on' if self.apps.enabled else 'off'} "
            f"({len(self.apps.aliases)} alias)",
            f"music          : {'on' if self.music.enabled else 'off'} "
            f"(provider={self.config.tools.music.provider})",
            f"web_search     : {'on' if self.web.enabled else 'off'} "
            f"(backend={self.config.tools.web_search.backend}{self._search_backend_detail()})",
            f"media          : {'on' if self.media.enabled else 'off'}",
            f"shell          : {'on' if self.shell.enabled else 'off'} "
            f"({len(self.shell.names)} lệnh whitelist)",
        ]
        if self.browser.enabled:
            rows.append(
                f"browser_use    : on (package {'đã cài' if self.browser.available else 'CHƯA cài'})"
            )
        else:
            rows.append("browser_use    : off")
        return rows


def _first_value(*candidates: Any) -> str:
    """First non-blank candidate, stripped. Used to tolerate invented field names."""
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""
