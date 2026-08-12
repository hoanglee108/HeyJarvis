"""App-launch alias tool and media-key control (mvp.md Task 9).

Aliases come from ``config.yaml`` so the assistant can only start things the user
explicitly listed. Media keys go through PyAutoGUI, which needs no elevated rights
and no reasoning loop.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import unicodedata
import webbrowser

from ..config import AppAliasSpec, AppsToolConfig, MediaToolConfig
from ..logging_setup import get_logger

log = get_logger("jarvis.tools.apps")

_CREATE_NO_WINDOW = 0x08000000


class AppToolError(RuntimeError):
    """Alias unknown or the target could not be launched."""


def _normalise(name: str) -> str:
    """Fold Vietnamese diacritics and punctuation so 'Máy tính' matches 'may_tinh'."""
    text = unicodedata.normalize("NFD", (name or "").strip().lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return "_".join(part for part in text.replace("-", " ").replace("_", " ").split())


class AppLauncher:
    """Resolves an alias to a target and opens it with the right Windows mechanism."""

    def __init__(self, config: AppsToolConfig) -> None:
        self._config = config
        # Index both the literal keys and their diacritic-folded forms.
        self._index: dict[str, tuple[str, AppAliasSpec]] = {}
        for alias, spec in config.aliases.items():
            self._index[alias] = (alias, spec)
            self._index[_normalise(alias)] = (alias, spec)

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.aliases)

    @property
    def aliases(self) -> list[str]:
        return sorted(self._config.aliases)

    def catalogue(self) -> str:
        lines = []
        for alias in self.aliases:
            spec = self._config.aliases[alias]
            lines.append(f"- {alias}: {spec.description or spec.target}")
        return "\n".join(lines)

    def resolve(self, alias: str) -> tuple[str, AppAliasSpec]:
        raw = (alias or "").strip().lower()
        if raw in self._index:
            return self._index[raw]
        folded = _normalise(alias)
        if folded in self._index:
            return self._index[folded]

        # STT output is fuzzy; accept a close spelling before giving up.
        matches = difflib.get_close_matches(folded, list(self._index), n=1, cutoff=0.75)
        if matches:
            return self._index[matches[0]]
        raise AppToolError(
            f"Chưa có alias {alias!r}. Các alias khả dụng: {', '.join(self.aliases) or '(trống)'}. "
            "Thêm alias mới trong config.yaml > tools.apps.aliases."
        )

    def open(self, alias: str) -> str:
        if not self._config.enabled:
            raise AppToolError("Công cụ mở ứng dụng đang bị tắt trong config.")

        key, spec = self.resolve(alias)
        target = os.path.expandvars(spec.target)
        label = spec.description or key
        log.info("Mở %s (%s: %s)", label, spec.kind, target)

        try:
            if spec.kind == "url":
                if not webbrowser.open(target, new=2):
                    raise AppToolError(f"Không mở được đường dẫn {target}.")
            elif spec.kind == "uri":
                os.startfile(target)  # noqa: S606 - protocol handler from user config
            elif spec.kind == "path":
                os.startfile(target)  # noqa: S606
            else:  # exe
                subprocess.Popen(  # noqa: S603 - executable comes from user config
                    [target, *spec.args],
                    shell=False,
                    creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        except FileNotFoundError as exc:
            raise AppToolError(
                f"Không tìm thấy {target!r}. Kiểm tra lại alias {key!r} trong config.yaml."
            ) from exc
        except OSError as exc:
            raise AppToolError(f"Không mở được {label}: {exc}") from exc

        return f"Đã mở {label}."


class MediaController:
    """Media transport and volume via virtual key presses."""

    ACTIONS = (
        "play_pause",
        "next",
        "previous",
        "stop",
        "volume_up",
        "volume_down",
        "mute",
    )

    _KEYS = {
        "play_pause": "playpause",
        "next": "nexttrack",
        "previous": "prevtrack",
        "stop": "stop",
        "volume_up": "volumeup",
        "volume_down": "volumedown",
        "mute": "volumemute",
    }

    _LABELS = {
        "play_pause": "Đã bật/tạm dừng phát nhạc.",
        "next": "Đã chuyển bài tiếp theo.",
        "previous": "Đã quay lại bài trước.",
        "stop": "Đã dừng phát nhạc.",
        "volume_up": "Đã tăng âm lượng.",
        "volume_down": "Đã giảm âm lượng.",
        "mute": "Đã bật/tắt tiếng.",
    }

    def __init__(self, config: MediaToolConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def perform(self, action: str) -> str:
        if not self._config.enabled:
            raise AppToolError("Công cụ điều khiển media đang bị tắt trong config.")

        key_name = _normalise(action)
        if key_name not in self._KEYS:
            raise AppToolError(
                f"Hành động {action!r} không hợp lệ. Chọn một trong: {', '.join(self.ACTIONS)}."
            )

        try:
            import pyautogui
        except Exception as exc:  # pragma: no cover - needs a desktop session
            raise AppToolError(f"Không dùng được PyAutoGUI: {exc}") from exc

        repeats = self._config.volume_step if key_name in ("volume_up", "volume_down") else 1
        try:
            for _ in range(repeats):
                pyautogui.press(self._KEYS[key_name])
        except Exception as exc:
            raise AppToolError(f"Không gửi được phím media: {exc}") from exc

        log.info("Media key %s x%d", key_name, repeats)
        return self._LABELS[key_name]
