"""Windows system-tray indicator (mvp.md Task 7).

pystray must own the main thread on Windows, so :func:`run_with_tray` runs the tray
loop in the foreground and the voice pipeline on a worker thread. Icons are drawn at
runtime with Pillow, which keeps the repo free of binary assets.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import TrayConfig
from .logging_setup import get_logger
from .state import State, StateMachine, label_for

log = get_logger("jarvis.tray")

ICON_SIZE = 64

#: (fill, ring) RGB colours per state.
_COLOURS: dict[State, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    State.STARTING: ((120, 120, 130), (70, 70, 80)),
    State.IDLE: ((60, 130, 220), (25, 70, 130)),
    State.LISTENING: ((60, 200, 110), (25, 120, 60)),
    State.THINKING: ((240, 180, 60), (150, 105, 20)),
    State.SPEAKING: ((170, 110, 235), (95, 55, 150)),
    State.ERROR: ((225, 70, 70), (140, 30, 30)),
    State.STOPPED: ((90, 90, 95), (50, 50, 55)),
}


def make_icon_image(state: State, size: int = ICON_SIZE):
    """Draw a filled circle whose colour encodes the state."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill, ring = _COLOURS.get(state, _COLOURS[State.IDLE])
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=fill, outline=ring, width=max(2, size // 16))

    if state is State.LISTENING:
        # Vertical bar suggesting a microphone.
        bar_w = size // 8
        draw.rounded_rectangle(
            [size // 2 - bar_w // 2, size // 3, size // 2 + bar_w // 2, size * 2 // 3],
            radius=bar_w // 2,
            fill=(255, 255, 255),
        )
    elif state is State.THINKING:
        dot = size // 12
        for offset in (-1, 0, 1):
            cx = size // 2 + offset * size // 6
            draw.ellipse([cx - dot, size // 2 - dot, cx + dot, size // 2 + dot], fill=(60, 40, 0))
    elif state is State.SPEAKING:
        draw.polygon(
            [
                (size // 3, size // 3),
                (size // 3, size * 2 // 3),
                (size * 2 // 3, size // 2),
            ],
            fill=(255, 255, 255),
        )
    elif state is State.ERROR:
        pad = size // 3
        draw.line([pad, pad, size - pad, size - pad], fill=(255, 255, 255), width=max(3, size // 14))
        draw.line([size - pad, pad, pad, size - pad], fill=(255, 255, 255), width=max(3, size // 14))

    return image


class TrayIcon:
    """Thin adapter between :class:`StateMachine` and ``pystray.Icon``."""

    def __init__(
        self,
        config: TrayConfig,
        state: StateMachine,
        *,
        on_quit: Any = None,
        on_reset: Any = None,
    ) -> None:
        self._config = config
        self._state = state
        self._on_quit = on_quit
        self._on_reset = on_reset
        self._icon: Any = None
        self._ready = threading.Event()

    def _build(self) -> Any:
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem(
                lambda _item: self._status_text(),
                lambda *_args: None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Xoá ngữ cảnh hội thoại", self._handle_reset),
            pystray.MenuItem("Thoát Jarvis", self._handle_quit),
        )
        return pystray.Icon(
            "jarvis",
            icon=make_icon_image(self._state.state),
            title=self._title_text(),
            menu=menu,
        )

    def _status_text(self) -> str:
        detail = self._state.detail
        base = label_for(self._state.state)
        return f"{base} — {detail}" if detail else base

    def _title_text(self) -> str:
        return f"{self._config.title}: {self._status_text()}"[:127]  # Windows tooltip limit

    # -- observer ----------------------------------------------------------------
    def _on_state_change(self, state: State, _detail: str) -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = make_icon_image(state)
            icon.title = self._title_text()
        except Exception:  # pragma: no cover - tray backends are finicky
            log.debug("Không cập nhật được tray icon", exc_info=True)

    # -- menu actions ------------------------------------------------------------
    def _handle_quit(self, icon: Any, _item: Any = None) -> None:
        log.info("Thoát từ tray menu")
        if callable(self._on_quit):
            self._on_quit()
        try:
            icon.stop()
        except Exception:  # pragma: no cover
            log.debug("icon.stop() lỗi", exc_info=True)

    def _handle_reset(self, _icon: Any, _item: Any = None) -> None:
        if callable(self._on_reset):
            self._on_reset()

    # -- lifecycle ---------------------------------------------------------------
    def run(self) -> None:
        """Blocking tray loop (must be the main thread on Windows)."""
        self._icon = self._build()
        self._state.subscribe(self._on_state_change)
        self._ready.set()
        self._icon.run()

    def stop(self) -> None:
        icon, self._icon = self._icon, None
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # pragma: no cover
                log.debug("Không dừng được tray icon", exc_info=True)


def run_with_tray(pipeline: Any, config: TrayConfig, *, hands_free: bool | None = None) -> int:
    """Run the pipeline on a worker thread with a tray icon on the main thread."""
    worker_error: list[BaseException] = []

    def worker() -> None:
        try:
            pipeline.run(hands_free=hands_free)
        except BaseException as exc:  # noqa: BLE001 - surfaced after the tray exits
            worker_error.append(exc)
            log.exception("Pipeline dừng vì lỗi")
        finally:
            tray.stop()

    tray = TrayIcon(
        config,
        pipeline.state,
        on_quit=pipeline.request_stop,
        on_reset=pipeline.llm.reset_history,
    )

    thread = threading.Thread(target=worker, name="jarvis-pipeline", daemon=True)
    thread.start()
    try:
        tray.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.request_stop()
        thread.join(timeout=10)

    if worker_error:
        print(f"Jarvis dừng vì lỗi: {worker_error[0]}")
        return 1
    return 0
