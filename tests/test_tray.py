"""mvp.md Task 7: tray icon rendering and state wiring (no real tray needed)."""

from __future__ import annotations

from typing import Any

from jarvis.config import TrayConfig
from jarvis.state import State, StateMachine
from jarvis.tray import TrayIcon, make_icon_image


def test_every_state_renders_a_distinct_icon() -> None:
    images = {}
    for state in State:
        image = make_icon_image(state, size=32)
        assert image.size == (32, 32)
        images[state] = image.tobytes()
    assert len(set(images.values())) == len(State), "mỗi trạng thái phải có icon riêng"


class FakeIcon:
    def __init__(self) -> None:
        self.icon: Any = None
        self.title = ""
        self.stopped = False

    def run(self) -> None:  # pragma: no cover - not used
        pass

    def stop(self) -> None:
        self.stopped = True


def make_tray(machine: StateMachine) -> tuple[TrayIcon, FakeIcon]:
    tray = TrayIcon(TrayConfig(), machine)
    fake = FakeIcon()
    tray._icon = fake  # noqa: SLF001 - inject instead of building a real tray
    machine.subscribe(tray._on_state_change)  # noqa: SLF001
    return tray, fake


def test_icon_and_tooltip_follow_the_state_machine() -> None:
    machine = StateMachine(State.IDLE)
    _tray, fake = make_tray(machine)

    machine.set(State.LISTENING)
    listening_icon = fake.icon
    assert "Đang nghe" in fake.title

    machine.set(State.THINKING, "đang gọi LLM")
    assert fake.icon is not listening_icon
    assert "Đang xử lý" in fake.title
    assert "đang gọi LLM" in fake.title


def test_tooltip_fits_the_windows_limit() -> None:
    machine = StateMachine(State.IDLE)
    _tray, fake = make_tray(machine)
    machine.set(State.THINKING, "x" * 500)
    assert len(fake.title) <= 127


def test_quit_menu_item_invokes_the_callback_and_stops_the_icon() -> None:
    machine = StateMachine(State.IDLE)
    quit_calls: list[bool] = []
    tray = TrayIcon(TrayConfig(), machine, on_quit=lambda: quit_calls.append(True))
    fake = FakeIcon()

    tray._handle_quit(fake)  # noqa: SLF001

    assert quit_calls == [True]
    assert fake.stopped


def test_reset_menu_item_invokes_the_callback() -> None:
    machine = StateMachine(State.IDLE)
    reset_calls: list[bool] = []
    tray = TrayIcon(TrayConfig(), machine, on_reset=lambda: reset_calls.append(True))
    tray._handle_reset(FakeIcon())  # noqa: SLF001
    assert reset_calls == [True]


def test_state_change_without_an_icon_is_a_no_op() -> None:
    tray = TrayIcon(TrayConfig(), StateMachine(State.IDLE))
    tray._on_state_change(State.SPEAKING, "")  # noqa: SLF001 - must not raise
