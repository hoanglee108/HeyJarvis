"""mvp.md Task 9: app alias resolution and media keys."""

from __future__ import annotations

import sys
import types

import pytest

from jarvis.config import AppAliasSpec, AppsToolConfig, MediaToolConfig
from jarvis.tools import AppLauncher, AppToolError, MediaController


def make_launcher() -> AppLauncher:
    aliases = {
        "spotify": AppAliasSpec(kind="uri", target="spotify:", description="Spotify"),
        "facebook": AppAliasSpec(kind="url", target="https://facebook.com"),
        "may_tinh": AppAliasSpec(kind="uri", target="calculator:", description="Máy tính"),
        "notepad": AppAliasSpec(kind="exe", target="notepad.exe"),
    }
    return AppLauncher(AppsToolConfig(enabled=True, aliases=aliases))


# -- resolution -----------------------------------------------------------------------
def test_exact_alias_resolves() -> None:
    key, spec = make_launcher().resolve("spotify")
    assert key == "spotify"
    assert spec.kind == "uri"


def test_alias_lookup_ignores_case_and_spacing() -> None:
    key, _ = make_launcher().resolve("  SpOtIfY ")
    assert key == "spotify"


def test_vietnamese_diacritics_are_folded() -> None:
    """STT gives 'máy tính', config key is 'may_tinh'."""
    key, _ = make_launcher().resolve("Máy tính")
    assert key == "may_tinh"


def test_close_spelling_still_resolves() -> None:
    key, _ = make_launcher().resolve("spotifi")
    assert key == "spotify"


def test_unknown_alias_lists_the_options() -> None:
    with pytest.raises(AppToolError) as excinfo:
        make_launcher().resolve("photoshop")
    message = str(excinfo.value)
    assert "spotify" in message and "config.yaml" in message


def test_disabled_tool_refuses() -> None:
    launcher = AppLauncher(AppsToolConfig(enabled=False, aliases={}))
    with pytest.raises(AppToolError, match="đang bị tắt"):
        launcher.open("spotify")
    assert launcher.enabled is False


def test_url_alias_opens_in_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        "jarvis.tools.apps.webbrowser.open",
        lambda url, new=0: opened.append(url) or True,  # noqa: ARG005
    )
    message = make_launcher().open("facebook")
    assert opened == ["https://facebook.com"]
    assert "Đã mở" in message


def test_exe_alias_spawns_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_popen(argv, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(argv)
        return types.SimpleNamespace(pid=1)

    monkeypatch.setattr("jarvis.tools.apps.subprocess.Popen", fake_popen)
    make_launcher().open("notepad")
    assert calls == [["notepad.exe"]]


def test_missing_executable_gives_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_argv, **_kwargs):  # noqa: ANN001, ANN202
        raise FileNotFoundError("nope")

    monkeypatch.setattr("jarvis.tools.apps.subprocess.Popen", boom)
    with pytest.raises(AppToolError, match="Không tìm thấy"):
        make_launcher().open("notepad")


# -- media ----------------------------------------------------------------------------
class FakePyAutoGui(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("pyautogui")
        self.pressed: list[str] = []

    def press(self, key: str) -> None:
        self.pressed.append(key)


@pytest.fixture
def fake_gui(monkeypatch: pytest.MonkeyPatch) -> FakePyAutoGui:
    module = FakePyAutoGui()
    monkeypatch.setitem(sys.modules, "pyautogui", module)
    return module


def test_play_pause_sends_one_key(fake_gui: FakePyAutoGui) -> None:
    controller = MediaController(MediaToolConfig(enabled=True))
    message = controller.perform("play_pause")
    assert fake_gui.pressed == ["playpause"]
    assert "nhạc" in message


def test_volume_up_repeats_by_configured_step(fake_gui: FakePyAutoGui) -> None:
    controller = MediaController(MediaToolConfig(enabled=True, volume_step=3))
    controller.perform("volume_up")
    assert fake_gui.pressed == ["volumeup"] * 3


def test_action_names_tolerate_spaces_and_accents(fake_gui: FakePyAutoGui) -> None:
    MediaController(MediaToolConfig(enabled=True)).perform("Next")
    assert fake_gui.pressed == ["nexttrack"]


def test_invalid_action_is_rejected(fake_gui: FakePyAutoGui) -> None:
    controller = MediaController(MediaToolConfig(enabled=True))
    with pytest.raises(AppToolError, match="không hợp lệ"):
        controller.perform("format_disk")
    assert fake_gui.pressed == []


def test_disabled_media_tool_refuses(fake_gui: FakePyAutoGui) -> None:
    controller = MediaController(MediaToolConfig(enabled=False))
    with pytest.raises(AppToolError, match="đang bị tắt"):
        controller.perform("play_pause")
