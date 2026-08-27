"""The deterministic intent router that runs before the LLM.

These are the highest-value tests in the suite: every case here is a command that the
reasoning model either got wrong or answered 10-20 s too slowly.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from jarvis.config import JarvisConfig
from jarvis.intents import IntentRouter, fold
from jarvis.tools import ToolBox


@pytest.fixture
def box(config_copy: JarvisConfig) -> ToolBox:
    """A real ToolBox with only the side effects stubbed out."""
    toolbox = ToolBox(config_copy)
    toolbox.apps.open = lambda alias: f"[app:{alias}]"  # type: ignore[method-assign]
    toolbox.music.play = lambda song: f"[song:{song}]"  # type: ignore[method-assign]
    toolbox.media.perform = lambda action: f"[media:{action}]"  # type: ignore[method-assign]
    toolbox.clock._now = lambda: datetime(2026, 8, 25, 14, 32)  # noqa: SLF001
    return toolbox


@pytest.fixture
def router(box: ToolBox) -> IntentRouter:
    return IntentRouter(box)


# -- folding ---------------------------------------------------------------------------
def test_fold_strips_vietnamese_diacritics() -> None:
    assert fold("Máy tính") == "may tinh"
    assert fold("Đường") == "duong"


# -- open app --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("utterance", "alias"),
    [
        ("mở youtube", "youtube"),
        ("mở youtube giúp tôi", "youtube"),
        ("jarvis mở spotify nhé", "spotify"),
        ("bật notepad đi", "notepad"),
        ("khởi động vscode", "vscode"),
        ("mở ứng dụng máy tính", "máy tính"),
        ("mở trang facebook", "facebook"),
        ("mở nhạc", "nhạc"),
    ],
)
def test_open_app_is_handled_without_the_llm(
    router: IntentRouter, utterance: str, alias: str
) -> None:
    match = router.route(utterance)
    assert match is not None
    assert match.tool == "open_application"
    assert match.reply == f"[app:{alias}]"


def test_unknown_alias_falls_through_to_the_llm(router: IntentRouter) -> None:
    """The LLM may still have a better idea (a whitelisted folder, a search, ...)."""
    assert router.route("mở photoshop") is None


# -- play song -------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("utterance", "title"),
    [
        # The live model turned this one into song_name="Em": it read "của ngày hôm
        # qua" as "of yesterday" instead of as part of the title.
        ("phát bài Em của ngày hôm qua", "Em của ngày hôm qua"),
        ("mở bài hát Chúng ta của hiện tại giúp mình", "Chúng ta của hiện tại"),
        ("nghe ca khúc Nơi này có anh", "Nơi này có anh"),
        ("phát nhạc Sơn Tùng MTP", "Sơn Tùng MTP"),
        ("bật bài Hãy trao cho anh nhé", "Hãy trao cho anh"),
    ],
)
def test_song_title_is_preserved_verbatim(
    router: IntentRouter, utterance: str, title: str
) -> None:
    match = router.route(utterance)
    assert match is not None
    assert match.tool == "play_song"
    assert match.reply == f"[song:{title}]"


def test_play_without_a_title_is_an_app_launch_not_a_song(router: IntentRouter) -> None:
    match = router.route("mở nhạc")
    assert match is not None
    assert match.tool == "open_application"


# -- media -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("tăng âm lượng", "volume_up"),
        ("to lên", "volume_up"),
        ("giảm âm lượng", "volume_down"),
        ("nhỏ tiếng", "volume_down"),
        ("tắt tiếng", "mute"),
        ("bài tiếp theo", "next"),
        ("bài trước", "previous"),
        ("tạm dừng", "play_pause"),
    ],
)
def test_media_keys_are_handled_without_the_llm(
    router: IntentRouter, utterance: str, action: str
) -> None:
    match = router.route(utterance)
    assert match is not None
    assert match.tool == "control_media"
    assert match.reply == f"[media:{action}]"


# -- date / time -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "utterance",
    [
        "bây giờ là mấy giờ rồi",
        "mấy giờ rồi",
        "hôm nay là ngày mấy",
        "hôm nay thứ mấy",
        "cho tôi biết ngày giờ hiện tại",
    ],
)
def test_datetime_questions_never_reach_the_llm(router: IntentRouter, utterance: str) -> None:
    """A model with a 2024-era training cut-off states a confidently wrong date."""
    match = router.route(utterance)
    assert match is not None
    assert match.tool == "get_current_datetime"
    assert "14 giờ 32 phút" in match.reply
    assert "Thứ Ba, ngày 25 tháng 8 năm 2026" in match.reply


# -- fall-through ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "utterance",
    [
        "xin chào",
        "tìm thông tin về LM Studio",
        "thời tiết Hà Nội hôm nay thế nào",
        "kể cho tôi một câu chuyện",
        "pin còn bao nhiêu phần trăm",
        "",
        "   ",
    ],
)
def test_open_ended_requests_go_to_the_llm(router: IntentRouter, utterance: str) -> None:
    assert router.route(utterance) is None


# -- disabled tools --------------------------------------------------------------------
def test_router_skips_disabled_tools(box: ToolBox) -> None:
    box.config.tools.clock.enabled = False
    box.config.tools.apps.enabled = False
    box.config.tools.music.enabled = False
    box.config.tools.media.enabled = False
    router = IntentRouter(box)
    for utterance in ("mấy giờ rồi", "mở youtube", "phát bài Em của ngày hôm qua", "tắt tiếng"):
        assert router.route(utterance) is None


# -- usage tracking --------------------------------------------------------------------
def test_routed_calls_are_recorded_like_llm_tool_calls(box: ToolBox) -> None:
    box.reset_usage()
    IntentRouter(box).route("mở youtube")
    assert box.last_used == ["open_application"]
