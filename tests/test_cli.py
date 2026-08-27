"""CLI wiring. ``jarvis chat`` must behave like one voice turn, router included."""

from __future__ import annotations

import pytest

from jarvis import cli, llm


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any use of the LLM in these tests is a bug, so make it fail loudly."""

    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("the CLI reached LM Studio for a command the router owns")

    monkeypatch.setattr(llm.LLMClient, "act", boom)
    monkeypatch.setattr(llm.LLMClient, "chat", boom)


def test_chat_answers_a_datetime_question_without_the_llm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["chat", "bây giờ mấy giờ rồi"]) == 0
    assert "Bây giờ là" in capsys.readouterr().out


def test_chat_routes_an_open_app_command_without_the_llm(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.tools.apps import AppLauncher

    monkeypatch.setattr(AppLauncher, "open", lambda _self, alias: f"Đã mở {alias}.")
    assert cli.main(["chat", "mở youtube giúp tôi"]) == 0
    assert "Đã mở youtube." in capsys.readouterr().out


def test_no_tools_flag_skips_the_router_and_goes_straight_to_the_model() -> None:
    """``--no-tools`` means "just the model", so the router must not intercept."""
    with pytest.raises(AssertionError, match="reached LM Studio"):
        cli.main(["chat", "--no-tools", "bây giờ mấy giờ rồi"])


def test_parser_exposes_every_documented_command() -> None:
    parser = cli.build_parser()
    actions = [
        action for action in parser._actions if getattr(action, "choices", None)  # noqa: SLF001
    ]
    commands = next(a.choices for a in actions if a.dest == "command")
    for name in (
        "run",
        "listen",
        "devices",
        "doctor",
        "chat",
        "transcribe",
        "speak",
        "record",
        "shell",
        "open",
        "media",
        "search",
        "browse",
    ):
        assert name in commands
