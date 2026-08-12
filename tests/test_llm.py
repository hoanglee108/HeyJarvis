"""mvp.md Task 2: LM Studio wrapper behaviour with a mocked SDK."""

from __future__ import annotations

import types

import pytest

from jarvis.config import LlmConfig
from jarvis.llm import LLMClient, LlmError, LlmUnavailableError, _extract_text


class FakeModel:
    def __init__(self, reply: str = "Xin chào bạn.") -> None:
        self.reply = reply
        self.respond_calls: list[object] = []
        self.act_calls: list[tuple[object, list]] = []

    def respond(self, history, **_kwargs):  # noqa: ANN001, ANN202
        self.respond_calls.append(history)
        return types.SimpleNamespace(content=self.reply)

    def act(self, chat, tools, *, on_message=None, **_kwargs):  # noqa: ANN001, ANN202
        self.act_calls.append((chat, list(tools)))
        if on_message is not None:
            on_message(types.SimpleNamespace(role="tool", content="Đã mở thư mục Downloads."))
            on_message(types.SimpleNamespace(role="assistant", content=self.reply))
        return types.SimpleNamespace(rounds=2, total_time_seconds=1.0)

    def get_info(self):  # noqa: ANN202
        return types.SimpleNamespace(identifier="qwen2.5-3b-instruct", context_length=4096)


def attach(client: LLMClient, model: FakeModel) -> FakeModel:
    """Bypass connect() by injecting a fake model handle."""
    client._model = model  # noqa: SLF001 - test seam
    return model


@pytest.fixture
def config() -> LlmConfig:
    return LlmConfig(model="test-model", history_turns=2)


# -- text extraction ------------------------------------------------------------------
def test_extract_text_handles_plain_string() -> None:
    assert _extract_text(types.SimpleNamespace(content="xin chào")) == "xin chào"


def test_extract_text_handles_content_parts() -> None:
    message = types.SimpleNamespace(
        content=[types.SimpleNamespace(text="xin "), types.SimpleNamespace(text="chào")]
    )
    assert _extract_text(message) == "xin chào"


def test_extract_text_handles_dicts() -> None:
    assert _extract_text({"content": [{"text": "ok"}]}) == "ok"


def test_extract_text_of_empty_message() -> None:
    assert _extract_text(types.SimpleNamespace(content=None)) == ""


# -- chat -----------------------------------------------------------------------------
def test_chat_returns_content(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeModel("Chào bạn."))
    assert client.chat("xin chào") == "Chào bạn."


def test_chat_records_history(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeModel("Chào bạn."))
    client.chat("xin chào")
    assert client.history == [("user", "xin chào"), ("assistant", "Chào bạn.")]


def test_history_is_trimmed_to_configured_turns(config: LlmConfig) -> None:
    client = LLMClient(config)  # history_turns=2 -> 4 entries
    attach(client, FakeModel("ok"))
    for index in range(5):
        client.chat(f"câu {index}")
    assert len(client.history) == 4
    assert client.history[0] == ("user", "câu 3")


def test_reset_history_clears_memory(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeModel("ok"))
    client.chat("xin chào")
    client.reset_history()
    assert client.history == []


def test_remember_false_skips_history(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeModel("ok"))
    client.chat("xin chào", remember=False)
    assert client.history == []


# -- act ------------------------------------------------------------------------------
def test_act_returns_last_assistant_message(config: LlmConfig) -> None:
    client = LLMClient(config)
    model = attach(client, FakeModel("Đã mở thư mục Downloads cho bạn."))
    answer = client.act("mở thư mục downloads", [lambda: None])
    assert answer == "Đã mở thư mục Downloads cho bạn."
    assert len(model.act_calls) == 1


def test_act_without_tools_falls_back_to_chat(config: LlmConfig) -> None:
    client = LLMClient(config)
    model = attach(client, FakeModel("Chào bạn."))
    assert client.act("xin chào", []) == "Chào bạn."
    assert model.act_calls == []
    assert len(model.respond_calls) == 1


def test_act_falls_back_to_tool_output_when_model_stays_silent(config: LlmConfig) -> None:
    class SilentModel(FakeModel):
        def act(self, chat, tools, *, on_message=None, **_kwargs):  # noqa: ANN001, ANN202
            if on_message is not None:
                on_message(types.SimpleNamespace(role="tool", content="Pin còn 87 phần trăm"))
            return types.SimpleNamespace(rounds=1, total_time_seconds=0.5)

    client = LLMClient(config)
    attach(client, SilentModel())
    assert client.act("pin còn bao nhiêu", [lambda: None]) == "Pin còn 87 phần trăm"


# -- error mapping ---------------------------------------------------------------------
def test_connection_refused_becomes_unavailable(config: LlmConfig) -> None:
    class DeadModel(FakeModel):
        def respond(self, history, **_kwargs):  # noqa: ANN001, ANN202
            raise ConnectionRefusedError("No connection could be made")

    client = LLMClient(config)
    attach(client, DeadModel())
    with pytest.raises(LlmUnavailableError) as excinfo:
        client.chat("xin chào")
    assert "LM Studio" in str(excinfo.value)


def test_generic_failure_becomes_llm_error(config: LlmConfig) -> None:
    class BrokenModel(FakeModel):
        def respond(self, history, **_kwargs):  # noqa: ANN001, ANN202
            raise ValueError("something odd")

    client = LLMClient(config)
    attach(client, BrokenModel())
    with pytest.raises(LlmError):
        client.chat("xin chào")


def test_invalid_tool_request_handler_returns_guidance() -> None:
    message = LLMClient._handle_invalid_tool_request(  # noqa: SLF001
        ValueError("thiếu tham số"), types.SimpleNamespace(name="run_system_command")
    )
    assert "không hợp lệ" in message


def test_ping_describes_the_model(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeModel())
    assert "qwen2.5-3b-instruct" in client.ping()


def test_api_host_normalisation() -> None:
    assert LlmConfig(api_host="http://localhost:1234/").api_host == "localhost:1234"
