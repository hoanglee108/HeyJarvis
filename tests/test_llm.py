"""LM Studio wrapper behaviour against a faked OpenAI-compatible server.

The single network seam is ``LLMClient._request``, so these tests drive the real tool
loop, history handling and reasoning scrubbing without a server or a model.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.config import LlmConfig
from jarvis.llm import (
    ANSWER_NUDGE,
    MAX_TOOL_RESULT_CHARS,
    LLMClient,
    LlmError,
    LlmUnavailableError,
    ToolSpec,
    clamp_tool_result,
    speakable,
    strip_reasoning,
)

REASONING_SENTINEL = "__LM_STUDIO_INTERNAL_LSEP_SYNTHETIC_REASONING_END_f4e9a8d2c6b1__"


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------
def completion(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def tool_call(name: str, arguments: str = "{}", call_id: str = "call-1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeServer:
    """Replays scripted ``/chat/completions`` bodies and records what was sent."""

    def __init__(self, *responses: dict[str, Any], models: list[str] | None = None) -> None:
        self.responses = list(responses)
        self.models = models if models is not None else ["test-model"]
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __call__(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((method, path, payload))
        if path == "/models":
            return {"data": [{"id": name} for name in self.models]}
        if not self.responses:
            raise AssertionError("model was called more times than the test scripted")
        return self.responses.pop(0)

    @property
    def completions(self) -> list[dict[str, Any]]:
        return [payload for _m, path, payload in self.calls if path == "/chat/completions"]


def attach(client: LLMClient, server: FakeServer) -> FakeServer:
    client._request = server  # type: ignore[method-assign]  # noqa: SLF001 - test seam
    return server


@pytest.fixture
def config() -> LlmConfig:
    # Retries off by default so the scripted response counts stay readable; the retry
    # behaviour has its own tests below.
    return LlmConfig(
        model="test-model", history_turns=2, max_tool_rounds=3, empty_reply_retries=0
    )


@pytest.fixture
def retrying_config() -> LlmConfig:
    return LlmConfig(
        model="test-model", history_turns=2, max_tool_rounds=3, empty_reply_retries=2
    )


@pytest.fixture
def echo_tool() -> ToolSpec:
    return ToolSpec(
        name="get_current_datetime",
        description="Xem ngày giờ.",
        handler=lambda **_kwargs: "Bây giờ là 14 giờ 32 phút.",
    )


# --------------------------------------------------------------------------------------
# reasoning scrubbing - the reason this module talks HTTP instead of using the SDK
# --------------------------------------------------------------------------------------
def test_strip_reasoning_removes_a_closed_think_block() -> None:
    text = "<think>The user greets me in Vietnamese.</think>Xin chào bạn."
    assert strip_reasoning(text) == "Xin chào bạn."


def test_strip_reasoning_drops_a_think_block_left_unclosed_by_max_tokens() -> None:
    # Nemotron opens <think> immediately; a truncated answer has no closing tag.
    assert strip_reasoning("<think>We need to figure out whether") == ""


def test_strip_reasoning_handles_a_missing_opening_tag() -> None:
    assert strip_reasoning("reasoning text</think> Chào bạn.") == "Chào bạn."


def test_strip_reasoning_splits_on_the_lm_studio_sdk_sentinel() -> None:
    text = f"English chain of thought{REASONING_SENTINEL}Được rồi, mở YouTube cho bạn."
    assert strip_reasoning(text) == "Được rồi, mở YouTube cho bạn."


def test_strip_reasoning_removes_tool_markup_that_leaked_as_text() -> None:
    text = '<tools>{"name": "open_application"}</tools> Đã mở YouTube.'
    assert strip_reasoning(text) == "Đã mở YouTube."


def test_strip_reasoning_keeps_ordinary_text_intact() -> None:
    assert strip_reasoning("  Xin chào   bạn.  ") == "Xin chào bạn."


def test_reasoning_content_field_never_reaches_the_answer(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer(completion("Chào bạn.", reasoning="The user said hello.")))
    assert client.chat("xin chào") == "Chào bạn."


# --------------------------------------------------------------------------------------
# a spoken answer must never contain a URL
# --------------------------------------------------------------------------------------
def test_speakable_strips_urls() -> None:
    raw = "1. LM Studio (https://lmstudio.ai/) chạy model cục bộ"
    assert speakable(raw) == "1. LM Studio chạy model cục bộ"


# --------------------------------------------------------------------------------------
# an oversized tool result must keep its tail: that is where search_web's
# "synthesise this into one sentence" instruction lives
# --------------------------------------------------------------------------------------
def test_clamp_tool_result_leaves_short_results_untouched() -> None:
    assert clamp_tool_result("kết quả ngắn") == "kết quả ngắn"


def test_clamp_tool_result_keeps_both_ends() -> None:
    text = "ĐẦU " + ("x" * 5000) + " CUỐI: hãy tổng hợp thành một câu."
    clamped = clamp_tool_result(text, limit=500)
    assert len(clamped) <= 500
    assert clamped.startswith("ĐẦU ")
    assert clamped.endswith("CUỐI: hãy tổng hợp thành một câu.")
    assert "cắt bớt" in clamped


def test_tool_result_budget_fits_a_full_web_search_block() -> None:
    """Guards the coupling between MAX_TOOL_RESULT_CHARS and web_search.context_chars."""
    from jarvis.config import WebSearchToolConfig

    default_context = WebSearchToolConfig().context_chars
    assert MAX_TOOL_RESULT_CHARS >= default_context + 400, (
        "khối tư liệu cộng phần rào và dòng chỉ dẫn phải vừa trong ngân sách tool result"
    )


def test_speakable_strips_bare_www_links() -> None:
    assert speakable("Xem tại www.example.com nhé") == "Xem tại nhé"


def test_speakable_truncates_on_a_word_boundary() -> None:
    result = speakable("một hai ba bốn năm sáu bảy", limit=12)
    assert result.endswith("…")
    assert "bả" not in result


# --------------------------------------------------------------------------------------
# empty-reply retry: Nemotron intermittently emits <think></think> then stops
# --------------------------------------------------------------------------------------
def test_empty_reply_is_retried(retrying_config: LlmConfig) -> None:
    client = LLMClient(retrying_config)
    server = attach(client, FakeServer(completion(""), completion("Chào bạn.")))
    assert client.chat("xin chào") == "Chào bạn."
    assert len(server.completions) == 2


def test_first_retry_repeats_the_identical_request(retrying_config: LlmConfig) -> None:
    client = LLMClient(retrying_config)
    server = attach(client, FakeServer(completion(""), completion("Chào bạn.")))
    client.chat("xin chào")
    assert server.completions[0]["messages"] == server.completions[1]["messages"]


def test_second_retry_nudges_the_model(retrying_config: LlmConfig) -> None:
    client = LLMClient(retrying_config)
    server = attach(
        client, FakeServer(completion(""), completion(""), completion("Chào bạn."))
    )
    assert client.chat("xin chào") == "Chào bạn."
    assert server.completions[2]["messages"][-1] == {"role": "user", "content": ANSWER_NUDGE}


def test_retries_are_bounded(retrying_config: LlmConfig) -> None:
    client = LLMClient(retrying_config)
    server = attach(client, FakeServer(*[completion("") for _ in range(3)]))
    assert client.chat("xin chào") == ""
    assert len(server.completions) == 3  # 1 attempt + 2 retries, then give up


def test_a_tool_call_is_not_treated_as_an_empty_reply(
    retrying_config: LlmConfig, echo_tool: ToolSpec
) -> None:
    """Empty ``content`` plus a tool call is the normal shape of a tool round."""
    client = LLMClient(retrying_config)
    server = attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("get_current_datetime")]),
            completion("Bây giờ là 14 giờ 32 phút."),
        ),
    )
    client.act("mấy giờ rồi", [echo_tool])
    assert len(server.completions) == 2


def test_silent_model_after_a_search_does_not_read_urls_aloud(
    config: LlmConfig,
) -> None:
    search = ToolSpec(
        name="search_web",
        description="Tìm kiếm.",
        handler=lambda **_k: "1. LM Studio\n   Chạy model cục bộ\n   (https://lmstudio.ai/)",
    )
    client = LLMClient(config)
    attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("search_web", '{"query": "lm studio"}')]),
            completion(""),
        ),
    )
    answer = client.act("tìm thông tin về lm studio", [search])
    assert "LM Studio" in answer
    assert "http" not in answer and "lmstudio.ai" not in answer


# --------------------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------------------
def test_chat_returns_content(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer(completion("Chào bạn.")))
    assert client.chat("xin chào") == "Chào bạn."


def test_chat_sends_the_system_prompt_first(config: LlmConfig) -> None:
    client = LLMClient(config)
    server = attach(client, FakeServer(completion("ok")))
    client.chat("xin chào")
    messages = server.completions[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == config.system_prompt
    assert messages[-1] == {"role": "user", "content": "xin chào"}


def test_chat_without_tools_omits_the_tools_field(config: LlmConfig) -> None:
    client = LLMClient(config)
    server = attach(client, FakeServer(completion("ok")))
    client.chat("xin chào")
    assert "tools" not in server.completions[0]


def test_chat_records_history(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer(completion("Chào bạn.")))
    client.chat("xin chào")
    assert client.history == [("user", "xin chào"), ("assistant", "Chào bạn.")]


def test_history_is_trimmed_to_configured_turns(config: LlmConfig) -> None:
    client = LLMClient(config)  # history_turns=2 -> 4 entries
    attach(client, FakeServer(*[completion("ok") for _ in range(5)]))
    for index in range(5):
        client.chat(f"câu {index}")
    assert len(client.history) == 4
    assert client.history[0] == ("user", "câu 3")


def test_history_is_replayed_to_the_model(config: LlmConfig) -> None:
    client = LLMClient(config)
    server = attach(client, FakeServer(completion("một"), completion("hai")))
    client.chat("câu 1")
    client.chat("câu 2")
    roles = [message["role"] for message in server.completions[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_reset_history_clears_memory(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer(completion("ok")))
    client.chat("xin chào")
    client.reset_history()
    assert client.history == []


def test_remember_false_skips_history(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer(completion("ok")))
    client.chat("xin chào", remember=False)
    assert client.history == []


# --------------------------------------------------------------------------------------
# act / tool loop
# --------------------------------------------------------------------------------------
def test_act_without_tools_falls_back_to_chat(config: LlmConfig) -> None:
    client = LLMClient(config)
    server = attach(client, FakeServer(completion("Chào bạn.")))
    assert client.act("xin chào", []) == "Chào bạn."
    assert "tools" not in server.completions[0]


def test_act_advertises_the_tool_schema(config: LlmConfig, echo_tool: ToolSpec) -> None:
    client = LLMClient(config)
    server = attach(client, FakeServer(completion("Chào bạn.")))
    client.act("xin chào", [echo_tool])
    advertised = server.completions[0]["tools"]
    assert advertised[0]["function"]["name"] == "get_current_datetime"
    assert advertised[0]["function"]["parameters"]["type"] == "object"


def test_act_runs_the_tool_then_returns_the_final_answer(
    config: LlmConfig, echo_tool: ToolSpec
) -> None:
    client = LLMClient(config)
    server = attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("get_current_datetime")], finish_reason="tool_calls"),
            completion("Bây giờ là 14 giờ 32 phút."),
        ),
    )
    assert client.act("mấy giờ rồi", [echo_tool]) == "Bây giờ là 14 giờ 32 phút."

    # The tool result must be fed back with the matching id.
    second_round = server.completions[1]["messages"]
    assert second_round[-1]["role"] == "tool"
    assert second_round[-1]["tool_call_id"] == "call-1"
    assert "14 giờ 32 phút" in second_round[-1]["content"]


def test_act_passes_parsed_arguments_to_the_handler(config: LlmConfig) -> None:
    seen: dict[str, Any] = {}

    def handler(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "Đã mở YouTube."

    spec = ToolSpec(name="open_application", description="Mở app.", handler=handler)
    client = LLMClient(config)
    attach(
        client,
        FakeServer(
            completion(
                "",
                tool_calls=[tool_call("open_application", '{"app_name": "youtube"}')],
            ),
            completion("Đã mở YouTube cho bạn."),
        ),
    )
    assert client.act("mở youtube", [spec]) == "Đã mở YouTube cho bạn."
    assert seen == {"app_name": "youtube"}


def test_act_falls_back_to_the_tool_output_when_the_model_stays_silent(
    config: LlmConfig, echo_tool: ToolSpec
) -> None:
    client = LLMClient(config)
    attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("get_current_datetime")]),
            completion(""),  # model produced only reasoning, no answer
        ),
    )
    assert client.act("mấy giờ rồi", [echo_tool]) == "Bây giờ là 14 giờ 32 phút."


def test_unknown_tool_name_is_reported_back_to_the_model(
    config: LlmConfig, echo_tool: ToolSpec
) -> None:
    client = LLMClient(config)
    server = attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("khong_ton_tai")]),
            completion("Xin lỗi, tôi chưa làm được việc đó."),
        ),
    )
    client.act("làm gì đó", [echo_tool])
    fed_back = server.completions[1]["messages"][-1]["content"]
    assert "không có công cụ" in fed_back
    assert "get_current_datetime" in fed_back


def test_malformed_tool_arguments_are_reported_back_to_the_model(
    config: LlmConfig, echo_tool: ToolSpec
) -> None:
    client = LLMClient(config)
    server = attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("get_current_datetime", "{not json")]),
            completion("Tôi gặp lỗi khi gọi công cụ."),
        ),
    )
    client.act("mấy giờ rồi", [echo_tool])
    assert "không hợp lệ" in server.completions[1]["messages"][-1]["content"]


def test_a_raising_tool_does_not_break_the_loop(config: LlmConfig) -> None:
    def explode(**_kwargs: Any) -> str:
        raise RuntimeError("ổ đĩa bốc cháy")

    spec = ToolSpec(name="boom", description="Nổ.", handler=explode)
    client = LLMClient(config)
    server = attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("boom")]),
            completion("Xin lỗi, không thực hiện được."),
        ),
    )
    assert client.act("nổ đi", [spec]) == "Xin lỗi, không thực hiện được."
    assert "ổ đĩa bốc cháy" in server.completions[1]["messages"][-1]["content"]


def test_tool_loop_is_capped_and_still_produces_an_answer(
    config: LlmConfig, echo_tool: ToolSpec
) -> None:
    """A model stuck in a tool loop must not leave the user with silence."""
    client = LLMClient(config)  # max_tool_rounds=3
    server = attach(
        client,
        FakeServer(
            *[completion("", tool_calls=[tool_call("get_current_datetime")]) for _ in range(3)],
            completion("Bây giờ là 14 giờ 32 phút."),
        ),
    )
    assert client.act("mấy giờ rồi", [echo_tool]) == "Bây giờ là 14 giờ 32 phút."
    # 3 capped rounds + 1 forced answer, and the last one withholds the tools.
    assert len(server.completions) == 4
    assert "tools" not in server.completions[-1]


def test_act_records_history_using_the_final_answer(
    config: LlmConfig, echo_tool: ToolSpec
) -> None:
    client = LLMClient(config)
    attach(
        client,
        FakeServer(
            completion("", tool_calls=[tool_call("get_current_datetime")]),
            completion("Bây giờ là 14 giờ 32 phút."),
        ),
    )
    client.act("mấy giờ rồi", [echo_tool])
    assert client.history == [
        ("user", "mấy giờ rồi"),
        ("assistant", "Bây giờ là 14 giờ 32 phút."),
    ]


# --------------------------------------------------------------------------------------
# connection / error mapping
# --------------------------------------------------------------------------------------
def test_connect_rejects_a_model_the_server_does_not_have(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer(models=["nvidia/nemotron-3-nano-4b"]))
    with pytest.raises(LlmUnavailableError) as excinfo:
        client.connect()
    assert "test-model" in str(excinfo.value)
    assert "nvidia/nemotron-3-nano-4b" in str(excinfo.value)


def test_connect_is_idempotent(config: LlmConfig) -> None:
    client = LLMClient(config)
    server = attach(client, FakeServer())
    client.connect()
    client.connect()
    assert len(server.calls) == 1


def test_connection_refused_becomes_unavailable(config: LlmConfig) -> None:
    import requests

    client = LLMClient(config)

    def dead(*_args: object, **_kwargs: object) -> None:
        raise requests.ConnectionError("No connection could be made")

    client._http = lambda: _Raising(dead)  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(LlmUnavailableError) as excinfo:
        client.connect()
    assert "LM Studio" in str(excinfo.value)


def test_timeout_becomes_a_speed_hint(config: LlmConfig) -> None:
    import requests

    client = LLMClient(config)

    def slow(*_args: object, **_kwargs: object) -> None:
        raise requests.Timeout("timed out")

    client._http = lambda: _Raising(slow)  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(LlmError) as excinfo:
        client.connect()
    assert "quá chậm" in str(excinfo.value)


def test_http_404_points_at_the_model_key(config: LlmConfig) -> None:
    client = LLMClient(config)
    client._http = lambda: _Responding(404, "model not found")  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(LlmUnavailableError) as excinfo:
        client.connect()
    assert "lms ls" in str(excinfo.value)


def test_http_500_becomes_a_generic_llm_error(config: LlmConfig) -> None:
    client = LLMClient(config)
    client._http = lambda: _Responding(500, "internal boom")  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(LlmError) as excinfo:
        client.connect()
    assert "500" in str(excinfo.value)


def test_ping_describes_the_model(config: LlmConfig) -> None:
    client = LLMClient(config)
    attach(client, FakeServer())
    assert "test-model" in client.ping()


def test_api_host_normalisation() -> None:
    assert LlmConfig(api_host="http://localhost:1234/").api_host == "localhost:1234"


def test_base_url_is_openai_compatible() -> None:
    assert LLMClient(LlmConfig(api_host="localhost:1234")).base_url == "http://localhost:1234/v1"


# --------------------------------------------------------------------------------------
# tiny request/response doubles
# --------------------------------------------------------------------------------------
class _Raising:
    def __init__(self, raiser: object) -> None:
        self._raiser = raiser

    def request(self, *args: object, **kwargs: object) -> None:
        self._raiser(*args, **kwargs)  # type: ignore[operator]


class _Responding:
    def __init__(self, status_code: int, text: str) -> None:
        self._status_code = status_code
        self._text = text

    def request(self, *_args: object, **_kwargs: object) -> _Responding:
        return self

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def text(self) -> str:
        return self._text
