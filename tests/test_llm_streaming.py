"""priority.md P1-1: server-sent-event parsing for streamed completions.

The seam here is ``LLMClient._http``, one level lower than ``_request`` used by
``test_llm.py``, because streaming bypasses the JSON-body path entirely.

What these tests protect:

* reasoning deltas never reach the sink (speaking English chain-of-thought out loud is
  the worst possible failure of a Vietnamese voice assistant),
* tool-call fragments are reassembled correctly - the tool loop is the most fragile part
  of the stack and streaming must not break it,
* a server that refuses streaming degrades to the ordinary path instead of failing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jarvis.config import LlmConfig
from jarvis.llm import LLMClient, ToolSpec, _StreamStartError

from .test_llm import FakeServer, attach, completion


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------
def sse(**delta: Any) -> str:
    """One ``data:`` line carrying a delta."""
    finish_reason = delta.pop("finish_reason", None)
    return "data: " + json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    )


def sse_tool(
    index: int = 0,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> str:
    function: dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    call: dict[str, Any] = {"index": index, "function": function}
    if call_id is not None:
        call["id"] = call_id
    return sse(tool_calls=[call])


class FakeSseResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200, text: str = "") -> None:
        self.lines = lines
        self.status_code = status_code
        self.text = text
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False) -> Any:  # noqa: ARG002
        yield from self.lines

    def close(self) -> None:
        self.closed = True


class FakeHttp:
    def __init__(self, response: FakeSseResponse) -> None:
        self.response = response
        self.payloads: list[dict[str, Any]] = []
        self.streamed: list[bool] = []

    def post(
        self,
        _url: str,
        json: dict[str, Any] | None = None,  # noqa: A002
        timeout: float | None = None,  # noqa: ARG002
        stream: bool = False,
    ) -> FakeSseResponse:
        self.payloads.append(json or {})
        self.streamed.append(stream)
        return self.response


class RecordingSink:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.discards = 0

    def push(self, chunk: str) -> None:
        self.chunks.append(chunk)

    def discard_pending(self) -> None:
        self.discards += 1

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def make_client(lines: list[str], **config_overrides: Any) -> tuple[LLMClient, FakeHttp]:
    client = LLMClient(LlmConfig(model="test-model", **config_overrides))
    http = FakeHttp(FakeSseResponse(lines))
    client._http = lambda: http  # type: ignore[method-assign] # noqa: SLF001
    client._connected = True  # noqa: SLF001
    return client, http


# --------------------------------------------------------------------------------------
# content streaming
# --------------------------------------------------------------------------------------
def test_content_deltas_are_forwarded_in_order() -> None:
    client, _http = make_client(
        [
            sse(content="Bây giờ "),
            sse(content="là 9 giờ "),
            sse(content="sáng.", finish_reason="stop"),
            "data: [DONE]",
        ]
    )
    sink = RecordingSink()

    step = client._complete([{"role": "user", "content": "mấy giờ"}], None, sink)  # noqa: SLF001

    assert sink.chunks == ["Bây giờ ", "là 9 giờ ", "sáng."]
    assert step["text"] == "Bây giờ là 9 giờ sáng."
    assert step["finish_reason"] == "stop"
    assert step["tool_calls"] == []


def test_streaming_flag_is_set_on_the_request() -> None:
    client, http = make_client([sse(content="Xong."), "data: [DONE]"])
    client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    assert http.streamed == [True]
    assert http.payloads[0]["stream"] is True


def test_no_sink_means_no_streaming() -> None:
    """``jarvis chat`` and the tool-schema calls have nobody to speak to."""
    client = LLMClient(LlmConfig(model="test-model"))
    server = attach(client, FakeServer(completion("Xong.")))
    client._connected = True  # noqa: SLF001

    step = client._complete([{"role": "user", "content": "x"}], None, None)  # noqa: SLF001

    assert step["text"] == "Xong."
    assert server.completions[0]["stream"] is False


def test_stream_disabled_in_config_uses_the_plain_path() -> None:
    client = LLMClient(LlmConfig(model="test-model", stream=False))
    server = attach(client, FakeServer(completion("Xong.")))
    client._connected = True  # noqa: SLF001

    step = client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001

    assert step["text"] == "Xong."
    assert server.completions[0]["stream"] is False


def test_response_is_always_closed() -> None:
    client, http = make_client([sse(content="Xong."), "data: [DONE]"])
    client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    assert http.response.closed


# --------------------------------------------------------------------------------------
# reasoning must not be spoken
# --------------------------------------------------------------------------------------
def test_reasoning_deltas_are_never_forwarded() -> None:
    client, _http = make_client(
        [
            sse(reasoning_content="The user is asking about the time. "),
            sse(reasoning_content="I will answer in Vietnamese."),
            sse(content="Bây giờ là 9 giờ."),
            "data: [DONE]",
        ]
    )
    sink = RecordingSink()

    step = client._complete([{"role": "user", "content": "mấy giờ"}], None, sink)  # noqa: SLF001

    assert sink.text == "Bây giờ là 9 giờ."
    assert "user is asking" not in sink.text
    assert step["text"] == "Bây giờ là 9 giờ."


def test_think_markup_in_content_is_scrubbed_from_the_returned_text() -> None:
    client, _http = make_client(
        [
            sse(content="<think>reasoning</think>"),
            sse(content="Bây giờ là 9 giờ."),
            "data: [DONE]",
        ]
    )
    step = client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    assert step["text"] == "Bây giờ là 9 giờ."


# --------------------------------------------------------------------------------------
# tool calls
# --------------------------------------------------------------------------------------
def test_tool_call_fragments_are_reassembled() -> None:
    client, _http = make_client(
        [
            sse_tool(call_id="call-1", name="get_current_datetime", arguments=""),
            sse_tool(arguments='{"time'),
            sse_tool(arguments='zone": "local"}'),
            sse(finish_reason="tool_calls"),
            "data: [DONE]",
        ]
    )

    step = client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001

    assert len(step["tool_calls"]) == 1
    call = step["tool_calls"][0]
    assert call["id"] == "call-1"
    assert call["function"]["name"] == "get_current_datetime"
    assert json.loads(call["function"]["arguments"]) == {"timezone": "local"}


def test_parallel_tool_calls_keep_their_own_arguments() -> None:
    client, _http = make_client(
        [
            sse_tool(0, call_id="a", name="tool_a", arguments='{"x":'),
            sse_tool(1, call_id="b", name="tool_b", arguments='{"y":'),
            sse_tool(0, arguments="1}"),
            sse_tool(1, arguments="2}"),
            "data: [DONE]",
        ]
    )

    step = client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001

    names = [call["function"]["name"] for call in step["tool_calls"]]
    arguments = [json.loads(call["function"]["arguments"]) for call in step["tool_calls"]]
    assert names == ["tool_a", "tool_b"]
    assert arguments == [{"x": 1}, {"y": 2}]


def test_a_tool_call_stops_forwarding_and_discards_the_preamble() -> None:
    """Text written alongside a tool call belongs to the call, not to the answer."""
    client, _http = make_client(
        [
            sse(content="Để tôi kiểm tra"),
            sse_tool(call_id="call-1", name="get_current_datetime", arguments="{}"),
            sse(content=" và trả lời bạn sau."),
            "data: [DONE]",
        ]
    )
    sink = RecordingSink()

    step = client._complete([{"role": "user", "content": "x"}], None, sink)  # noqa: SLF001

    assert sink.chunks == ["Để tôi kiểm tra"], "phần sau tool call không được đọc"
    assert sink.discards == 1
    assert step["tool_calls"][0]["function"]["name"] == "get_current_datetime"


def test_tool_calls_without_a_name_are_dropped() -> None:
    client, _http = make_client([sse_tool(arguments="{}"), "data: [DONE]"])
    step = client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    assert step["tool_calls"] == []


# --------------------------------------------------------------------------------------
# malformed input and failures
# --------------------------------------------------------------------------------------
def test_unparseable_lines_are_skipped() -> None:
    client, _http = make_client(
        [
            ": keep-alive comment",
            "data: {not json}",
            "",
            sse(content="Vẫn chạy bình thường."),
            "data: [DONE]",
            sse(content="sau DONE thì bỏ qua"),
        ]
    )
    sink = RecordingSink()

    step = client._complete([{"role": "user", "content": "x"}], None, sink)  # noqa: SLF001

    assert step["text"] == "Vẫn chạy bình thường."
    assert "sau DONE" not in sink.text


def test_events_without_choices_are_ignored() -> None:
    client, _http = make_client(
        ['data: {"id":"x","object":"chat.completion.chunk"}', sse(content="Xong rồi."), "data: [DONE]"]
    )
    step = client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    assert step["text"] == "Xong rồi."


def test_http_error_before_any_token_falls_back_to_the_plain_path() -> None:
    """A server that cannot stream must still be usable."""
    client = LLMClient(LlmConfig(model="test-model"))
    http = FakeHttp(FakeSseResponse([], status_code=400, text="stream not supported"))
    client._http = lambda: http  # type: ignore[method-assign] # noqa: SLF001
    server = attach(client, FakeServer(completion("Trả lời không stream.")))
    client._connected = True  # noqa: SLF001

    sink = RecordingSink()
    step = client._complete([{"role": "user", "content": "x"}], None, sink)  # noqa: SLF001

    assert step["text"] == "Trả lời không stream."
    assert sink.chunks == [], "không được đọc gì trước khi fallback"
    assert client._stream_supported is False  # noqa: SLF001
    assert server.completions[-1]["stream"] is False


def test_fallback_is_remembered_for_the_rest_of_the_session() -> None:
    client = LLMClient(LlmConfig(model="test-model"))
    http = FakeHttp(FakeSseResponse([], status_code=400, text="nope"))
    client._http = lambda: http  # type: ignore[method-assign] # noqa: SLF001
    server = attach(client, FakeServer(completion("một"), completion("hai")))
    client._connected = True  # noqa: SLF001

    client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    client._complete([{"role": "user", "content": "y"}], None, RecordingSink())  # noqa: SLF001

    assert len(http.payloads) == 1, "chỉ thử stream một lần rồi thôi"


def test_missing_model_error_is_not_swallowed_by_the_fallback() -> None:
    """A 404 means the model is gone; retrying without streaming would not help."""
    client = LLMClient(LlmConfig(model="test-model"))
    http = FakeHttp(FakeSseResponse([], status_code=404, text="no model loaded"))
    client._http = lambda: http  # type: ignore[method-assign] # noqa: SLF001
    client._connected = True  # noqa: SLF001

    with pytest.raises(Exception) as excinfo:
        client._complete([{"role": "user", "content": "x"}], None, RecordingSink())  # noqa: SLF001
    assert not isinstance(excinfo.value, _StreamStartError)


# --------------------------------------------------------------------------------------
# through the tool loop
# --------------------------------------------------------------------------------------
def test_act_streams_the_answer_and_returns_the_full_text() -> None:
    client, _http = make_client(
        [sse(content="Bây giờ là 9 giờ "), sse(content="sáng nhé bạn."), "data: [DONE]"]
    )
    sink = RecordingSink()
    spec = ToolSpec(name="noop", description="", handler=lambda: "")

    answer = client.act("mấy giờ rồi", [spec], sink=sink)

    assert answer == "Bây giờ là 9 giờ sáng nhé bạn."
    assert sink.text == "Bây giờ là 9 giờ sáng nhé bạn."
