"""LM Studio connectivity layer (OpenAI-compatible HTTP).

Why plain HTTP instead of the ``lmstudio-python`` SDK
-----------------------------------------------------
``nvidia/nemotron-3-nano-4b`` is a *reasoning* model: its chat template always opens
an ``<think>`` block, and the reasoning is written in English even when the answer is
Vietnamese. Two consequences drive the design here:

1. The SDK's ``model.act()`` hands back one assistant message whose text is
   ``<reasoning>`` + an internal separator + ``<answer>``. Speaking that through TTS
   would read the entire English chain of thought out loud, and the separator is an
   undocumented internal token with a build-specific nonce. The REST API instead
   returns the reasoning in a *separate* ``reasoning_content`` field, so the spoken
   answer is clean by construction.
2. Owning the tool loop lets us cap the rounds, log every call, and force a final
   spoken answer when the model would otherwise keep calling tools forever.

Anything that means "LM Studio is not reachable / model not loaded" is normalised into
:class:`LlmUnavailableError` so the pipeline can speak a friendly fallback.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import LlmConfig
from .logging_setup import get_logger

log = get_logger("jarvis.llm")

#: Cap on how much of a tool result is fed back into the context.
MAX_TOOL_RESULT_CHARS = 1500
#: Cap on a raw tool result used as a last-resort spoken answer.
MAX_SPOKEN_FALLBACK_CHARS = 400

#: Sent when the model returns nothing at all, to break it out of that state.
ANSWER_NUDGE = (
    "Bạn vừa trả lời rỗng. Hãy trả lời ngay bằng một hoặc hai câu tiếng Việt, "
    "không gọi thêm công cụ, không giải thích quá trình suy luận."
)


class LlmError(RuntimeError):
    """Generic LLM failure."""


class LlmUnavailableError(LlmError):
    """LM Studio is offline, the server is not started, or the model is missing."""


# --------------------------------------------------------------------------------------
# Tool description handed to the model
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
class ToolSpec:
    """One callable exposed to the model, with an explicit JSON schema.

    An explicit schema (rather than one inferred from type hints) matters here: the
    description is what a 4B model actually reads, and we inline the live whitelist /
    alias catalogue into it.
    """

    name: str
    description: str
    handler: Callable[..., str]
    #: JSON-schema ``properties`` map for the arguments.
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": dict(self.properties),
                    "required": list(self.required),
                },
            },
        }


# --------------------------------------------------------------------------------------
# Reasoning / markup scrubbing
# --------------------------------------------------------------------------------------
#: LM Studio's SDK transport splices reasoning and answer together with this sentinel.
#: The trailing hex is build-specific, hence the pattern rather than a literal.
_REASONING_SEPARATOR = re.compile(
    r"__LM_STUDIO_INTERNAL_LSEP_SYNTHETIC_REASONING_END_[0-9A-Za-z]*__"
)
_CLOSED_THINK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_DANGLING_OPEN_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_LEADING_CLOSE_THINK = re.compile(r"\A.*?</think\s*>", re.DOTALL | re.IGNORECASE)
#: Nemotron's tool-call syntax. If the server's parser misses one it arrives as text;
#: reading raw XML out loud is worse than saying nothing.
_TOOL_MARKUP = re.compile(
    r"<(tool_call|tools|function|parameter)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE
)


#: A voice assistant must never read a URL out loud.
_URL = re.compile(r"\(?\b(?:https?://|www\.)\S+\)?", re.IGNORECASE)


def speakable(text: str, limit: int = MAX_SPOKEN_FALLBACK_CHARS) -> str:
    """Make a raw tool result safe to read aloud: no URLs, bounded length."""
    cleaned = " ".join(_URL.sub("", text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "…"


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought and stray tool markup from a spoken answer.

    LM Studio already returns Nemotron's reasoning in a separate field, so this is a
    second line of defence for the cases that field does not cover: a ``<think>`` block
    truncated by ``max_tokens``, a differently configured server, or a tool call the
    server's parser failed to extract.
    """
    if not text:
        return ""

    # The SDK-style sentinel: everything before it is reasoning.
    match = None
    for match in _REASONING_SEPARATOR.finditer(text):
        pass
    if match is not None:
        text = text[match.end() :]

    text = _CLOSED_THINK.sub(" ", text)
    # An unbalanced tag means the block was cut off at one end or the other.
    if re.search(r"</think\s*>", text, re.IGNORECASE):
        text = _LEADING_CLOSE_THINK.sub("", text)
    text = _DANGLING_OPEN_THINK.sub("", text)
    text = _TOOL_MARKUP.sub(" ", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------
class LLMClient:
    """Thin, thread-safe façade over one LM Studio model."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._session: Any = None
        self._connected = False
        self._lock = threading.RLock()
        #: Rolling conversation memory as ``(role, text)`` pairs.
        self._history: list[tuple[str, str]] = []

    # -- transport ---------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self._config.api_host}/v1"

    def _http(self) -> Any:
        if self._session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover
                raise LlmError("Chưa cài requests: pip install -r requirements.txt") from exc
            session = requests.Session()
            session.headers.update(
                {
                    "Content-Type": "application/json",
                    # LM Studio ignores the key; some proxies in front of it do not.
                    "Authorization": "Bearer lm-studio",
                }
            )
            self._session = session
        return self._session

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        """The single network seam; tests replace this instead of patching requests."""
        import requests

        url = f"{self.base_url}{path}"
        try:
            response = self._http().request(
                method,
                url,
                json=payload,
                timeout=self._config.request_timeout_s,
            )
        except requests.Timeout as exc:
            raise LlmError(
                f"LM Studio phản hồi quá chậm (>{self._config.request_timeout_s:.0f}s). "
                "Thử giảm llm.max_tokens hoặc dùng model nhỏ hơn."
            ) from exc
        except requests.RequestException as exc:
            raise self._as_unavailable(exc) from exc

        if response.status_code >= 400:
            raise self._as_http_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise LlmError(f"LM Studio trả về dữ liệu không phải JSON: {response.text[:200]}") from exc

    # -- errors ------------------------------------------------------------------
    def _as_unavailable(self, exc: Exception) -> LlmUnavailableError:
        return LlmUnavailableError(
            f"Không kết nối được LM Studio tại {self._config.api_host}. "
            "Hãy mở LM Studio > Developer > Start Server. Chi tiết: " + (str(exc) or type(exc).__name__)
        )

    def _as_http_error(self, response: Any) -> LlmError:
        body = (response.text or "")[:400]
        lowered = body.lower()
        model_missing = response.status_code == 404 or "no model" in lowered or "not found" in lowered
        if model_missing:
            return LlmUnavailableError(
                f"LM Studio không nạp được model {self._config.model!r}. "
                "Kiểm tra `lms ls` / `lms ps` và sửa `llm.model` trong config.yaml. "
                f"Chi tiết: {body}"
            )
        return LlmError(f"LM Studio trả lỗi HTTP {response.status_code}: {body}")

    # -- connection --------------------------------------------------------------
    def connect(self) -> None:
        """Verify the server answers and the configured model exists (idempotent)."""
        if self._connected:
            return
        with self._lock:
            if self._connected:
                return
            payload = self._request("GET", "/models")
            available = [
                str(entry.get("id"))
                for entry in (payload.get("data") or [])
                if entry.get("id")
            ]
            if available and self._config.model not in available:
                raise LlmUnavailableError(
                    f"LM Studio không có model {self._config.model!r}. "
                    f"Đang có: {', '.join(available)}. "
                    "Sửa `llm.model` trong config.yaml hoặc chạy "
                    f"`lms load {self._config.model}`."
                )
            self._connected = True
            log.info(
                "Đã kết nối LM Studio %s, model=%s", self._config.api_host, self._config.model
            )

    def close(self) -> None:
        with self._lock:
            session, self._session = self._session, None
            self._connected = False
            if session is not None:
                try:
                    session.close()
                except Exception:  # pragma: no cover - cleanup must not raise
                    log.debug("Lỗi khi đóng HTTP session", exc_info=True)

    def ping(self) -> str:
        """Sanity-check the connection; returns a short description of the model."""
        self.connect()
        payload = self._request("GET", "/models")
        for entry in payload.get("data") or []:
            if entry.get("id") == self._config.model:
                quant = entry.get("quantization") or entry.get("object") or "model"
                context = entry.get("max_context_length") or self._config.context_length
                return f"{self._config.model} ({quant}, context {context})"
        return f"{self._config.model} (context {self._config.context_length})"

    # -- history -----------------------------------------------------------------
    def reset_history(self) -> None:
        with self._lock:
            self._history.clear()

    @property
    def history(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._history)

    def _remember(self, user_text: str, assistant_text: str) -> None:
        with self._lock:
            self._history.append(("user", user_text))
            self._history.append(("assistant", assistant_text))
            limit = self._config.history_turns * 2
            if limit <= 0:
                self._history.clear()
            elif len(self._history) > limit:
                del self._history[: len(self._history) - limit]

    def _build_messages(self, prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._config.system_prompt}
        ]
        for role, text in self.history:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": prompt})
        return messages

    # -- one completion ----------------------------------------------------------
    def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = [spec.to_openai_schema() for spec in tools]
            payload["tool_choice"] = "auto"

        body = self._request("POST", "/chat/completions", payload)
        choices = body.get("choices") or []
        if not choices:
            raise LlmError(f"LM Studio không trả về choices nào: {str(body)[:200]}")
        message = choices[0].get("message") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            log.debug("Reasoning (%d ký tự, không đọc ra loa): %s", len(reasoning), reasoning[:300])
        return {
            "text": strip_reasoning(message.get("content") or ""),
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choices[0].get("finish_reason"),
        }

    def _complete_answer(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec] | None = None,
    ) -> dict[str, Any]:
        """Complete, retrying when the model produces nothing at all.

        Nemotron intermittently emits an empty ``<think></think>`` and stops after
        three tokens, with ``finish_reason: stop``. It is not a truncation and not an
        error, just a dead turn, and retrying the same request usually resolves it.
        """
        step = self._complete(messages, tools)
        for attempt in range(1, self._config.empty_reply_retries + 1):
            if step["text"] or step["tool_calls"]:
                return step
            log.warning("Model trả lời rỗng, thử lại lần %d", attempt)
            retry = messages if attempt == 1 else [*messages, {"role": "user", "content": ANSWER_NUDGE}]
            step = self._complete(retry, tools)
        return step

    # -- inference ---------------------------------------------------------------
    def chat(self, prompt: str, *, remember: bool = True) -> str:
        """Single-turn (plus history) completion without tools."""
        self.connect()
        answer = self._complete_answer(self._build_messages(prompt))["text"]
        if remember and answer:
            self._remember(prompt, answer)
        log.debug("LLM chat -> %r", answer)
        return answer

    def act(
        self,
        prompt: str,
        tools: Sequence[ToolSpec] | Iterable[ToolSpec],
        *,
        remember: bool = True,
        on_round: Callable[[int], None] | None = None,
    ) -> str:
        """Run the tool-calling loop and return the final spoken answer."""
        tool_list = list(tools)
        if not tool_list:
            return self.chat(prompt, remember=remember)

        self.connect()
        by_name = {spec.name: spec for spec in tool_list}
        messages = self._build_messages(prompt)
        last_tool_result = ""

        for round_index in range(1, self._config.max_tool_rounds + 1):
            if on_round is not None:
                on_round(round_index)
            step = self._complete_answer(messages, tool_list)
            calls = step["tool_calls"]

            if not calls:
                answer = step["text"]
                if not answer and last_tool_result:
                    # The model called a tool then went silent even after retrying;
                    # say the raw result rather than nothing at all, minus anything
                    # that must not be read aloud.
                    answer = speakable(last_tool_result)
                    log.warning("Model im lặng sau tool, đọc thẳng kết quả: %r", answer)
                if remember and answer:
                    self._remember(prompt, answer)
                log.debug("LLM act -> %r (vòng %d)", answer, round_index)
                return answer

            messages.append(
                {
                    "role": "assistant",
                    "content": step["text"],
                    "tool_calls": calls,
                }
            )
            for call in calls:
                name, result = self._dispatch(call, by_name)
                last_tool_result = result
                log.info("Tool %s -> %s", name, result[:200])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or name,
                        "content": result[:MAX_TOOL_RESULT_CHARS],
                    }
                )

        # Round budget exhausted: ask once more with tools withheld so the user
        # always gets a spoken answer instead of silence.
        log.warning("Hết %d vòng gọi tool, buộc trả lời bằng lời", self._config.max_tool_rounds)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Đừng gọi thêm công cụ nào nữa. Hãy trả lời ngắn gọn bằng tiếng Việt "
                    "dựa trên kết quả công cụ đã có."
                ),
            }
        )
        answer = self._complete_answer(messages)["text"] or speakable(last_tool_result)
        if remember and answer:
            self._remember(prompt, answer)
        return answer

    def _dispatch(self, call: dict[str, Any], by_name: dict[str, ToolSpec]) -> tuple[str, str]:
        """Run one tool call. Never raises: the error text goes back to the model."""
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        spec = by_name.get(name)
        if spec is None:
            log.warning("Model gọi tool không tồn tại: %r", name)
            return name or "?", (
                f"Lỗi: không có công cụ tên {name!r}. "
                f"Các công cụ khả dụng: {', '.join(sorted(by_name)) or '(trống)'}."
            )

        raw_args = function.get("arguments")
        try:
            arguments = self._parse_arguments(raw_args)
        except ValueError as exc:
            log.warning("Tham số tool %s không hợp lệ: %s", name, exc)
            return name, f"Lỗi: tham số không hợp lệ ({exc}). Hãy gọi lại đúng định dạng JSON."

        try:
            return name, str(spec.handler(**arguments))
        except TypeError as exc:
            # Wrong/extra keyword names: tell the model instead of aborting the turn.
            log.warning("Tool %s nhận sai tham số: %s", name, exc)
            allowed = ", ".join(spec.properties) or "(không có)"
            return name, f"Lỗi: sai tên tham số ({exc}). Các tham số hợp lệ: {allowed}."
        except Exception as exc:  # noqa: BLE001 - a tool must never break the loop
            log.exception("Tool %s lỗi không mong đợi", name)
            return name, f"Lỗi không mong đợi khi chạy {name}: {exc}"

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            raise ValueError(f"kiểu tham số không hỗ trợ: {type(raw).__name__}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"không phải JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tham số phải là một object JSON")
        return parsed

    # -- context manager ---------------------------------------------------------
    def __enter__(self) -> LLMClient:
        self.connect()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
