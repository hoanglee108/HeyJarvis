"""LM Studio connectivity layer.

Wraps the ``lmstudio-python`` SDK in a small, mockable surface:

* :meth:`LLMClient.chat` - plain text in, text out (no tools).
* :meth:`LLMClient.act`  - the SDK's agentic loop; the SDK does the tool
  dispatching for us, we only collect the final assistant text.

Anything that means "LM Studio is not reachable / model not loaded" is normalised
into :class:`LlmUnavailableError` so the pipeline can speak a friendly fallback.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .config import LlmConfig
from .logging_setup import get_logger

log = get_logger("jarvis.llm")

ToolFn = Callable[..., Any]


class LlmError(RuntimeError):
    """Generic LLM failure."""


class LlmUnavailableError(LlmError):
    """LM Studio is offline, the server is not started, or the model is missing."""


def _extract_text(message: Any) -> str:
    """Pull plain text out of an SDK message object (or dict)."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, (list, tuple)):
        for part in content:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                parts.append(str(text))
    return "".join(parts)


def _role_of(message: Any) -> str:
    role = getattr(message, "role", None)
    if role is None and isinstance(message, dict):
        role = message.get("role")
    return str(role or "")


class LLMClient:
    """Thin, thread-safe façade over one LM Studio model handle."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._client: Any = None
        self._model: Any = None
        self._lock = threading.RLock()
        #: Rolling conversation memory as ``(role, text)`` pairs.
        self._history: list[tuple[str, str]] = []

    # -- connection --------------------------------------------------------------
    def connect(self) -> None:
        """Open the websocket session and resolve the model handle (idempotent)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import lmstudio as lms
            except ImportError as exc:  # pragma: no cover
                raise LlmError("Chưa cài lmstudio SDK: pip install -r requirements.txt") from exc

            lms.set_sync_api_timeout(self._config.request_timeout_s)
            try:
                client = lms.Client(self._config.api_host)
                model = client.llm.model(
                    self._config.model,
                    config={"contextLength": self._config.context_length},
                )
            except Exception as exc:
                self._close_quietly()
                raise self._as_unavailable(exc) from exc

            self._client = client
            self._model = model
            log.info(
                "Đã kết nối LM Studio %s, model=%s",
                self._config.api_host,
                self._config.model,
            )

    def close(self) -> None:
        with self._lock:
            self._close_quietly()

    def _close_quietly(self) -> None:
        client, self._client = self._client, None
        self._model = None
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover
                log.debug("Lỗi khi đóng LM Studio client", exc_info=True)

    def _as_unavailable(self, exc: Exception) -> LlmError:
        message = str(exc) or exc.__class__.__name__
        lowered = message.lower()
        if "not found" in lowered or "no model" in lowered:
            return LlmUnavailableError(
                f"LM Studio không có model {self._config.model!r}. "
                "Kiểm tra `lms ls` và sửa `llm.model` trong config.yaml. Chi tiết: " + message
            )
        return LlmUnavailableError(
            f"Không kết nối được LM Studio tại {self._config.api_host}. "
            "Hãy mở LM Studio > Developer > Start Server. Chi tiết: " + message
        )

    def ping(self) -> str:
        """Sanity-check the connection; returns a short description of the model."""
        self.connect()
        assert self._model is not None
        try:
            info = self._model.get_info()
        except Exception as exc:
            raise self._as_unavailable(exc) from exc
        identifier = getattr(info, "identifier", None) or getattr(info, "model_key", "?")
        context = getattr(info, "context_length", None) or getattr(info, "contextLength", "?")
        return f"{identifier} (context {context})"

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

    def _build_chat(self, prompt: str) -> Any:
        import lmstudio as lms

        chat = lms.Chat(self._config.system_prompt)
        for role, text in self.history:
            if role == "user":
                chat.add_user_message(text)
            else:
                chat.add_assistant_response(text)
        chat.add_user_message(prompt)
        return chat

    def _prediction_config(self) -> dict[str, Any]:
        return {
            "temperature": self._config.temperature,
            "maxTokens": self._config.max_tokens,
        }

    # -- inference ---------------------------------------------------------------
    def chat(self, prompt: str, *, remember: bool = True) -> str:
        """Single-turn (plus history) completion without tools."""
        self.connect()
        assert self._model is not None
        chat_history = self._build_chat(prompt)
        try:
            result = self._model.respond(chat_history, config=self._prediction_config())
        except Exception as exc:
            raise self._wrap_prediction_error(exc) from exc

        answer = (getattr(result, "content", "") or "").strip()
        if remember and answer:
            self._remember(prompt, answer)
        log.debug("LLM chat -> %r", answer)
        return answer

    def act(
        self,
        prompt: str,
        tools: Sequence[ToolFn] | Iterable[ToolFn],
        *,
        remember: bool = True,
        on_round: Callable[[int], None] | None = None,
    ) -> str:
        """Run the SDK tool-calling loop and return the final spoken answer.

        ``ActResult`` does not carry the assistant text, so we harvest it from the
        ``on_message`` stream and keep the last assistant turn.
        """
        tool_list = list(tools)
        if not tool_list:
            return self.chat(prompt, remember=remember)

        self.connect()
        assert self._model is not None
        chat_history = self._build_chat(prompt)

        assistant_chunks: list[str] = []
        tool_calls: list[str] = []

        def _on_message(message: Any) -> None:
            role = _role_of(message)
            text = _extract_text(message)
            if role == "assistant":
                if text.strip():
                    assistant_chunks.append(text.strip())
            elif role == "tool":
                tool_calls.append(text[:400])
                log.info("Tool result: %s", text[:400])

        try:
            self._model.act(
                chat_history,
                tool_list,
                max_prediction_rounds=self._config.max_tool_rounds,
                config=self._prediction_config(),
                on_message=_on_message,
                on_round_start=on_round,
                handle_invalid_tool_request=self._handle_invalid_tool_request,
            )
        except Exception as exc:
            raise self._wrap_prediction_error(exc) from exc

        answer = assistant_chunks[-1].strip() if assistant_chunks else ""
        if not answer and tool_calls:
            # The model called a tool but never summarised; surface the raw result
            # instead of going silent.
            answer = tool_calls[-1].strip()
        if remember and answer:
            self._remember(prompt, answer)
        log.debug("LLM act -> %r (%d tool message)", answer, len(tool_calls))
        return answer

    @staticmethod
    def _handle_invalid_tool_request(exc: Exception, request: Any) -> str:
        """Feed malformed tool calls back to the model instead of aborting the turn."""
        name = getattr(request, "name", "?") if request is not None else "?"
        log.warning("Tool call không hợp lệ (%s): %s", name, exc)
        return (
            f"Lời gọi công cụ không hợp lệ ({exc}). "
            "Hãy gọi lại đúng tên công cụ và tham số, hoặc trả lời trực tiếp bằng lời."
        )

    def _wrap_prediction_error(self, exc: Exception) -> LlmError:
        try:
            import lmstudio as lms
        except ImportError:  # pragma: no cover
            return LlmError(str(exc))

        if isinstance(exc, (lms.LMStudioWebsocketError, lms.LMStudioModelNotFoundError)):
            self.close()
            return self._as_unavailable(exc)
        if isinstance(exc, lms.LMStudioTimeoutError):
            return LlmError(
                f"LM Studio phản hồi quá chậm (>{self._config.request_timeout_s:.0f}s). "
                "Thử model nhỏ hơn hoặc giảm max_tokens."
            )
        if isinstance(exc, (ConnectionError, OSError)):
            self.close()
            return self._as_unavailable(exc)
        if isinstance(exc, lms.LMStudioError):
            return LlmError(f"LM Studio lỗi: {exc}")
        return LlmError(f"Lỗi khi gọi LLM: {exc}")

    # -- context manager ---------------------------------------------------------
    def __enter__(self) -> LLMClient:
        self.connect()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
