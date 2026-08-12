"""browser-use integration (mvp.md Task 11) - optional and off by default.

browser-use is not a plain function tool: it is a second agent with its own LLM
loop that reads the DOM and decides what to click. We point it at the same LM Studio
server (OpenAI-compatible endpoint) so the whole stack stays local, run it in its own
event loop on a worker thread, and hard-cap it with a timeout so a stuck browser can
never freeze the voice pipeline.

Install separately::

    pip install -r requirements-browser.txt
    playwright install chromium

Then set ``tools.browser_use.enabled: true`` in ``config.yaml``.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

from ..config import BrowserUseToolConfig, LlmConfig
from ..logging_setup import get_logger

log = get_logger("jarvis.tools.browser")

INSTALL_HINT = (
    "browser-use chưa được cài. Chạy: pip install -r requirements-browser.txt "
    "&& playwright install chromium"
)


class BrowserToolError(RuntimeError):
    """browser-use missing, disabled, timed out or failed."""


def _import_chat_model():
    """browser-use moved its OpenAI chat wrapper around between releases."""
    try:
        from browser_use.llm import ChatOpenAI  # browser-use >= 0.3

        return ChatOpenAI
    except Exception:  # noqa: BLE001 - fall through to the older location
        pass
    try:
        from langchain_openai import ChatOpenAI  # browser-use <= 0.2

        return ChatOpenAI
    except Exception as exc:  # noqa: BLE001
        raise BrowserToolError(INSTALL_HINT) from exc


class BrowserAgentTool:
    """Runs one browser-use task at a time, synchronously, with a timeout."""

    def __init__(self, config: BrowserUseToolConfig, llm_config: LlmConfig) -> None:
        self._config = config
        self._llm_config = llm_config
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def available(self) -> bool:
        try:
            import browser_use  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def _model_name(self) -> str:
        return self._config.model or self._llm_config.model

    def _base_url(self) -> str:
        return f"http://{self._llm_config.api_host}/v1"

    # -- execution ---------------------------------------------------------------
    def run(self, task: str) -> str:
        task = (task or "").strip()
        if not task:
            raise BrowserToolError("Nội dung tác vụ trình duyệt đang trống.")
        if not self._config.enabled:
            raise BrowserToolError(
                "Công cụ browser-use đang tắt. Bật tools.browser_use.enabled trong config.yaml."
            )
        if not self._lock.acquire(blocking=False):
            raise BrowserToolError("Đang có một tác vụ trình duyệt khác chạy, hãy thử lại sau.")

        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-use") as pool:
                future = pool.submit(self._run_blocking, task)
                try:
                    return future.result(timeout=self._config.timeout_s)
                except FutureTimeout as exc:
                    raise BrowserToolError(
                        f"Tác vụ trình duyệt vượt quá {self._config.timeout_s:.0f} giây và đã bị bỏ."
                    ) from exc
        finally:
            self._lock.release()

    def _run_blocking(self, task: str) -> str:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._run_async(task))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    async def _run_async(self, task: str) -> str:
        try:
            from browser_use import Agent
        except Exception as exc:  # noqa: BLE001
            raise BrowserToolError(INSTALL_HINT) from exc

        chat_cls = _import_chat_model()
        llm = chat_cls(
            model=self._model_name(),
            base_url=self._base_url(),
            api_key="lm-studio",  # LM Studio ignores the key but the client requires one
            temperature=0.2,
        )

        log.info("browser-use bắt đầu: %s", task)
        agent_kwargs: dict[str, Any] = {"task": task, "llm": llm}
        # ``use_vision`` is expensive on a 3B model with 4 GB VRAM; opt-in only.
        agent = _construct_agent(Agent, agent_kwargs, use_vision=self._config.use_vision)

        try:
            history = await agent.run(max_steps=self._config.max_steps)
        except Exception as exc:  # noqa: BLE001 - browser-use raises many types
            raise BrowserToolError(f"browser-use lỗi: {exc}") from exc
        finally:
            await _close_agent(agent)

        return _summarise(history)


def _construct_agent(agent_cls: Any, kwargs: dict[str, Any], *, use_vision: bool) -> Any:
    """Construct the Agent, tolerating signature differences across releases."""
    try:
        return agent_cls(**kwargs, use_vision=use_vision)
    except TypeError:
        return agent_cls(**kwargs)


async def _close_agent(agent: Any) -> None:
    for method_name in ("close", "stop"):
        method = getattr(agent, method_name, None)
        if method is None:
            continue
        try:
            result = method()
            if asyncio.iscoroutine(result):
                await result
            return
        except Exception:  # noqa: BLE001 - cleanup must not mask the real error
            log.debug("Không đóng được browser-use agent bằng %s()", method_name, exc_info=True)


def _summarise(history: Any) -> str:
    """Pull a short, speakable result out of browser-use's history object."""
    for attribute in ("final_result", "extracted_content"):
        getter = getattr(history, attribute, None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # noqa: BLE001
                continue
            if value:
                text = value if isinstance(value, str) else str(value)
                return text.strip()[:1500]
    text = str(history).strip()
    return text[:1500] if text else "Tác vụ trình duyệt đã chạy xong."
