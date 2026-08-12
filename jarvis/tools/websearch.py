"""Free web search for the LLM (mvp.md Task 10).

Two backends, no API keys:

* ``duckduckgo`` - scrapes the no-JavaScript HTML endpoint ``html.duckduckgo.com``.
* ``searxng``    - queries a self-hosted SearxNG instance's JSON API.

Results are returned as compact numbered snippets; the LLM summarises them in its
final spoken answer.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass

from ..config import WebSearchToolConfig
from ..logging_setup import get_logger

log = get_logger("jarvis.tools.websearch")

DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
MAX_SNIPPET_CHARS = 320


class WebSearchError(RuntimeError):
    """The search request failed or returned nothing usable."""


@dataclass(slots=True)
class SearchResult:
    title: str
    snippet: str
    url: str

    def render(self, index: int) -> str:
        parts = [f"{index}. {self.title}"]
        if self.snippet:
            parts.append(f"   {self.snippet}")
        if self.url:
            parts.append(f"   ({self.url})")
        return "\n".join(parts)


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return " ".join(text.split())


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo HTML wraps results in ``/l/?uddg=<encoded>``."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        query = urllib.parse.parse_qs(parsed.query)
        target = query.get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


def parse_duckduckgo_html(markup: str, max_results: int) -> list[SearchResult]:
    """Extract results from DuckDuckGo's HTML endpoint.

    Kept as a module-level pure function so it can be unit-tested against a saved
    HTML fixture without network access.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise WebSearchError("Chưa cài beautifulsoup4.") from exc

    soup = BeautifulSoup(markup, "lxml")
    results: list[SearchResult] = []

    for block in soup.select("div.result, div.web-result"):
        if block.select_one(".result--ad") is not None:
            continue
        link = block.select_one("a.result__a")
        if link is None:
            continue
        title = _clean(link.get_text())
        url = _unwrap_ddg_url(link.get("href", ""))
        snippet_node = block.select_one(".result__snippet")
        snippet = _clean(snippet_node.get_text()) if snippet_node else ""
        if not title:
            continue
        results.append(SearchResult(title, snippet[:MAX_SNIPPET_CHARS], url))
        if len(results) >= max_results:
            break

    if not results:
        # Fallback for layout changes: any anchor that looks like a result link.
        for link in soup.select("a.result__a")[:max_results]:
            title = _clean(link.get_text())
            if title:
                results.append(SearchResult(title, "", _unwrap_ddg_url(link.get("href", ""))))

    return results


def parse_searxng_json(payload: dict, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in (payload.get("results") or [])[:max_results]:
        title = _clean(str(item.get("title", "")))
        if not title:
            continue
        results.append(
            SearchResult(
                title,
                _clean(str(item.get("content", "")))[:MAX_SNIPPET_CHARS],
                str(item.get("url", "")),
            )
        )
    return results


class WebSearch:
    def __init__(self, config: WebSearchToolConfig) -> None:
        self._config = config
        self._session = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def _http(self):
        if self._session is None:
            import requests

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": self._config.user_agent,
                    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.6",
                }
            )
            self._session = session
        return self._session

    # -- public API --------------------------------------------------------------
    def search(self, query: str) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            raise WebSearchError("Câu truy vấn tìm kiếm đang trống.")
        if not self._config.enabled:
            raise WebSearchError("Công cụ tìm kiếm web đang bị tắt trong config.")

        if self._config.backend == "searxng":
            return self._search_searxng(query)
        return self._search_duckduckgo(query)

    def search_as_text(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return f"Không tìm thấy kết quả nào cho {query!r}."
        header = f"Kết quả tìm kiếm cho {query!r}:"
        body = "\n".join(result.render(index) for index, result in enumerate(results, start=1))
        return f"{header}\n{body}"

    # -- backends ----------------------------------------------------------------
    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        import requests

        data = {"q": query, "kl": self._config.region}
        try:
            response = self._http().post(
                DDG_ENDPOINT, data=data, timeout=self._config.timeout_s
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebSearchError(
                f"Không kết nối được DuckDuckGo: {exc}. Kiểm tra mạng."
            ) from exc

        results = parse_duckduckgo_html(response.text, self._config.max_results)
        log.info("DuckDuckGo %r -> %d kết quả", query, len(results))
        return results

    def _search_searxng(self, query: str) -> list[SearchResult]:
        import requests

        base = (self._config.searxng_url or "").rstrip("/")
        if not base:
            raise WebSearchError(
                "backend=searxng nhưng chưa đặt tools.web_search.searxng_url trong config."
            )
        params = {"q": query, "format": "json", "language": "vi"}
        try:
            response = self._http().get(
                f"{base}/search", params=params, timeout=self._config.timeout_s
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise WebSearchError(f"Không kết nối được SearxNG ({base}): {exc}") from exc
        except ValueError as exc:
            raise WebSearchError(
                "SearxNG không trả về JSON. Bật `search.formats: [json]` trong settings.yml."
            ) from exc

        results = parse_searxng_json(payload, self._config.max_results)
        log.info("SearxNG %r -> %d kết quả", query, len(results))
        return results
