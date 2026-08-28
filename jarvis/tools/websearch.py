"""Web search for the LLM (mvp.md Task 10).

Four backends:

* ``brave``      - the official Brave Search API. Default. Needs a subscription token,
  but it is an API rather than a scrape, so it cannot be throttled into an anti-bot
  interstitial the way the DuckDuckGo scraper is. It also returns up to five
  query-relevant excerpts per result (``extra_snippets``), which is material the other
  backends can only get by opening the page.
* ``duckduckgo`` - scrapes the no-JavaScript HTML endpoint ``html.duckduckgo.com``.
  Keyless fallback, needs nothing beyond ``requests``.
* ``ddgs``       - the optional ``ddgs`` package, a metasearch client that queries
  several engines at once. Worth it because a single engine will rate-limit a scraper
  eventually; when DuckDuckGo answered HTTP 202 for hours, ddgs still returned results
  from Bing, Google, Brave and Yandex. Install with ``requirements-search.txt``.
* ``searxng``    - queries a self-hosted SearxNG instance's JSON API.

Output shape
------------
This is a voice assistant, so the tool does **not** return a ranked list of links.
:meth:`WebSearch.answer_context` opens the top results, extracts their body text,
strips every URL, and returns one block of prose for the LLM to condense into a
sentence or two. Snippets alone are page *descriptions* and rarely carry the figure
the user actually asked for, which is why the pages get read.

:meth:`WebSearch.search` still returns structured results *with* URLs, because
:mod:`jarvis.tools.music` needs the video link. Only the LLM-facing path drops them.

Security
--------
Two boundaries are crossed here, both documented in the security checklist:

* **SSRF.** Result URLs are attacker-influenceable and get fetched automatically, so
  :func:`is_fetchable_url` rejects anything that is not public http(s) - notably
  loopback, which is where LM Studio itself listens.
* **Prompt injection (LLM01).** Page text lands in the model's context and may contain
  "ignore previous instructions" style payloads. That cannot be filtered reliably, so
  the material is fenced and labelled as quoted data. The real boundary stays where it
  already is: the shell whitelist and regex-validated arguments.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import WebSearchToolConfig
from ..logging_setup import get_logger

log = get_logger("jarvis.tools.websearch")

DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
#: Documented ceiling for Brave's ``count`` parameter.
BRAVE_MAX_COUNT = 20
MAX_SNIPPET_CHARS = 320

DDGS_INSTALL_HINT = (
    "backend=ddgs nhưng chưa cài package `ddgs`. "
    "Chạy: pip install -r requirements-search.txt"
)
BRAVE_KEY_HINT = (
    "backend=brave nhưng chưa có API key. Lấy key miễn phí tại "
    "api-dashboard.search.brave.com rồi đặt biến môi trường BRAVE_API_KEY, "
    "hoặc ghi tools.web_search.brave_api_key vào config.local.yaml "
    "(file này đã được git-ignore). Không đặt key vào config.yaml."
)
#: The exact message ddgs uses when a search matched nothing at all, as opposed to
#: when it wraps an engine failure in the same exception type.
_DDGS_NO_RESULTS = "No results found."

#: DuckDuckGo answers a throttled scrape with HTTP 202 and an "anomaly" interstitial
#: instead of an error status, so ``raise_for_status`` lets it through and the parser
#: quietly yields zero results. Detect it explicitly to avoid reporting "no results".
_DDG_THROTTLE_MARKERS = ("anomaly-modal", "anomaly.js", "/anomaly")


def is_duckduckgo_throttled(status_code: int, markup: str) -> bool:
    """True when DuckDuckGo served its anti-bot page rather than search results."""
    if status_code == 200:
        return False
    lowered = (markup or "").lower()
    return any(marker in lowered for marker in _DDG_THROTTLE_MARKERS)


class WebSearchError(RuntimeError):
    """The search request failed or returned nothing usable."""


@dataclass(slots=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    #: Additional query-relevant excerpts from the same page. Only the Brave backend
    #: fills this (``extra_snippets``); it is the API-side equivalent of what
    #: :meth:`WebSearch.read_page` does by hand, minus the fetch and its SSRF surface.
    extras: list[str] = field(default_factory=list)

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


#: Anything a listener would have to spell out letter by letter.
_URL_RE = re.compile(r"\(?\b(?:https?://|www\.)\S+\)?", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
#: A bare host such as "vnexpress.net" or "sjc.com.vn". The TLD list is deliberately
#: closed: a permissive ``\.\w{2,}`` would eat decimals like "1.5" and prices.
_BARE_HOST_RE = re.compile(
    r"\b[\w-]{2,}(?:\.[\w-]{2,})*\.(?:com|net|org|vn|edu|gov|info|biz|io|dev|me|tv|co|xyz)"
    r"(?:\.[a-z]{2})?\b",
    re.IGNORECASE,
)


def strip_urls(text: str) -> str:
    """Remove links, e-mails and bare hostnames from text meant to be spoken.

    The system prompt already forbids reading links aloud, but a prompt is not a
    guarantee: keeping them out of the material in the first place is what actually
    stops a 4B model from reciting ``https://www.sjc.com.vn/gia-vang-online``.
    """
    without = _URL_RE.sub(" ", text or "")
    without = _EMAIL_RE.sub(" ", without)
    without = _BARE_HOST_RE.sub(" ", without)
    # Punctuation left stranded by the removals ("Theo , giá vàng…").
    without = re.sub(r"\s+([,.;:!?])", r"\1", without)
    without = re.sub(r"(^|[.!?])\s*,\s*", r"\1 ", without)
    return " ".join(without.split())


def is_fetchable_url(url: str) -> bool:
    """True only for a public http(s) URL that is safe to fetch automatically.

    Result URLs come from a search engine, i.e. from outside the trust boundary, and
    are opened without the user reviewing them. Loopback matters most: LM Studio is
    listening on ``127.0.0.1``, so a poisoned result pointing there would make Jarvis
    fetch its own model server. Link-local also covers the cloud metadata endpoint.
    """
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip()
    if not host:
        return False

    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        # Unresolvable: refuse rather than hand it to requests and hope.
        return False

    for info in infos:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:  # pragma: no cover - getaddrinfo always yields literals
            return False
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            log.warning("Bỏ qua URL trỏ vào địa chỉ nội bộ: %s (%s)", url, address)
            return False
    return True


#: Tags whose text is chrome, not content.
_BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "iframe",
)


def extract_main_text(markup: str, limit: int, keywords: Sequence[str] = ()) -> str:
    """Pull the readable body text out of an HTML page.

    Pure function so it can be tested against a saved page without network access.
    Paragraph-level tags are preferred over ``soup.get_text()`` on the whole document,
    which would splice menu labels into the middle of sentences.

    ``keywords`` (from :func:`query_keywords`) pushes paragraphs that do not mention
    the query to the back, which keeps a category page's unrelated teasers out of the
    material. Omit it to get plain document order.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise WebSearchError("Chưa cài beautifulsoup4.") from exc

    soup = BeautifulSoup(markup or "", "lxml")
    for tag in soup(list(_BOILERPLATE_TAGS)):
        tag.decompose()

    # Pick the *biggest* content container, not the first one. A listing page holds one
    # <article> per teaser, so ``select_one`` there returns a single unrelated card and
    # throws the rest of the page away.
    root = max(
        soup.select("article, main, [role=main]") or [],
        key=lambda node: len(node.get_text(" ")),
        default=None,
    ) or (soup.body or soup)

    chunks: list[str] = []
    seen: set[str] = set()
    # Gather more than the budget so the relevance pass below has something to choose
    # from, but stay bounded: a forum thread can hold thousands of paragraphs.
    harvest_cap = limit * 4
    for node in root.select("h1, h2, h3, p, li, td, dd, blockquote"):
        chunk = _clean(node.get_text(" "))
        # Drop nav crumbs and dedupe: list pages repeat the same teaser text.
        if len(chunk) < 25 or chunk in seen:
            continue
        seen.add(chunk)
        chunks.append(chunk)
        if sum(len(item) for item in chunks) >= harvest_cap:
            break

    if not chunks:  # Pages that render everything in bare divs.
        chunks = [_clean(root.get_text(" "))]

    selected = _rank_by_relevance(chunks, keywords, limit)
    if not selected:
        return ""
    return _truncate_words(" ".join(selected), limit)


#: Words too common in a Vietnamese question to say anything about a paragraph.
_QUERY_STOPWORDS = frozenset(
    """
    là gì của và cho với các những một hôm nay bây giờ thế nào sao bao nhiêu tôi bạn
    hãy cái này đó ở đâu khi để về có không được thì mà ra vào đi tin tức thông
    """.split()
)


def query_keywords(query: str) -> list[str]:
    """Content words of a query, used to tell relevant paragraphs from filler."""
    words = re.findall(r"\w+", (query or "").lower(), re.UNICODE)
    return [word for word in words if len(word) >= 2 and word not in _QUERY_STOPWORDS]


def _mentions_any(text: str, keywords: Sequence[str]) -> bool:
    """True when ``text`` shares a content word with the query, or there are none."""
    if not keywords:
        return True
    lowered = text.lower()
    return any(word in lowered for word in keywords)


def _rank_by_relevance(chunks: list[str], keywords: Sequence[str], limit: int) -> list[str]:
    """Keep the paragraphs that mention the query, in document order.

    Category and homepage URLs are common in search results, and their body is a list
    of unrelated teasers: asking for the gold price once yielded a paragraph about a
    pickleball player. Feeding that to the model produces a confidently off-topic
    answer, so paragraphs that share no content word with the query go last.

    Returns nothing when the page mentions no content word of the query at all. That
    is deliberate: handing a 4B model an article about something else, labelled as
    material about the question, gets a confident and wrong spoken answer. Contributing
    nothing lets the caller fall back to the search snippets, which are on topic by
    construction.
    """
    if not keywords:
        return chunks

    matched: list[str] = []
    rest: list[str] = []
    for chunk in chunks:
        lowered = chunk.lower()
        (matched if any(word in lowered for word in keywords) else rest).append(chunk)

    if not matched:
        log.debug("Trang không nhắc tới từ nào trong %s, bỏ qua", list(keywords))
        return []

    # Top up with unrelated paragraphs, but only whole ones. A truncated off-topic
    # fragment spends budget the matches could have used and can still mislead the
    # model, so it is never worth splicing on the end.
    room = limit - sum(len(chunk) for chunk in matched) - max(0, len(matched) - 1)
    for chunk in rest:
        if len(chunk) + 1 > room:
            break
        matched.append(chunk)
        room -= len(chunk) + 1
    return matched


def _truncate_words(text: str, limit: int) -> str:
    """Cut to at most ``limit`` characters on a word boundary.

    The ellipsis is counted inside the budget, not added on top of it, so callers can
    sum the lengths of several chunks and trust the total.
    """
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    head = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return (head or text[: limit - 1]) + "…"


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


def parse_brave_json(payload: dict, max_results: int) -> list[SearchResult]:
    """Extract results from the Brave Search API's ``web.results`` array.

    Brave marks the matched terms with ``<strong>`` inside ``description``, so the
    text goes through :func:`_clean` like every other backend's does.

    ``extra_snippets`` is kept whole rather than folded into ``snippet``: the excerpts
    are long, and truncating them to ``MAX_SNIPPET_CHARS`` would throw away exactly
    the sentences that carry the figures.
    """
    results: list[SearchResult] = []
    for item in ((payload.get("web") or {}).get("results") or [])[:max_results]:
        title = _clean(str(item.get("title", "")))
        if not title:
            continue
        extras = [
            cleaned
            for cleaned in (_clean(str(snippet)) for snippet in item.get("extra_snippets") or [])
            if cleaned
        ]
        results.append(
            SearchResult(
                title,
                _clean(str(item.get("description", "")))[:MAX_SNIPPET_CHARS],
                str(item.get("url", "")),
                extras,
            )
        )
    return results


def brave_error_detail(body: str) -> str:
    """Human-readable reason out of Brave's ``ErrorResponse`` envelope.

    Brave answers a bad request with a nested JSON error rather than a plain message,
    and the useful part (which parameter it rejected) is buried in ``meta.errors``.
    Surfacing it is what turns "HTTP 422" into something actionable.
    """
    import json

    fallback = " ".join((body or "").split())[:200]

    try:
        payload = json.loads(body or "")
    except (ValueError, TypeError):
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None

    if not isinstance(error, dict):
        return fallback or "(không rõ lý do)"

    parts = [str(error.get("detail") or error.get("code") or "").strip()]
    meta = error.get("meta")
    if isinstance(meta, dict):
        for entry in (meta.get("errors") or [])[:3]:
            if not isinstance(entry, dict):
                continue
            location = ".".join(str(item) for item in entry.get("loc") or [])
            message = str(entry.get("msg") or "").strip()
            if location or message:
                parts.append(f"{location}: {message}" if location else message)

    detail = " | ".join(part for part in parts if part)[:400]
    return detail or fallback or "(không rõ lý do)"


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

        if self._config.backend == "brave":
            return self._search_brave(query)
        if self._config.backend == "searxng":
            return self._search_searxng(query)
        if self._config.backend == "ddgs":
            return self._search_ddgs(query)
        return self._search_duckduckgo(query)

    def read_page(self, url: str, keywords: Sequence[str] = ()) -> str:
        """Fetch one result page and return its body text, or ``""`` on any problem.

        Never raises: a page that is slow, gated, binary or simply gone must not cost
        the user their answer, so the caller just moves on to the next result.
        """
        import requests

        if not is_fetchable_url(url):
            return ""
        try:
            with self._http().get(
                url, timeout=self._config.page_timeout_s, allow_redirects=True
            ) as response:
                response.raise_for_status()
                content_type = (response.headers.get("content-type") or "").lower()
                if "html" not in content_type and "text/plain" not in content_type:
                    log.debug("Bỏ qua %s: content-type %s", url, content_type or "(trống)")
                    return ""
                # A redirect can land somewhere the pre-flight check never saw, so
                # re-check the URL actually served before trusting the body.
                if response.url != url and not is_fetchable_url(response.url):
                    return ""
                markup = response.text
        except requests.RequestException as exc:
            log.debug("Không đọc được %s: %s", url, exc)
            return ""
        except Exception:  # noqa: BLE001 - reading a page is best-effort
            log.debug("Lỗi không mong đợi khi đọc %s", url, exc_info=True)
            return ""

        return extract_main_text(markup, self._config.page_chars, keywords)

    def answer_context(self, query: str) -> str:
        """Return one block of URL-free material for the LLM to synthesise from.

        Deliberately not a ranked list: the caller is a voice assistant, and reading
        five titles with their sources aloud is unusable.

        Material is added in descending order of how much it usually carries: page
        bodies first (concrete figures live there), then Brave's per-query excerpts,
        then the one-line descriptions. The last two mean the model still has something
        to work with when every fetch fails, or when ``read_pages`` is 0.
        """
        results = self.search(query)
        if not results:
            return f"Không tìm thấy thông tin nào về {query!r}."

        budget = self._config.context_chars
        keywords = query_keywords(query)
        parts: list[str] = []
        pages_read = 0

        def remaining() -> int:
            """Budget left, counting the space ``" ".join`` will insert."""
            used = sum(len(part) for part in parts) + max(0, len(parts) - 1)
            return budget - used - (1 if parts else 0)

        #: Below this a chunk is a word fragment, worth less than the space it costs.
        floor = 80

        for result in results:
            if pages_read >= self._config.read_pages or remaining() < floor:
                break
            body = strip_urls(self.read_page(result.url, keywords))
            if len(body) < 120:  # A stub or a cookie wall carries no information.
                continue
            parts.append(_truncate_words(body, remaining()))
            pages_read += 1

        # Brave's extra_snippets are excerpts the engine picked *for this query*, so
        # they carry whole sentences with figures in them. That makes them a direct
        # substitute for reading the page, which is why the Brave backend works well
        # with ``read_pages: 0``. Other backends leave ``extras`` empty; this loop is
        # then a no-op.
        #
        # They still get the same relevance filter as page text: on a news portal's
        # category page some excerpts are neighbouring teasers ("HDBank xác định chủ
        # nhân 64 sổ tiết kiệm…"), and those mislead a 4B model just as effectively.
        for result in results:
            if remaining() < floor:
                break
            for extra in result.extras:
                if remaining() < floor:
                    break
                text = strip_urls(extra)
                if len(text) < 40 or any(text in part for part in parts):
                    continue
                if not _mentions_any(text, keywords):
                    log.debug("Bỏ đoạn trích lạc đề: %s…", text[:80])
                    continue
                parts.append(_truncate_words(text, remaining()))

        # Snippets are the floor: without them a fully blocked fetch yields nothing.
        for result in results:
            if remaining() < floor:
                break
            snippet = strip_urls(result.snippet)
            if len(snippet) < 40 or any(snippet in part for part in parts):
                continue
            parts.append(_truncate_words(snippet, remaining()))

        material = " ".join(parts).strip()
        if not material:
            return (
                f"Đã tìm thấy nguồn về {query!r} nhưng không đọc được nội dung nào. "
                "Hãy nói với người dùng là chưa tra được thông tin này."
            )

        log.info(
            "answer_context %r: %d trang đã đọc, %d ký tự tư liệu",
            query,
            pages_read,
            len(material),
        )
        # The fence marks everything inside as quoted data. It does not stop prompt
        # injection - nothing at this layer can - but it stops the model from reading
        # the instruction line below as part of the article.
        return (
            f"Tư liệu thu được về {query!r} (đã bỏ mọi đường dẫn):\n"
            "--- BẮT ĐẦU TƯ LIỆU ---\n"
            f"{material}\n"
            "--- HẾT TƯ LIỆU ---\n"
            "Hãy tổng hợp tư liệu trên thành MỘT câu trả lời tiếng Việt duy nhất, "
            "dài 1 đến 2 câu, nêu rõ con số và mốc thời gian nếu tư liệu có. "
            "Không liệt kê từng nguồn, không đánh số, không đọc tên trang web. "
            "Nếu tư liệu không trả lời được câu hỏi, hãy nói thẳng là chưa tra được."
        )

    def search_as_text(self, query: str) -> str:
        """Ranked list *with* URLs. For ``jarvis search --raw`` only, never for TTS."""
        results = self.search(query)
        if not results:
            return f"Không tìm thấy kết quả nào cho {query!r}."
        header = f"Kết quả tìm kiếm cho {query!r}:"
        body = "\n".join(result.render(index) for index, result in enumerate(results, start=1))
        return f"{header}\n{body}"

    # -- backends ----------------------------------------------------------------
    def _search_brave(self, query: str) -> list[SearchResult]:
        """Query the official Brave Search API.

        The subscription token is passed per request, never merged into the shared
        session headers: that same session is what :meth:`read_page` uses to fetch
        arbitrary result pages, and a token on the session would be sent to every one
        of them.
        """
        import requests

        key = self._config.resolved_brave_api_key()
        if not key:
            raise WebSearchError(BRAVE_KEY_HINT)

        params: dict[str, str | int] = {
            "q": query,
            "count": min(self._config.max_results, BRAVE_MAX_COUNT),
            "country": self._config.brave_country,
            "search_lang": self._config.brave_search_lang,
        }
        if self._config.brave_extra_snippets:
            params["extra_snippets"] = "true"
        if self._config.brave_freshness:
            params["freshness"] = self._config.brave_freshness

        headers = {
            "X-Subscription-Token": key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        try:
            response = self._http().get(
                BRAVE_ENDPOINT,
                params=params,
                headers=headers,
                timeout=self._config.timeout_s,
            )
        except requests.RequestException as exc:
            raise WebSearchError(
                f"Không kết nối được Brave Search API: {exc}. Kiểm tra mạng."
            ) from exc

        self._raise_for_brave_status(response.status_code, response.text)

        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError(
                "Brave Search API không trả về JSON hợp lệ."
            ) from exc
        if not isinstance(payload, dict):
            raise WebSearchError("Brave Search API trả về JSON không đúng định dạng.")

        results = parse_brave_json(payload, self._config.max_results)
        log.info(
            "Brave %r -> %d kết quả, %d đoạn trích phụ (quota còn %s)",
            query,
            len(results),
            sum(len(result.extras) for result in results),
            response.headers.get("x-ratelimit-remaining", "?"),
        )
        return results

    @staticmethod
    def _raise_for_brave_status(status_code: int, body: str) -> None:
        """Turn Brave's error statuses into messages that say what to do next.

        Each of these is a different user action - fix the key, wait, adjust a
        parameter - so collapsing them into one "search failed" would hide the fix.

        The status code alone is not enough to tell them apart. Verified against the
        live API: an invalid subscription token comes back as **HTTP 422** with
        ``"The provided API key is invalid."``, sharing a status with genuine parameter
        errors. Routing on the code alone told the user to fix ``brave_country`` when
        the real problem was the key, so the body decides first.
        """
        if status_code == 200:
            return

        detail = brave_error_detail(body)
        lowered = detail.lower()

        if status_code in (401, 403) or "api key" in lowered or "subscription" in lowered:
            raise WebSearchError(
                f"Brave Search API từ chối API key (HTTP {status_code}): {detail} "
                "Kiểm tra BRAVE_API_KEY hoặc tools.web_search.brave_api_key trong "
                "config.local.yaml, và xem key có thuộc plan Search đang hoạt động không."
            )
        if status_code == 429 or "rate limit" in lowered or "quota" in lowered:
            raise WebSearchError(
                f"Brave Search API báo vượt hạn mức (HTTP {status_code}): {detail} "
                "Free tier là 1 truy vấn/giây và khoảng 1.000 truy vấn/tháng từ 5 đô "
                "credit; hãy chờ một chút rồi thử lại."
            )
        if status_code == 422:
            hint = (
                " Lưu ý brave_country là enum đóng của Brave và KHÔNG có Việt Nam, "
                "hãy để 'ALL'."
                if "country" in lowered
                else ""
            )
            raise WebSearchError(f"Brave Search API từ chối tham số: {detail}.{hint}")
        raise WebSearchError(f"Brave Search API lỗi HTTP {status_code}: {detail}")

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

        if is_duckduckgo_throttled(response.status_code, response.text):
            log.warning("DuckDuckGo chặn tạm thời (HTTP %s, trang anomaly)", response.status_code)
            raise WebSearchError(
                "DuckDuckGo đang tạm chặn máy này vì gửi quá nhiều truy vấn "
                "(HTTP 202, trang chống bot). Hãy chờ ít phút rồi thử lại, hoặc "
                "chuyển tools.web_search.backend sang searxng trong config.yaml."
            )

        results = parse_duckduckgo_html(response.text, self._config.max_results)
        log.info("DuckDuckGo %r -> %d kết quả", query, len(results))
        return results

    def _search_ddgs(self, query: str) -> list[SearchResult]:
        """Query several engines at once through the optional ``ddgs`` package."""
        try:
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException, TimeoutException
        except ImportError as exc:
            raise WebSearchError(DDGS_INSTALL_HINT) from exc

        try:
            with DDGS(timeout=max(1, int(self._config.timeout_s))) as client:
                raw = client.text(
                    query,
                    region=self._config.region,
                    max_results=self._config.max_results,
                    backend=self._config.ddgs_engines,
                )
        except TimeoutException as exc:
            raise WebSearchError(
                f"Các engine của ddgs đều quá {self._config.timeout_s:.0f} giây: {exc}"
            ) from exc
        except DDGSException as exc:
            # ddgs raises rather than returning an empty list, and reuses one exception
            # type for "nothing matched" and "every engine failed". Only the bare
            # sentinel means the query genuinely found nothing; anything else carries
            # the underlying engine error and must not be reported as "no results".
            if str(exc).strip() == _DDGS_NO_RESULTS:
                log.info("ddgs %r -> 0 kết quả", query)
                return []
            raise WebSearchError(
                f"ddgs không tìm được kết quả nào: {exc}. "
                "Thử đổi tools.web_search.ddgs_engines, ví dụ 'google,bing,brave'."
            ) from exc

        results = [
            SearchResult(
                _clean(str(item.get("title", ""))),
                _clean(str(item.get("body", "")))[:MAX_SNIPPET_CHARS],
                str(item.get("href", "")),
            )
            for item in raw
        ]
        results = [result for result in results if result.title]
        log.info("ddgs (%s) %r -> %d kết quả", self._config.ddgs_engines, query, len(results))
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
