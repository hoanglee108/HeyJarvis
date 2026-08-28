"""mvp.md Task 10: parsing search results without touching the network."""

from __future__ import annotations

import re
import socket
import sys
import types

import pytest

from jarvis.config import BRAVE_API_KEY_ENV_VARS, WebSearchToolConfig
from jarvis.tools import SearchResult, WebSearch, WebSearchError
from jarvis.tools.websearch import (
    BRAVE_MAX_COUNT,
    brave_error_detail,
    extract_main_text,
    is_fetchable_url,
    parse_brave_json,
    parse_duckduckgo_html,
    parse_searxng_json,
    query_keywords,
    strip_urls,
)

DDG_HTML = """
<html><body>
  <div class="result results_links results_links_deep web-result">
    <h2 class="result__title">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvnexpress.net%2Fthoi-tiet&amp;rut=abc">
        Thời tiết Hà Nội hôm nay
      </a>
    </h2>
    <a class="result__snippet">Hà Nội nhiều mây, nhiệt độ <b>28</b> độ C, có mưa rào rải rác.</a>
  </div>
  <div class="result results_links web-result">
    <h2 class="result__title">
      <a class="result__a" href="https://www.accuweather.com/vi/vn/hanoi">
        Dự báo thời tiết Hà Nội - AccuWeather
      </a>
    </h2>
    <a class="result__snippet">Dự báo 10 ngày tới cho Hà Nội.</a>
  </div>
  <div class="result results_links_deep result--ad">
    <h2><a class="result__a" href="https://ads.example.com">Quảng cáo</a></h2>
  </div>
</body></html>
"""


def test_parses_titles_snippets_and_urls() -> None:
    results = parse_duckduckgo_html(DDG_HTML, max_results=5)
    assert len(results) >= 2
    first = results[0]
    assert first.title == "Thời tiết Hà Nội hôm nay"
    assert "28 độ C" in first.snippet
    assert first.url == "https://vnexpress.net/thoi-tiet", "phải bỏ lớp redirect của DuckDuckGo"


def test_html_tags_inside_snippets_are_stripped() -> None:
    snippet = parse_duckduckgo_html(DDG_HTML, max_results=1)[0].snippet
    assert "<b>" not in snippet


def test_max_results_is_respected() -> None:
    assert len(parse_duckduckgo_html(DDG_HTML, max_results=1)) == 1


def test_empty_markup_yields_no_results() -> None:
    assert parse_duckduckgo_html("<html><body>không có gì</body></html>", max_results=5) == []


def test_searxng_json_is_parsed() -> None:
    payload = {
        "results": [
            {"title": "Kết quả 1", "content": "Nội dung 1", "url": "https://a.vn"},
            {"title": "", "content": "bỏ qua vì không có tiêu đề", "url": "https://b.vn"},
            {"title": "Kết quả 2", "content": "Nội dung 2", "url": "https://c.vn"},
        ]
    }
    results = parse_searxng_json(payload, max_results=5)
    assert [r.title for r in results] == ["Kết quả 1", "Kết quả 2"]


def test_empty_query_is_rejected() -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True))
    with pytest.raises(WebSearchError, match="đang trống"):
        search.search("   ")


def test_disabled_tool_refuses() -> None:
    search = WebSearch(WebSearchToolConfig(enabled=False))
    with pytest.raises(WebSearchError, match="đang bị tắt"):
        search.search("thời tiết")


def test_searxng_without_url_gives_actionable_error() -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True, backend="searxng", searxng_url=None))
    with pytest.raises(WebSearchError, match="searxng_url"):
        search.search("thời tiết")


def test_results_render_as_numbered_text(monkeypatch: pytest.MonkeyPatch) -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True))
    monkeypatch.setattr(
        search, "search", lambda _query: parse_duckduckgo_html(DDG_HTML, max_results=2)
    )
    text = search.search_as_text("thời tiết hà nội")
    assert text.startswith("Kết quả tìm kiếm cho")
    assert "1. Thời tiết Hà Nội hôm nay" in text
    assert "2. Dự báo thời tiết Hà Nội" in text


# --------------------------------------------------------------------------------------
# Voice-facing output: one synthesised block, never a list of links
# --------------------------------------------------------------------------------------
ARTICLE_HTML = """
<html><body>
  <nav>Trang chủ Tin tức Liên hệ</nav>
  <script>var ads = 1;</script>
  <style>.x { color: red }</style>
  <article>
    <h1>Giá vàng SJC hôm nay</h1>
    <p>Giá vàng miếng SJC sáng nay được niêm yết ở mức 119.500.000 đồng mỗi lượng
       chiều mua vào và 121.500.000 đồng chiều bán ra, theo công bố lúc 9 giờ.</p>
    <p>Giá vàng miếng SJC sáng nay được niêm yết ở mức 119.500.000 đồng mỗi lượng
       chiều mua vào và 121.500.000 đồng chiều bán ra, theo công bố lúc 9 giờ.</p>
    <p>ngắn</p>
    <p>Chênh lệch giữa hai chiều là 2.000.000 đồng, giữ nguyên so với hôm qua.</p>
  </article>
  <footer>Bản quyền thuộc về toà soạn</footer>
</body></html>
"""

def test_strip_urls_removes_links_hosts_and_emails() -> None:
    text = strip_urls(
        "Xem tại https://www.sjc.com.vn/gia-vang hoặc www.24h.com.vn, "
        "theo vnexpress.net, liên hệ toasoan@example.org"
    )
    for fragment in ("http", "www.", "sjc.com.vn", "vnexpress.net", "@"):
        assert fragment not in text, f"còn sót {fragment!r} trong {text!r}"
    assert "Xem tại" in text


def test_strip_urls_keeps_numbers_that_look_like_hosts() -> None:
    """The bare-host pattern must not eat prices: 119.500.000 is not a domain."""
    text = strip_urls("Giá vàng 119.500.000 đồng, tăng 1.5 phần trăm so với 2.000.000.")
    assert "119.500.000" in text
    assert "1.5" in text
    assert "2.000.000" in text


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:1234/v1/models",  # LM Studio itself
        "http://localhost:8080/",
        "https://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://192.168.1.10/admin",
        "file:///C:/Windows/system32/calc.exe",
        "ftp://example.com/x",
        "",
    ],
)
def test_internal_and_non_http_urls_are_not_fetchable(url: str) -> None:
    assert is_fetchable_url(url) is False


def test_public_url_is_fetchable(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS is stubbed so the check stays offline like the rest of this module."""
    monkeypatch.setattr(
        "jarvis.tools.websearch.socket.getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert is_fetchable_url("https://example.com/bai-viet") is True


def test_unresolvable_host_is_not_fetchable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise socket.gaierror("không phân giải được")

    monkeypatch.setattr("jarvis.tools.websearch.socket.getaddrinfo", boom)
    assert is_fetchable_url("https://khong-ton-tai.invalid/") is False


def test_extract_main_text_keeps_body_and_drops_chrome() -> None:
    text = extract_main_text(ARTICLE_HTML, limit=2000)
    assert "119.500.000" in text
    assert "Chênh lệch" in text
    for chrome in ("var ads", "color: red", "Liên hệ", "Bản quyền"):
        assert chrome not in text


def test_extract_main_text_dedupes_and_skips_stubs() -> None:
    text = extract_main_text(ARTICLE_HTML, limit=2000)
    assert text.count("Chênh lệch giữa hai chiều") == 1
    assert text.count("chiều bán ra") == 1, "đoạn trùng lặp phải bị bỏ"
    assert "ngắn" not in text.split(), "đoạn quá ngắn là nav crumb, phải bỏ"


def test_extract_main_text_respects_the_limit() -> None:
    assert len(extract_main_text(ARTICLE_HTML, limit=120)) <= 120


# --------------------------------------------------------------------------------------
# relevance: a category page's unrelated teasers must not reach the model
# --------------------------------------------------------------------------------------
CATEGORY_HTML = """
<html><body><main>
  <p>Tay vợt pickleball Đỗ Minh Quân rời sân bằng xe cứu thương sau chung kết
     giải đấu tại Thanh Hóa, phải nhờ tới sự chăm sóc y tế ngay tại sân.</p>
  <p>Giá vàng miếng SJC sáng nay niêm yết 119.500.000 đồng mỗi lượng chiều mua vào,
     tăng 500.000 đồng so với hôm qua theo công bố lúc 9 giờ.</p>
  <p>Đội tuyển bóng đá quốc gia sẽ tập trung vào tháng sau để chuẩn bị cho vòng
     loại, danh sách cầu thủ dự kiến công bố trong tuần này.</p>
</main></body></html>
"""


def test_query_keywords_drops_filler_words() -> None:
    assert query_keywords("giá vàng SJC hôm nay") == ["giá", "vàng", "sjc"]
    assert query_keywords("bây giờ là mấy giờ") == ["mấy"]


def test_relevant_paragraph_comes_first_on_a_category_page() -> None:
    text = extract_main_text(
        CATEGORY_HTML, limit=2000, keywords=query_keywords("giá vàng SJC hôm nay")
    )
    assert text.startswith("Giá vàng miếng SJC"), (
        "đoạn khớp truy vấn phải đứng trước tin lạc đề"
    )
    assert "119.500.000" in text


def test_off_topic_paragraph_is_dropped_when_the_budget_is_tight() -> None:
    """This is the pickleball case: a narrow budget must spend it on the gold price."""
    text = extract_main_text(
        CATEGORY_HTML, limit=170, keywords=query_keywords("giá vàng SJC hôm nay")
    )
    assert "119.500.000" in text
    assert "pickleball" not in text
    assert "bóng đá" not in text


def test_page_about_something_else_contributes_nothing() -> None:
    """Off-topic material labelled as an answer is worse than no material at all."""
    assert extract_main_text(CATEGORY_HTML, limit=2000, keywords=["pin", "laptop"]) == ""


def test_biggest_container_wins_on_a_listing_page() -> None:
    """One <article> per teaser: taking the first would keep a single unrelated card."""
    markup = """
    <html><body>
      <article><p>Thẻ tin lạc đề về giải pickleball tại Thanh Hóa năm nay.</p></article>
      <main>
        <p>Giá vàng miếng SJC sáng nay niêm yết 119.500.000 đồng mỗi lượng chiều mua.</p>
        <p>Chênh lệch hai chiều mua bán giữ ở mức 2.000.000 đồng mỗi lượng.</p>
      </main>
    </body></html>
    """
    text = extract_main_text(markup, limit=2000, keywords=query_keywords("giá vàng SJC"))
    assert "119.500.000" in text
    assert "pickleball" not in text


def test_no_keywords_keeps_document_order() -> None:
    text = extract_main_text(CATEGORY_HTML, limit=2000)
    assert text.startswith("Tay vợt pickleball")


def _search_stub(results: list[SearchResult]):
    return lambda _query: list(results)


def test_answer_context_returns_one_block_without_any_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True, read_pages=1))
    monkeypatch.setattr(
        search,
        "search",
        _search_stub([SearchResult("Giá vàng", "đoạn trích", "https://vnexpress.net/a")]),
    )
    monkeypatch.setattr(
        search,
        "read_page",
        lambda _url, keywords=(): extract_main_text(ARTICLE_HTML, 1200, keywords),
    )

    text = search.answer_context("giá vàng SJC hôm nay")

    assert "http" not in text and "vnexpress.net" not in text
    assert "119.500.000" in text, "phải mang được con số thật từ nội dung trang"
    assert "TƯ LIỆU" in text, "tư liệu phải được rào lại để tách khỏi dòng chỉ dẫn"
    assert "1 đến 2 câu" in text, "phải yêu cầu model tổng hợp thành một câu trả lời"
    assert not re.search(r"^\s*\d+\.\s", text, re.MULTILINE), "không được đánh số kiểu danh sách"


def test_answer_context_falls_back_to_snippets_when_pages_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True, read_pages=2))
    monkeypatch.setattr(
        search,
        "search",
        _search_stub(
            [
                SearchResult(
                    "Thời tiết",
                    "Hà Nội nhiều mây, nhiệt độ 28 độ C, có mưa rào rải rác vào chiều nay.",
                    "https://vnexpress.net/thoi-tiet",
                )
            ]
        ),
    )
    monkeypatch.setattr(search, "read_page", lambda *_a, **_k: "")  # mọi trang đều chặn

    text = search.answer_context("thời tiết hà nội")
    assert "28 độ C" in text
    assert "http" not in text


def test_answer_context_says_so_when_there_is_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True))
    monkeypatch.setattr(search, "search", _search_stub([]))
    assert "Không tìm thấy" in search.answer_context("truy vấn vô nghĩa")


def test_answer_context_stays_within_the_configured_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True, read_pages=3, context_chars=400))
    monkeypatch.setattr(
        search,
        "search",
        _search_stub([SearchResult(f"T{i}", "x" * 300, f"https://a{i}.vn") for i in range(5)]),
    )
    monkeypatch.setattr(search, "read_page", lambda *_a, **_k: "nội dung dài " * 200)

    text = search.answer_context("câu hỏi")
    material = text.split("--- BẮT ĐẦU TƯ LIỆU ---")[1].split("--- HẾT TƯ LIỆU ---")[0]
    assert len(material.strip()) <= 400 + 1


def test_read_page_refuses_internal_urls_without_any_request() -> None:
    """The SSRF guard must run before requests is ever reached."""
    search = WebSearch(WebSearchToolConfig(enabled=True))

    def explode() -> None:
        raise AssertionError("không được gửi request tới địa chỉ nội bộ")

    search._session = type("Boom", (), {"get": lambda *_a, **_k: explode()})()
    assert search.read_page("http://127.0.0.1:1234/v1/models") == ""


# --------------------------------------------------------------------------------------
# backend "ddgs": the optional metasearch package
#
# `ddgs` is stubbed into sys.modules so these run offline and behave the same whether or
# not the real package happens to be installed in the developer's venv.
# --------------------------------------------------------------------------------------
class _FakeDDGSException(Exception):
    pass


class _FakeTimeoutException(_FakeDDGSException):
    pass


def _install_fake_ddgs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: list[dict[str, object]] | None = None,
    raises: Exception | None = None,
) -> dict[str, object]:
    """Register a fake `ddgs` package. Returns a dict recording the call arguments."""
    seen: dict[str, object] = {}

    class FakeDDGS:
        def __init__(self, timeout: int | None = None, **kwargs: object) -> None:
            seen["timeout"] = timeout

        def __enter__(self) -> "FakeDDGS":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def text(self, query: str, **kwargs: object) -> list[dict[str, object]]:
            seen["query"] = query
            seen.update(kwargs)
            if raises is not None:
                raise raises
            return list(results or [])

    ddgs_module = types.ModuleType("ddgs")
    ddgs_module.DDGS = FakeDDGS  # type: ignore[attr-defined]
    exceptions_module = types.ModuleType("ddgs.exceptions")
    exceptions_module.DDGSException = _FakeDDGSException  # type: ignore[attr-defined]
    exceptions_module.TimeoutException = _FakeTimeoutException  # type: ignore[attr-defined]
    ddgs_module.exceptions = exceptions_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ddgs", ddgs_module)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exceptions_module)
    return seen


def _ddgs_search(**overrides: object) -> WebSearch:
    return WebSearch(WebSearchToolConfig(enabled=True, backend="ddgs", **overrides))


def test_ddgs_backend_explains_how_to_install_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry in sys.modules makes the import fail, as if it were never installed.
    monkeypatch.setitem(sys.modules, "ddgs", None)
    with pytest.raises(WebSearchError, match="requirements-search.txt"):
        _ddgs_search().search("giá vàng")


def test_ddgs_results_become_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ddgs(
        monkeypatch,
        results=[
            {
                "title": "Giá vàng SJC hôm nay",
                "body": "Vàng SJC 1L niêm yết 147.600.000 đồng mỗi lượng.",
                "href": "https://sjc.com.vn/",
            },
            {"title": "", "body": "bỏ vì không có tiêu đề", "href": "https://x.vn/"},
        ],
    )
    results = _ddgs_search().search("giá vàng SJC")

    assert len(results) == 1, "kết quả không có tiêu đề phải bị loại"
    assert results[0].title == "Giá vàng SJC hôm nay"
    assert "147.600.000" in results[0].snippet
    assert results[0].url == "https://sjc.com.vn/"


def test_ddgs_receives_the_configured_region_engines_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_fake_ddgs(monkeypatch, results=[{"title": "x", "body": "y", "href": "z"}])
    _ddgs_search(region="vn-vi", max_results=3, ddgs_engines="google,bing").search("thời tiết")

    assert seen["query"] == "thời tiết"
    assert seen["region"] == "vn-vi"
    assert seen["max_results"] == 3
    assert seen["backend"] == "google,bing"


def test_ddgs_empty_result_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """ddgs raises instead of returning []; the bare sentinel means 'nothing matched'."""
    _install_fake_ddgs(monkeypatch, raises=_FakeDDGSException("No results found."))
    assert _ddgs_search().search("truy vấn vô nghĩa hoàn toàn") == []


def test_ddgs_engine_failure_is_reported_not_disguised_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same exception type as 'no results', so the message must decide. Reporting an
    engine failure as 'không tìm thấy' is the bug already fixed for DuckDuckGo."""
    _install_fake_ddgs(
        monkeypatch, raises=_FakeDDGSException("HTTPError('403 Forbidden from bing')")
    )
    with pytest.raises(WebSearchError, match="ddgs_engines"):
        _ddgs_search().search("giá vàng")


def test_ddgs_timeout_is_reported_as_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ddgs(monkeypatch, raises=_FakeTimeoutException("timed out"))
    with pytest.raises(WebSearchError, match="giây"):
        _ddgs_search(timeout_s=9).search("giá vàng")


def test_ddgs_backend_feeds_the_same_synthesis_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL-free block must be built the same way regardless of which engine ran."""
    _install_fake_ddgs(
        monkeypatch,
        results=[
            {
                "title": "Giá vàng",
                "body": "Vàng miếng SJC niêm yết 147.600.000 đồng mỗi lượng sáng nay, "
                "tăng so với hôm qua theo công bố mới nhất.",
                "href": "https://sjc.com.vn/gia-vang",
            }
        ],
    )
    search = _ddgs_search(read_pages=0)
    block = search.answer_context("giá vàng SJC hôm nay")

    assert "147.600.000" in block
    assert "http" not in block and "sjc.com.vn" not in block
    assert "1 đến 2 câu" in block


def test_ddgs_engine_rotation_default_is_unchanged() -> None:
    assert WebSearchToolConfig().ddgs_engines == "auto"


# --------------------------------------------------------------------------------------
# backend "brave": the official Brave Search API
#
# A fake session stands in for `requests`, so these stay offline and no test spends a
# request from the monthly free credit.
# --------------------------------------------------------------------------------------
BRAVE_PAYLOAD = {
    "query": {"original": "giá vàng SJC hôm nay"},
    "web": {
        "results": [
            {
                "title": "Giá vàng SJC hôm nay",
                "url": "https://baomoi.com/tien-ich-gia-vang-sjc.epi",
                "description": (
                    "Sáng nay giá vàng miếng SJC đứng im ở mức "
                    "<strong>147,6 triệu đồng/lượng</strong>."
                ),
                "extra_snippets": [
                    "Sáng nay (26/8), giá vàng miếng SJC đứng im ở mức 147,6 triệu "
                    "đồng mỗi lượng, trong khi vàng nhẫn quay đầu giảm.",
                    "Chiều 24-8, giá vàng trong nước tiếp tục tăng dù giá vàng thế "
                    "giới đi ngang, có nơi bán vàng nhẫn cao hơn vàng miếng.",
                    "   ",
                ],
            },
            {
                "title": "",
                "url": "https://bo-qua.vn/",
                "description": "không có tiêu đề nên phải bị loại",
            },
            {
                "title": "Biểu đồ giá vàng",
                "url": "https://sjc.com.vn/bieu-do-gia-vang",
                "description": "Điện thoại và fax của công ty.",
            },
        ]
    },
}


class _FakeResponse:
    def __init__(self, status_code: int, body: str, payload: object = None) -> None:
        self.status_code = status_code
        self.text = body
        self.headers = {"x-ratelimit-remaining": "49, 1998"}
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    """Records the one request the Brave backend makes."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.headers: dict[str, str] = {"User-Agent": "jarvis-test"}
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self._response


def _brave_search(response: _FakeResponse, **overrides: object) -> tuple[WebSearch, _FakeSession]:
    overrides.setdefault("brave_api_key", "BSA-test-key")
    search = WebSearch(WebSearchToolConfig(enabled=True, backend="brave", **overrides))
    session = _FakeSession(response)
    search._session = session
    return search, session


@pytest.fixture(autouse=True)
def _no_ambient_brave_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key in the developer's own environment must not change what these assert."""
    for name in BRAVE_API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_brave_is_the_default_backend() -> None:
    """config.yaml ships backend: brave; the code default must not disagree with it."""
    assert WebSearchToolConfig().backend == "brave"


def test_brave_country_defaults_to_all_because_vn_is_rejected() -> None:
    """Verified against the live API: country=VN answers HTTP 422 (closed enum)."""
    config = WebSearchToolConfig()
    assert config.brave_country == "ALL"
    assert config.brave_search_lang == "vi"


def test_brave_json_is_parsed_with_extras() -> None:
    results = parse_brave_json(BRAVE_PAYLOAD, max_results=5)

    assert [result.title for result in results] == ["Giá vàng SJC hôm nay", "Biểu đồ giá vàng"], (
        "kết quả không có tiêu đề phải bị loại"
    )
    first = results[0]
    assert "147,6 triệu đồng/lượng" in first.snippet
    assert "<strong>" not in first.snippet, "Brave bọc từ khoá trong <strong>, phải bóc ra"
    assert first.url == "https://baomoi.com/tien-ich-gia-vang-sjc.epi"
    assert len(first.extras) == 2, "đoạn trích chỉ có khoảng trắng phải bị bỏ"
    assert "147,6 triệu đồng mỗi lượng" in first.extras[0]
    assert results[1].extras == [], "kết quả không có extra_snippets thì extras rỗng"


def test_brave_extras_are_not_truncated_to_snippet_length() -> None:
    """The excerpts are where the figures live; cutting them at MAX_SNIPPET_CHARS
    would throw away the sentence that answers the question."""
    long_excerpt = "Giá vàng SJC " + "x" * 600
    payload = {"web": {"results": [{"title": "T", "description": "d", "url": "https://a.vn", "extra_snippets": [long_excerpt]}]}}
    assert len(parse_brave_json(payload, 5)[0].extras[0]) == len(long_excerpt)


def test_brave_missing_key_says_where_to_put_one() -> None:
    search = WebSearch(WebSearchToolConfig(enabled=True, backend="brave", brave_api_key=None))
    with pytest.raises(WebSearchError, match="BRAVE_API_KEY"):
        search.search("giá vàng")


def test_brave_key_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """config.yaml is committed, so the env var has to win over any value in a file."""
    monkeypatch.setenv("BRAVE_API_KEY", "BSA-from-env")
    config = WebSearchToolConfig(backend="brave", brave_api_key="BSA-from-file")
    assert config.resolved_brave_api_key() == "BSA-from-env"


def test_brave_key_falls_back_to_the_config_file() -> None:
    config = WebSearchToolConfig(backend="brave", brave_api_key="  BSA-from-file  ")
    assert config.resolved_brave_api_key() == "BSA-from-file"


def test_brave_sends_the_configured_query_params() -> None:
    search, session = _brave_search(
        _FakeResponse(200, "", BRAVE_PAYLOAD),
        max_results=4,
        brave_search_lang="vi",
        brave_country="ALL",
        brave_freshness="pw",
    )
    search.search("giá vàng SJC")

    call = session.calls[0]
    assert call["url"] == "https://api.search.brave.com/res/v1/web/search"
    params = call["params"]
    assert params["q"] == "giá vàng SJC"
    assert params["count"] == 4
    assert params["search_lang"] == "vi"
    assert params["country"] == "ALL"
    assert params["extra_snippets"] == "true"
    assert params["freshness"] == "pw"


def test_brave_count_is_capped_at_the_documented_maximum() -> None:
    search, session = _brave_search(_FakeResponse(200, "", BRAVE_PAYLOAD), max_results=15)
    search.search("x")
    assert session.calls[0]["params"]["count"] <= BRAVE_MAX_COUNT


def test_brave_freshness_is_omitted_when_unset() -> None:
    search, session = _brave_search(_FakeResponse(200, "", BRAVE_PAYLOAD), brave_freshness=None)
    search.search("x")
    assert "freshness" not in session.calls[0]["params"]


def test_brave_extra_snippets_can_be_turned_off() -> None:
    search, session = _brave_search(
        _FakeResponse(200, "", BRAVE_PAYLOAD), brave_extra_snippets=False
    )
    search.search("x")
    assert "extra_snippets" not in session.calls[0]["params"]


def test_brave_token_never_lands_on_the_shared_session() -> None:
    """read_page() fetches arbitrary result URLs with this same session. A token in the
    session headers would be sent to every one of those third-party hosts."""
    search, session = _brave_search(_FakeResponse(200, "", BRAVE_PAYLOAD))
    search.search("giá vàng")

    assert "X-Subscription-Token" not in session.headers
    assert session.calls[0]["headers"]["X-Subscription-Token"] == "BSA-test-key"


BRAVE_422_BODY = (
    '{"type":"ErrorResponse","error":{"id":"abc","status":422,'
    '"detail":"Unable to validate request parameter(s)",'
    '"meta":{"errors":[{"type":"enum","loc":["query","country"],'
    "\"msg\":\"Input should be 'AR', 'AU' or 'ALL'\",\"input\":\"VN\"}]}}}"
)


def test_brave_error_detail_names_the_rejected_parameter() -> None:
    detail = brave_error_detail(BRAVE_422_BODY)
    assert "Unable to validate" in detail
    assert "query.country" in detail, "phải chỉ rõ tham số nào bị từ chối"


def test_brave_error_detail_survives_a_non_json_body() -> None:
    assert "gateway" in brave_error_detail("502 bad gateway")
    assert brave_error_detail("") == "(không rõ lý do)"


#: Verified against the live API: a wrong subscription token is reported as HTTP 422,
#: the same status as a genuine parameter error.
BRAVE_BAD_KEY_BODY = (
    '{"type":"ErrorResponse","error":{"id":"abc","status":422,'
    '"code":"SUBSCRIPTION_TOKEN_INVALID",'
    '"detail":"The provided API key is invalid."}}'
)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, "", "API key"),
        (403, "", "API key"),
        (422, BRAVE_422_BODY, "query.country"),
        (429, "", "hạn mức"),
        (500, "boom", "HTTP 500"),
    ],
)
def test_brave_http_errors_are_actionable(status: int, body: str, expected: str) -> None:
    """Each status needs a different fix, so one generic message would hide it."""
    search, _ = _brave_search(_FakeResponse(status, body))
    with pytest.raises(WebSearchError, match=re.escape(expected)):
        search.search("giá vàng")


def test_brave_422_on_country_points_at_the_vietnam_trap() -> None:
    search, _ = _brave_search(_FakeResponse(422, BRAVE_422_BODY), brave_country="VN")
    with pytest.raises(WebSearchError, match="ALL"):
        search.search("giá vàng")


def test_brave_invalid_key_is_reported_as_a_key_problem_not_a_parameter_one() -> None:
    """Regression: Brave sends HTTP 422 for a bad key, so routing on the status code
    alone told the user to go and fix brave_country instead."""
    search, _ = _brave_search(_FakeResponse(422, BRAVE_BAD_KEY_BODY))
    with pytest.raises(WebSearchError) as caught:
        search.search("giá vàng")

    message = str(caught.value)
    assert "API key" in message
    assert "brave_country" not in message, "không được hướng người dùng đi sửa sai chỗ"
    assert "config.local.yaml" in message, "phải nói rõ đặt key ở đâu"


def test_brave_country_hint_only_appears_for_a_country_error() -> None:
    body = (
        '{"error":{"detail":"Unable to validate request parameter(s)",'
        '"meta":{"errors":[{"loc":["query","freshness"],"msg":"bad value"}]}}}'
    )
    search, _ = _brave_search(_FakeResponse(422, body))
    with pytest.raises(WebSearchError) as caught:
        search.search("giá vàng")
    assert "freshness" in str(caught.value)
    assert "brave_country" not in str(caught.value)


def test_brave_non_json_success_body_is_reported() -> None:
    search, _ = _brave_search(_FakeResponse(200, "<html>not json</html>", None))
    with pytest.raises(WebSearchError, match="JSON"):
        search.search("giá vàng")


def test_brave_extras_carry_the_answer_without_reading_any_page() -> None:
    """The point of extra_snippets: read_pages=0 must still yield the real figure."""
    search, _ = _brave_search(_FakeResponse(200, "", BRAVE_PAYLOAD), read_pages=0)

    def explode(*_a: object, **_k: object) -> str:
        raise AssertionError("read_pages=0 thì không được mở trang nào")

    search.read_page = explode  # type: ignore[method-assign]

    block = search.answer_context("giá vàng SJC hôm nay")
    assert "147,6 triệu" in block
    assert "http" not in block and "baomoi.com" not in block
    assert "1 đến 2 câu" in block


def test_brave_extras_are_deduped_against_the_description() -> None:
    """extra_snippets[0] is often the description verbatim; repeating it wastes budget."""
    payload = {
        "web": {
            "results": [
                {
                    "title": "T",
                    "url": "https://a.vn",
                    "description": "Giá vàng miếng SJC niêm yết 147,6 triệu đồng mỗi lượng.",
                    "extra_snippets": [
                        "Giá vàng miếng SJC niêm yết 147,6 triệu đồng mỗi lượng."
                    ],
                }
            ]
        }
    }
    search, _ = _brave_search(_FakeResponse(200, "", payload), read_pages=0)
    block = search.answer_context("giá vàng SJC")
    material = block.split("--- BẮT ĐẦU TƯ LIỆU ---")[1].split("--- HẾT TƯ LIỆU ---")[0]
    assert material.count("147,6 triệu đồng mỗi lượng") == 1


def test_brave_off_topic_extras_are_dropped() -> None:
    """Observed live: a news portal's category page returns neighbouring teasers as
    extra_snippets. Off-topic material labelled as an answer misleads a 4B model."""
    payload = {
        "web": {
            "results": [
                {
                    "title": "Giá vàng",
                    "url": "https://baomoi.com/x.epi",
                    "description": "Giá vàng miếng SJC niêm yết 147,6 triệu đồng mỗi lượng.",
                    "extra_snippets": [
                        "Sáng nay giá vàng miếng SJC đứng im ở mức 147,6 triệu đồng mỗi lượng.",
                        "Iconia Lakeside gia tăng đặc quyền cho chủ nhân nhà phố giới hạn "
                        "bằng loạt chính sách mới trong tháng này.",
                    ],
                }
            ]
        }
    }
    search, _ = _brave_search(_FakeResponse(200, "", payload), read_pages=0)
    block = search.answer_context("giá vàng SJC hôm nay")
    assert "147,6 triệu" in block
    assert "Iconia" not in block


def test_brave_extras_survive_when_the_query_has_no_content_words() -> None:
    """query_keywords() strips filler, and can legitimately return nothing. That must
    not silently discard every excerpt."""
    payload = {
        "web": {
            "results": [
                {
                    "title": "T",
                    "url": "https://a.vn",
                    "description": "d",
                    "extra_snippets": ["Một đoạn trích đủ dài để được giữ lại trong tư liệu."],
                }
            ]
        }
    }
    search, _ = _brave_search(_FakeResponse(200, "", payload), read_pages=0)
    assert query_keywords("là gì của và cho") == []
    assert "đủ dài" in search.answer_context("là gì của và cho")


def test_brave_material_stays_within_the_budget() -> None:
    payload = {
        "web": {
            "results": [
                {
                    "title": f"T{index}",
                    "url": f"https://a{index}.vn",
                    "description": "mô tả " * 40,
                    "extra_snippets": ["đoạn trích dài " * 40 for _ in range(5)],
                }
                for index in range(5)
            ]
        }
    }
    search, _ = _brave_search(
        _FakeResponse(200, "", payload), read_pages=0, context_chars=500
    )
    block = search.answer_context("câu hỏi")
    material = block.split("--- BẮT ĐẦU TƯ LIỆU ---")[1].split("--- HẾT TƯ LIỆU ---")[0]
    assert len(material.strip()) <= 500 + 1


def test_brave_results_keep_urls_for_the_music_tool() -> None:
    """play_song resolves a YouTube link out of search(), so URLs must survive even
    though answer_context strips them."""
    payload = {
        "web": {
            "results": [
                {
                    "title": "Em của ngày hôm qua - Sơn Tùng",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "description": "MV chính thức",
                }
            ]
        }
    }
    search, _ = _brave_search(_FakeResponse(200, "", payload))
    assert search.search("em của ngày hôm qua youtube")[0].url == (
        "https://www.youtube.com/watch?v=abc123"
    )
