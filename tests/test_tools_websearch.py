"""mvp.md Task 10: parsing search results without touching the network."""

from __future__ import annotations

import pytest

from jarvis.config import WebSearchToolConfig
from jarvis.tools import WebSearch, WebSearchError
from jarvis.tools.websearch import parse_duckduckgo_html, parse_searxng_json

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
