from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.runtime import AgentRuntimeConfig
from deepagents.websearch import (
    FetchedPage,
    WebSearchResponse,
    WebSearchService,
    blocked_query_reason,
    create_web_agent_tools,
    format_fetched_page,
    format_web_search_results,
)
from deepagents.websearch.service import (
    _decode_response_body,
    _normalize_url,
    _page_quality_flags,
    _plan_search_queries,
    _RedisSearchCache,
    _should_render_after_static,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_web_search_service_normalizes_searxng_results() -> None:
    config = AgentRuntimeConfig(enable_web_search=True, web_search_cache_ttl_seconds=0)
    service = WebSearchService(config)

    def fake_searxng_json(query: str, *, max_results: int) -> dict[str, object]:
        assert query == "Kyuriagents"
        assert max_results == 4
        return {
            "results": [
                {"title": "Kyuriagents", "url": "HTTPS://Example.COM/docs#frag", "content": "Agent platform"},
                {"title": "Duplicate", "url": "https://example.com/docs", "content": "duplicate"},
                {"title": "Other", "url": "https://example.org/path", "content": "other"},
            ]
        }

    service._searxng_json = fake_searxng_json  # ty: ignore[method-assign]

    results = service.search("Kyuriagents", max_results=2)

    assert [result.url for result in results] == ["https://example.com/docs", "https://example.org/path"]
    assert results[0].snippet == "Agent platform"


def test_web_search_plans_chinese_queries_without_translation() -> None:
    query = "\u0032\u0030\u0032\u0036\u5e74\u590f\u5929 \u8bbf\u95ee\u4e2d\u56fd \u5916\u56fd\u9886\u5bfc\u4eba"
    planned = _plan_search_queries(query, max_queries=3)

    assert planned[0] == query
    assert any("\u5b98\u65b9" in item for item in planned)
    assert any("\u8bbf\u534e" in item for item in planned)
    assert all("China" not in item for item in planned)


def test_web_search_normalizes_tracking_query_params() -> None:
    first = _normalize_url("https://example.com/news?id=1&utm_source=x&spm=abc#frag")
    second = _normalize_url("HTTPS://EXAMPLE.COM/news?spm=def&id=1")

    assert first == "https://example.com/news?id=1"
    assert second == "https://example.com/news?id=1"


def test_web_search_formats_sources() -> None:
    config = AgentRuntimeConfig(enable_web_search=True, web_search_cache_ttl_seconds=0)
    service = WebSearchService(config)

    def fake_searxng_json(query: str, *, max_results: int) -> dict[str, object]:
        assert query == "docs"
        assert max_results == 16
        return {"results": [{"title": "Docs", "url": "https://example.com/docs", "content": "Official docs"}]}

    service._searxng_json = fake_searxng_json  # ty: ignore[method-assign]

    output = format_web_search_results(service.search_with_diagnostics("docs"), query="docs")

    assert "<web_search_results" in output
    assert "<diagnostics>" in output
    assert "planned_queries:" in output
    assert "https://example.com/docs" in output
    assert "Official docs" in output


def test_web_search_formats_empty_cached_response() -> None:
    response = WebSearchResponse(query="docs", planned_queries=("docs",), cache_hit=True)

    output = format_web_search_results(response, query="docs")

    assert "No web search results found." in output


def test_web_fetch_formats_markdown_diagnostics_and_content() -> None:
    page = FetchedPage(
        requested_url="https://example.com/original",
        url="https://example.com/final",
        title="Example",
        text="useful page body",
        status="fetched",
        method="http",
        content_type="text/html",
        text_chars=16,
        returned_chars=16,
        quality_flags=("too_short",),
    )

    output = format_fetched_page(page)

    assert "# Web Page Fetch Result 1" in output
    assert "- Method: static" in output
    assert "- Quality flags: too_short" in output
    assert "## Extracted Content" in output
    assert "useful page body" in output


def test_web_fetch_quality_flags_can_trigger_render_fallback() -> None:
    flags = _page_quality_flags(
        "\u6b63\u5728\u52a0\u8f7d",
        raw_html="<html><body>\u6b63\u5728\u52a0\u8f7d</body></html>",
        download_truncated=False,
        text_truncated=False,
    )
    page = FetchedPage(
        requested_url="https://example.com",
        url="https://example.com",
        text="\u6b63\u5728\u52a0\u8f7d",
        status="fetched",
        method="http",
        quality_flags=flags,
    )

    assert "too_short" in flags
    assert "maybe_js_required" in flags
    assert _should_render_after_static(page)


def test_web_search_cache_skips_empty_failure_responses() -> None:
    cache = _RedisSearchCache(AgentRuntimeConfig(enable_web_search=True, web_search_cache_ttl_seconds=30))

    class FakeRedis:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str]] = []

        def setex(self, key: str, ttl: int, value: str) -> None:
            self.calls.append((key, ttl, value))

    client = FakeRedis()
    cache._client = client

    cache.set(
        WebSearchResponse(query="docs", planned_queries=("docs",), failures=("SearXNG 502",)),
        max_results=8,
    )

    assert client.calls == []


def test_web_fetch_decodes_html_meta_charset() -> None:
    html = '<html><head><meta charset="gb2312"><title>中国政府网</title></head><body>国务院</body></html>'
    body = html.encode("gb18030")

    decoded = _decode_response_body(body, "utf-8")

    assert "中国政府网" in decoded
    assert "国务院" in decoded


def test_web_search_falls_back_to_single_engine_when_aggregate_has_no_results() -> None:
    config = AgentRuntimeConfig(
        enable_web_search=True,
        web_search_query_plan_size=1,
        web_search_cache_ttl_seconds=0,
        web_search_fallback_engines=("bing", "baidu"),
    )
    service = WebSearchService(config)
    engines: list[str | None] = []

    def fake_try_searxng_json(httpx_module: object, *, base_url: str, params: Mapping[str, str], engine: str | None) -> dict[str, object]:
        del httpx_module, base_url, params
        engines.append(engine)
        if engine == "bing":
            return {"results": [{"title": "BJUT", "url": "https://example.com/bjut", "content": "校长任命信息"}]}
        return {"results": []}

    service._try_searxng_json = fake_try_searxng_json  # ty: ignore[method-assign]

    results = service.search("北京工业大学 校长 2024 任命")

    assert engines == [None, "bing"]
    assert results[0].title == "BJUT"


def test_web_search_retries_when_results_are_single_character_lookup() -> None:
    config = AgentRuntimeConfig(
        enable_web_search=True,
        web_search_query_plan_size=1,
        web_search_cache_ttl_seconds=0,
        web_search_fallback_engines=("duckduckgo",),
    )
    service = WebSearchService(config)
    engines: list[str | None] = []

    def fake_try_searxng_json(httpx_module: object, *, base_url: str, params: Mapping[str, str], engine: str | None) -> dict[str, object]:
        del httpx_module, base_url, params
        engines.append(engine)
        if engine is None:
            return {
                "results": [
                    {
                        "title": "\u7279\uff08\u6c49\u8bed\u6c49\u5b57\uff09_\u767e\u5ea6\u767e\u79d1",
                        "url": "https://example.com/zi1",
                        "content": "\u7279\u7684\u62fc\u97f3\u548c\u90e8\u9996",
                    },
                    {
                        "title": "\u7279\u7684\u610f\u601d_\u65b0\u534e\u5b57\u5178",
                        "url": "https://example.com/zi2",
                        "content": "\u7279\u5b57\u7b14\u987a",
                    },
                    {"title": "\u300a\u7279\u300b\u7684\u62fc\u97f3", "url": "https://example.com/zi3", "content": "\u7279\u5b57\u7684\u89e3\u91ca"},
                ]
            }
        return {"results": [{"title": "Trump visit timeline", "url": "https://example.com/trump", "content": "official timeline"}]}

    service._try_searxng_json = fake_try_searxng_json  # ty: ignore[method-assign]

    results = service.search("\u7279\u6717\u666e \u8bbf\u534e \u65f6\u95f4")

    assert engines == [None, "duckduckgo"]
    assert results[0].title == "Trump visit timeline"


def test_web_fetch_rejects_localhost_without_network() -> None:
    page = WebSearchService(AgentRuntimeConfig(enable_web_search=True)).fetch_url("http://127.0.0.1/private")

    assert page.status == "skipped"
    assert "not allowed" in page.error


def test_web_agent_tools_expose_search_static_fetch_and_render() -> None:
    tools = create_web_agent_tools(AgentRuntimeConfig(enable_web_search=True))

    assert {tool.name for tool in tools} == {"web_search", "web_fetch_static", "web_render_page"}


def test_web_static_and_render_tools_reject_localhost_without_network() -> None:
    tools = {tool.name: tool for tool in create_web_agent_tools(AgentRuntimeConfig(enable_web_search=True))}

    static_output = tools["web_fetch_static"].invoke({"url": "http://127.0.0.1/private"})
    render_output = tools["web_render_page"].invoke({"url": "http://127.0.0.1/private"})

    assert "not allowed" in static_output
    assert "not allowed" in render_output


def test_web_search_blocks_piracy_queries() -> None:
    reason = blocked_query_reason("某电视剧 夸克网盘资源")

    assert "pirated" in reason
