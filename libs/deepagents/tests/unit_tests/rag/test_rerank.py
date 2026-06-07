from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.rag import ChunkMetadata, DashScopeTextReranker, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Mapping


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, object]) -> _Response:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _Response(
            {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.98},
                        {"index": 0, "relevance_score": 0.12},
                    ]
                }
            }
        )


def _chunk(chunk_id: str, text: str, *, fused_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            tenant_id="tenant-a",
            kb_id="kb-main",
            doc_id=f"doc-{chunk_id}",
            doc_version=f"version-{chunk_id}",
            chunk_index=0,
            content_hash=f"hash-{chunk_id}",
            source_type="md",
            source_uri=f"/docs/{chunk_id}.md",
            title=f"title {chunk_id}",
            embedding_model="test-embedding",
            embedding_version="v1",
        ),
        fused_score=fused_score,
    )


def test_dashscope_text_reranker_orders_by_api_scores() -> None:
    client = _Client()
    reranker = DashScopeTextReranker(api_key="key", client=client)

    results = reranker.rerank(
        "auth setup",
        [
            _chunk("first", "first candidate", fused_score=0.5),
            _chunk("second", "second candidate", fused_score=0.1),
        ],
        limit=2,
    )

    assert [result.chunk_id for result in results] == ["second", "first"]
    assert results[0].rerank_score == 0.98
    call = client.calls[0]
    assert call["url"].endswith("/text-rerank")
    assert call["headers"]["Authorization"] == "Bearer key"
    assert call["json"]["model"] == "qwen3-vl-rerank"


def test_dashscope_text_reranker_falls_back_without_api_key() -> None:
    reranker = DashScopeTextReranker(api_key="")

    results = reranker.rerank(
        "auth setup",
        [
            _chunk("first", "first candidate", fused_score=0.1),
            _chunk("second", "second candidate", fused_score=0.9),
        ],
        limit=1,
    )

    assert [result.chunk_id for result in results] == ["second"]
