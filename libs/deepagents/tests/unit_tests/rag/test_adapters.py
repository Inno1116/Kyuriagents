from __future__ import annotations

from typing import Any

from deepagents.rag import ChunkMetadata, RetrievalScope
from deepagents.rag.elasticsearch import ElasticsearchKeywordStore
from deepagents.rag.milvus import MilvusVectorStore, build_milvus_filter


class _FakeElasticsearch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 12.5,
                        "_source": {
                            **_metadata().to_es_document("auth token setup"),
                            "keywords": ["auth"],
                        },
                    }
                ]
            }
        }


class _FakeMilvus:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return [
            [
                {
                    "distance": 0.91,
                    "entity": _metadata().to_milvus_fields(),
                }
            ]
        ]


def _metadata() -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="chunk-1",
        tenant_id="tenant-a",
        user_id="user-1",
        kb_id="kb-main",
        doc_id="doc-1",
        doc_version="version-1",
        chunk_index=0,
        content_hash="hash-1",
        source_type="md",
        source_uri="/docs/auth.md",
        title="Auth",
        section_path="Security / Auth",
        tags=("security",),
        embedding_model="test-embedding",
        embedding_version="v1",
    )


def test_elasticsearch_keyword_store_builds_filtered_query_and_parses_hits() -> None:
    client = _FakeElasticsearch()
    store = ElasticsearchKeywordStore(index="rag_chunks", client=client)

    results = store.search(
        "auth token",
        scope=RetrievalScope(
            tenant_id="tenant-a",
            user_id="user-1",
            kb_ids=("kb-main",),
            tags=("security",),
        ),
        limit=3,
    )

    assert results[0].chunk_id == "chunk-1"
    assert results[0].text == "auth token setup"
    assert results[0].keyword_score == 12.5
    call = client.calls[0]
    assert call["index"] == "rag_chunks"
    assert call["size"] == 3
    filters = call["query"]["bool"]["filter"]
    assert {"term": {"tenant_id": "tenant-a"}} in filters
    assert {"terms": {"kb_id": ["kb-main"]}} in filters
    assert {"term": {"tags": "security"}} in filters


def test_milvus_vector_store_builds_filtered_search_and_parses_hits() -> None:
    client = _FakeMilvus()
    store = MilvusVectorStore(
        collection_name="rag_chunks",
        client=client,
        embed_query=lambda _query: (1.0, 0.0),
    )

    results = store.search(
        "auth token",
        scope=RetrievalScope(tenant_id="tenant-a", user_id="user-1"),
        limit=5,
    )

    assert results[0].chunk_id == "chunk-1"
    assert results[0].vector_score == 0.91
    call = client.calls[0]
    assert call["collection_name"] == "rag_chunks"
    assert call["data"] == [[1.0, 0.0]]
    assert call["limit"] == 5
    assert 'tenant_id == "tenant-a"' in call["filter"]
    assert '(user_id == "" or user_id == "user-1")' in call["filter"]


def test_build_milvus_filter_escapes_values() -> None:
    expr = build_milvus_filter(
        RetrievalScope(
            tenant_id='tenant-"a"',
            kb_ids=("kb-main",),
            tags=("security",),
        )
    )

    assert 'tenant_id == "tenant-\\"a\\""' in expr
    assert 'kb_id in ["kb-main"]' in expr
    assert 'ARRAY_CONTAINS(tags, "security")' in expr
