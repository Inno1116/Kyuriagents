from __future__ import annotations

import pytest

from deepagents.rag import (
    ChunkMetadata,
    DocumentChunk,
    HybridRAGRetriever,
    HybridSearchConfig,
    IdentityQueryRewriter,
    InMemoryKeywordStore,
    InMemoryVectorStore,
    LexicalReranker,
    RetrievalScope,
    RetrievedChunk,
)


def _metadata(
    chunk_id: str,
    *,
    tenant_id: str = "tenant-a",
    kb_id: str = "kb-main",
    user_id: str | None = None,
    doc_id: str | None = None,
    tags: tuple[str, ...] = (),
    is_active: bool = True,
) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        user_id=user_id,
        kb_id=kb_id,
        doc_id=doc_id or f"doc-{chunk_id}",
        doc_version=f"version-{chunk_id}",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
        source_type="md",
        source_uri=f"/docs/{chunk_id}.md",
        title=f"title {chunk_id}",
        section_path="RAG / Retrieval",
        tags=tags,
        embedding_model="test-embedding",
        embedding_version="v1",
        is_active=is_active,
    )


def _embed_query(query: str) -> tuple[float, float, float]:
    lowered = query.lower()
    return (
        1.0 if "auth" in lowered or "login" in lowered else 0.0,
        1.0 if "billing" in lowered else 0.0,
        1.0 if "rerank" in lowered else 0.0,
    )


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text="Authentication and login tokens are configured here.",
            metadata=_metadata("auth-1", tags=("security",)),
            embedding=(1.0, 0.0, 0.0),
            keywords=("auth", "login"),
        ),
        DocumentChunk(
            text="Billing invoices and payment history live in this section.",
            metadata=_metadata("billing-1"),
            embedding=(0.0, 1.0, 0.0),
            keywords=("billing", "invoice"),
        ),
        DocumentChunk(
            text="Rerank models reorder hybrid retrieval candidates.",
            metadata=_metadata("rerank-1", user_id="user-1"),
            embedding=(0.0, 0.0, 1.0),
            keywords=("rerank", "hybrid"),
        ),
        DocumentChunk(
            text="Authentication from another tenant must not leak.",
            metadata=_metadata("other-tenant", tenant_id="tenant-b"),
            embedding=(1.0, 0.0, 0.0),
            keywords=("auth",),
        ),
        DocumentChunk(
            text="Inactive authentication content must not be returned.",
            metadata=_metadata("inactive", is_active=False),
            embedding=(1.0, 0.0, 0.0),
            keywords=("auth",),
        ),
    ]


def _retriever() -> HybridRAGRetriever:
    chunks = _chunks()
    return HybridRAGRetriever(
        vector_searcher=InMemoryVectorStore(chunks, embed_query=_embed_query),
        keyword_searcher=InMemoryKeywordStore(chunks),
        reranker=LexicalReranker(),
        config=HybridSearchConfig(
            vector_candidates=10,
            keyword_candidates=10,
            rerank_candidates=10,
            top_k=5,
        ),
    )


def test_hybrid_retrieval_filters_scope_before_fusion() -> None:
    retriever = _retriever()

    results = retriever.retrieve(
        "auth login",
        scope=RetrievalScope(tenant_id="tenant-a", kb_ids=("kb-main",)),
    )

    assert [result.chunk_id for result in results] == ["auth-1"]


def test_hybrid_retrieval_allows_private_user_chunks_for_owner() -> None:
    retriever = _retriever()

    results = retriever.retrieve(
        "rerank hybrid",
        scope=RetrievalScope(tenant_id="tenant-a", user_id="user-1"),
    )

    assert results[0].chunk_id == "rerank-1"


def test_hybrid_retrieval_hides_private_user_chunks_from_other_users() -> None:
    retriever = _retriever()

    results = retriever.retrieve(
        "rerank hybrid",
        scope=RetrievalScope(tenant_id="tenant-a", user_id="user-2"),
    )

    assert [result.chunk_id for result in results] == []


def test_query_rewrite_expansions_participate_in_retrieval() -> None:
    chunks = _chunks()
    retriever = HybridRAGRetriever(
        vector_searcher=InMemoryVectorStore(chunks, embed_query=_embed_query),
        keyword_searcher=InMemoryKeywordStore(chunks),
        query_rewriter=IdentityQueryRewriter({"auth": ("login",)}),
        config=HybridSearchConfig(top_k=1),
    )

    results = retriever.retrieve(
        "auth setup",
        scope=RetrievalScope(tenant_id="tenant-a"),
    )

    assert results[0].chunk_id == "auth-1"
    assert results[0].vector_score is not None
    assert results[0].keyword_score is not None


def test_top_k_override_limits_results_after_rerank() -> None:
    retriever = _retriever()

    results = retriever.retrieve(
        "auth billing rerank",
        scope=RetrievalScope(tenant_id="tenant-a", user_id="user-1"),
        top_k=2,
    )

    assert len(results) == 2
    assert all(result.rerank_score is not None for result in results)


def test_fusion_prefers_keyword_text_when_vector_hit_is_metadata_only() -> None:
    metadata = _metadata("auth-1")

    class VectorOnly:
        def search(self, query, *, scope, limit):
            return [
                RetrievedChunk(
                    text="",
                    metadata=metadata,
                    vector_score=0.9,
                )
            ]

    class KeywordWithText:
        def search(self, query, *, scope, limit):
            return [
                RetrievedChunk(
                    text="Authentication setup text",
                    metadata=metadata,
                    keyword_score=3.0,
                )
            ]

    retriever = HybridRAGRetriever(
        vector_searcher=VectorOnly(),
        keyword_searcher=KeywordWithText(),
    )

    results = retriever.retrieve("auth", scope=RetrievalScope(tenant_id="tenant-a"))

    assert results[0].text == "Authentication setup text"
    assert results[0].vector_score == 0.9
    assert results[0].keyword_score == 3.0


def test_hybrid_retrieval_hydrates_vector_only_text_before_rerank() -> None:
    metadata = _metadata("auth-1")

    class VectorOnly:
        def search(self, query, *, scope, limit):
            return [
                RetrievedChunk(
                    text="",
                    metadata=metadata,
                    vector_score=0.9,
                )
            ]

    class NoKeywords:
        def search(self, query, *, scope, limit):
            return []

    class Hydrator:
        def hydrate(self, candidates):
            return [candidate.with_text("Authentication setup text from PostgreSQL") for candidate in candidates]

    retriever = HybridRAGRetriever(
        vector_searcher=VectorOnly(),
        keyword_searcher=NoKeywords(),
        chunk_hydrator=Hydrator(),
    )

    results = retriever.retrieve("auth", scope=RetrievalScope(tenant_id="tenant-a"))

    assert results[0].text == "Authentication setup text from PostgreSQL"
    assert results[0].rerank_score is not None


def test_invalid_hybrid_config_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        HybridSearchConfig(top_k=0)
