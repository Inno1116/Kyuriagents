from __future__ import annotations

from deepagents.rag import ChunkMetadata, RetrievalScope


def _metadata(*, user_id: str | None = None, is_active: bool = True) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="chunk-1",
        tenant_id="tenant-a",
        user_id=user_id,
        kb_id="kb-main",
        doc_id="doc-1",
        doc_version="version-1",
        chunk_index=0,
        content_hash="hash-1",
        source_type="pdf",
        source_uri="/docs/source.pdf",
        tags=("policy",),
        is_active=is_active,
    )


def test_scope_requires_tenant_and_respects_private_user_owner() -> None:
    metadata = _metadata(user_id="user-1")

    assert RetrievalScope(tenant_id="tenant-a", user_id="user-1").matches(metadata)
    assert not RetrievalScope(tenant_id="tenant-a", user_id="user-2").matches(metadata)
    assert not RetrievalScope(tenant_id="tenant-a").matches(metadata)
    assert not RetrievalScope(tenant_id="tenant-b", user_id="user-1").matches(metadata)


def test_scope_filters_inactive_chunks_by_default() -> None:
    metadata = _metadata(is_active=False)

    assert not RetrievalScope(tenant_id="tenant-a").matches(metadata)
    assert RetrievalScope(tenant_id="tenant-a", active_only=False).matches(metadata)


def test_es_document_omits_empty_date_fields() -> None:
    document = _metadata().to_es_document("hello")

    assert document["chunk_text"] == "hello"
    assert "created_at" not in document
    assert "updated_at" not in document
