from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from deepagents.memory import MemoryRecord, MemoryScope


def _memory(
    *,
    memory_id: str = "mem-1",
    tenant_id: str = "tenant-a",
    user_id: str | None = "user-1",
    visibility: str = "private",
    status: str = "active",
    expires_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        tenant_id=tenant_id,
        user_id=user_id,
        scope_type="user",
        scope_id=user_id or "tenant",
        memory_type="preference",
        content="The user prefers concise Python examples.",
        tags=("python",),
        visibility=visibility,
        status=status,
        expires_at=expires_at,
    )


def test_memory_scope_enforces_tenant_and_private_owner() -> None:
    memory = _memory()

    assert MemoryScope(tenant_id="tenant-a", user_id="user-1").matches(memory)
    assert not MemoryScope(tenant_id="tenant-a", user_id="user-2").matches(memory)
    assert not MemoryScope(tenant_id="tenant-a").matches(memory)
    assert not MemoryScope(tenant_id="tenant-b", user_id="user-1").matches(memory)


def test_memory_scope_filters_status_and_expiration() -> None:
    expired = datetime.now(tz=UTC) - timedelta(days=1)

    assert not MemoryScope(tenant_id="tenant-a", user_id="user-1").matches(_memory(expires_at=expired.isoformat()))
    assert not MemoryScope(tenant_id="tenant-a", user_id="user-1").matches(_memory(status="superseded"))
    assert MemoryScope(tenant_id="tenant-a", user_id="user-1", active_only=False).matches(_memory(status="superseded"))


def test_memory_record_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="confidence"):
        MemoryRecord(
            memory_id="mem-bad",
            tenant_id="tenant-a",
            scope_type="user",
            scope_id="user-1",
            memory_type="fact",
            content="Important fact",
            confidence=1.5,
        )


def test_memory_record_converts_to_rag_chunk() -> None:
    memory = _memory()

    chunk = memory.to_document_chunk(embedding=(0.1, 0.2))

    assert chunk.text == memory.index_text
    assert chunk.embedding == (0.1, 0.2)
    assert chunk.metadata.chunk_id == "memory:mem-1"
    assert chunk.metadata.source_type == "memory"
    assert chunk.metadata.tenant_id == "tenant-a"
    assert chunk.metadata.user_id == "user-1"
    assert chunk.metadata.kb_id == "memory:user:user-1"
    assert "memory" in chunk.metadata.tags
    assert "preference" in chunk.metadata.tags
