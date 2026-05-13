from __future__ import annotations

from deepagents.memory import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemoryWriteCandidate,
    format_memory_context,
)


def _memory(memory_id: str, *, tenant_id: str = "tenant-a", user_id: str | None = "user-1") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        tenant_id=tenant_id,
        user_id=user_id,
        scope_type="user",
        scope_id=user_id or "tenant",
        memory_type="workflow",
        content="Use PostgreSQL for relational metadata and Milvus for vectors.",
        tags=("rag", "postgres"),
        importance=0.9,
    )


def test_in_memory_store_searches_visible_memories() -> None:
    store = InMemoryMemoryStore(
        [
            _memory("mem-visible"),
            _memory("mem-other-user", user_id="user-2"),
            _memory("mem-other-tenant", tenant_id="tenant-b"),
        ]
    )

    results = store.search(
        "postgres vectors",
        scope=MemoryScope(tenant_id="tenant-a", user_id="user-1"),
        limit=5,
    )

    assert [result.memory_id for result in results] == ["mem-visible"]
    assert results[0].lexical_score is not None


def test_in_memory_store_soft_deletes_visible_memory() -> None:
    store = InMemoryMemoryStore([_memory("mem-delete")])
    scope = MemoryScope(tenant_id="tenant-a", user_id="user-1")

    assert store.delete("mem-delete", scope=scope)
    assert store.get("mem-delete", scope=scope) is None
    assert store.get("mem-delete", scope=MemoryScope(tenant_id="tenant-a", user_id="user-1", active_only=False)) is not None


def test_memory_service_stamps_candidate_metadata() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)

    record = service.save_candidate(
        MemoryWriteCandidate(
            scope_type="user",
            scope_id="user-1",
            memory_type="preference",
            content="The user prefers concrete deployment artifacts.",
            source_thread_id="thread-1",
            source_message_ids=("msg-1",),
        ),
        tenant_id="tenant-a",
        user_id="user-1",
        memory_id="mem-fixed",
    )

    assert record.memory_id == "mem-fixed"
    assert record.tenant_id == "tenant-a"
    assert record.user_id == "user-1"
    assert record.created_at
    assert record.updated_at
    assert service.search("deployment artifacts", scope=MemoryScope(tenant_id="tenant-a", user_id="user-1"))[0].memory_id == "mem-fixed"


def test_memory_context_formats_retrieved_top_k() -> None:
    store = InMemoryMemoryStore([_memory("mem-context")])
    service = MemoryService(store)

    context = service.build_context(
        "postgres metadata",
        scope=MemoryScope(tenant_id="tenant-a", user_id="user-1"),
    )

    assert context.startswith("<agent_long_term_memory>")
    assert "workflow" in context
    assert "Use PostgreSQL for relational metadata" in context


def test_empty_memory_context_is_explicit() -> None:
    assert "No relevant long-term memory" in format_memory_context([])
