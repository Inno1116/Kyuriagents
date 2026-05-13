"""High-level helpers for dynamic memory operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from deepagents.memory.types import MemoryRecord, MemoryScope, MemorySearchResult, MemoryStore, MemoryWriteCandidate


def format_memory_context(results: list[MemorySearchResult]) -> str:
    """Format retrieved memories for prompt injection.

    Args:
        results: Ranked memory search results.

    Returns:
        XML-like context block containing only relevant memories.
    """
    if not results:
        return "<agent_long_term_memory>\n(No relevant long-term memory found.)\n</agent_long_term_memory>"

    lines = ["<agent_long_term_memory>"]
    for result in results:
        memory = result.memory
        text = memory.summary or memory.content
        lines.append(
            f"- [{memory.memory_type}; scope={memory.scope_type}/{memory.scope_id}; "
            f"confidence={memory.confidence:.2f}; importance={memory.importance:.2f}] {text}"
        )
    lines.append("</agent_long_term_memory>")
    return "\n".join(lines)


class MemoryService:
    """Small orchestration layer around a `MemoryStore`.

    LangMem or another extractor can produce `MemoryWriteCandidate` instances;
    this service stamps tenant, owner, and audit metadata before persistence.
    """

    def __init__(self, store: MemoryStore) -> None:
        """Initialize the service.

        Args:
            store: Durable memory store.
        """
        self._store = store

    def save_candidate(
        self,
        candidate: MemoryWriteCandidate,
        *,
        tenant_id: str,
        user_id: str | None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        """Persist one extracted memory candidate.

        Args:
            candidate: Proposed memory item.
            tenant_id: Tenant or organization identifier.
            user_id: Optional owner for private user memory.
            memory_id: Optional stable identifier supplied by an upstream store.

        Returns:
            Persisted memory record.
        """
        now = datetime.now(tz=UTC).isoformat()
        record = MemoryRecord(
            memory_id=memory_id or f"mem_{uuid.uuid4().hex}",
            tenant_id=tenant_id,
            user_id=user_id,
            scope_type=candidate.scope_type,
            scope_id=candidate.scope_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            summary=candidate.summary,
            visibility=candidate.visibility,
            confidence=candidate.confidence,
            importance=candidate.importance,
            tags=candidate.tags,
            source_thread_id=candidate.source_thread_id,
            source_message_ids=candidate.source_message_ids,
            created_at=now,
            updated_at=now,
            expires_at=candidate.expires_at,
        )
        return self._store.upsert(record)

    def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Search memory records visible to a caller.

        Args:
            query: User query or rewritten query.
            scope: Tenant and authorization filters.
            limit: Maximum number of records to return.

        Returns:
            Ranked memory results.
        """
        return self._store.search(query, scope=scope, limit=limit)

    def build_context(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 5,
    ) -> str:
        """Retrieve relevant memories and format them for prompt injection.

        Args:
            query: User query or rewritten query.
            scope: Tenant and authorization filters.
            limit: Maximum number of memories to include.

        Returns:
            Prompt-ready long-term memory context block.
        """
        return format_memory_context(self.search(query, scope=scope, limit=limit))

    def get(self, memory_id: str, *, scope: MemoryScope) -> MemoryRecord | None:
        """Load one visible memory record.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            Matching record, or `None`.
        """
        return self._store.get(memory_id, scope=scope)

    def delete(self, memory_id: str, *, scope: MemoryScope) -> bool:
        """Soft delete one visible memory record.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            `True` when a record was updated.
        """
        return self._store.delete(memory_id, scope=scope)
