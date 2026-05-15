from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from deepagents.memory import MemoryRecord, MemoryScope, PostgresMemoryStore


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.fetchone_row: dict[str, object] | None = None
        self.fetchall_rows: list[dict[str, object]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params: object = None) -> object:
        self.calls.append((query, params))
        if "INSERT INTO agent_memory_items" in query:
            row = dict(params or {})
            row["source_message_ids"] = row["source_message_ids"].obj
            row["tags"] = row["tags"].obj
            row["metadata"] = {}
            self.fetchone_row = row
        return self

    def fetchone(self) -> dict[str, object] | None:
        return self.fetchone_row

    def fetchall(self) -> list[dict[str, object]]:
        return self.fetchall_rows


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self.cursor_obj = cursor

    def cursor(self, **kwargs: Any) -> _FakeCursor:
        self.cursor_kwargs = kwargs
        return self.cursor_obj


def _memory() -> MemoryRecord:
    now = datetime.now(tz=UTC).isoformat()
    return MemoryRecord(
        memory_id="mem-1",
        tenant_id="tenant-a",
        user_id="user-1",
        scope_type="user",
        scope_id="user-1",
        memory_type="preference",
        content="The user prefers concise deployment notes.",
        tags=("deploy",),
        created_at=now,
        updated_at=now,
    )


def test_postgres_memory_store_upserts_and_maps_returned_row() -> None:
    cursor = _FakeCursor()
    store = PostgresMemoryStore(connection=_FakeConnection(cursor))

    saved = store.upsert(_memory())

    query, params = cursor.calls[0]
    assert "ON CONFLICT (memory_id) DO UPDATE" in query
    assert params["memory_id"] == "mem-1"
    assert params["tags"].obj == ["deploy"]
    assert params["source_message_ids"].obj == []
    assert saved.memory_id == "mem-1"
    assert saved.tags == ("deploy",)


def test_postgres_memory_store_search_applies_scope_filters() -> None:
    cursor = _FakeCursor()
    cursor.fetchall_rows = [
        {
            **_memory().__dict__,
            "source_message_ids": [],
            "search_score": 0.88,
        }
    ]
    store = PostgresMemoryStore(connection=_FakeConnection(cursor))

    results = store.search(
        "deployment",
        scope=MemoryScope(tenant_id="tenant-a", user_id="user-1", tags=("deploy",)),
        limit=3,
    )

    query, params = cursor.calls[0]
    assert "tenant_id = %(tenant_id)s" in query
    assert "tags ?& %(tags)s" in query
    assert "ILIKE ANY(%(like_terms)s)" in query
    assert params["tenant_id"] == "tenant-a"
    assert params["user_id"] == "user-1"
    assert params["tags"] == ["deploy"]
    assert params["like_terms"] == ["%deployment%"]
    assert results[0].memory_id == "mem-1"
    assert results[0].score == 0.88


def test_postgres_memory_store_soft_delete_returns_boolean() -> None:
    cursor = _FakeCursor()
    cursor.fetchone_row = {"memory_id": "mem-1"}
    store = PostgresMemoryStore(connection=_FakeConnection(cursor))

    assert store.delete("mem-1", scope=MemoryScope(tenant_id="tenant-a", user_id="user-1"))

    query, params = cursor.calls[0]
    assert "SET status = 'deleted'" in query
    assert params["memory_id"] == "mem-1"
