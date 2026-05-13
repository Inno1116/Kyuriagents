from __future__ import annotations

from typing import Any, Self

from deepagents.tools import PostgresToolAuditSink, ToolCallRecord


class _FakeCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params: object = None) -> object:
        self.calls.append((query, params))
        return self

    def fetchone(self) -> dict[str, object] | None:
        return None

    def fetchall(self) -> list[dict[str, object]]:
        return []


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self.cursor_obj = cursor
        self.cursor_kwargs: dict[str, object] = {}

    def cursor(self, **kwargs: Any) -> _FakeCursor:
        self.cursor_kwargs = kwargs
        return self.cursor_obj


def test_postgres_tool_audit_sink_inserts_record() -> None:
    cursor = _FakeCursor()
    sink = PostgresToolAuditSink(connection=_FakeConnection(cursor))

    sink.record(
        ToolCallRecord(
            call_id="call-1",
            tenant_id="tenant-a",
            user_id="user-1",
            thread_id="thread-1",
            tool_name="search",
            source="native",
            risk="read_only",
            status="success",
            input_summary='{"query": "postgres"}',
        )
    )

    query, params = cursor.calls[0]
    assert "INSERT INTO agent_tool_calls" in query
    assert params["call_id"] == "call-1"
    assert params["tenant_id"] == "tenant-a"
    assert params["tool_name"] == "search"
