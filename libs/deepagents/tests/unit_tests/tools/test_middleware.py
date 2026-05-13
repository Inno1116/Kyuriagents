from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from deepagents.tools import (
    InMemoryToolAuditSink,
    ToolContextDefaults,
    ToolDescriptor,
    ToolGovernanceMiddleware,
    ToolPolicy,
    ToolRegistry,
)


def _runtime() -> ToolRuntime:
    return ToolRuntime(
        state={},
        context={"tenant_id": "tenant-a", "user_id": "user-1", "thread_id": "thread-1"},
        tool_call_id="call-1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _request(name: str = "search") -> ToolCallRequest:
    return ToolCallRequest(
        runtime=_runtime(),
        tool_call={"id": "call-1", "name": name, "args": {"query": "postgres"}},
        state={},
        tool=None,
    )


def test_tool_governance_blocks_disallowed_tool_and_records_audit() -> None:
    registry = ToolRegistry()
    registry.register(ToolDescriptor(name="danger", risk="destructive"))
    sink = InMemoryToolAuditSink()
    middleware = ToolGovernanceMiddleware(registry=registry, audit_sink=sink)

    result = middleware.wrap_tool_call(
        _request("danger"),
        lambda _request: ToolMessage(content="ran", tool_call_id="call-1", name="danger"),
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "disallowed risk" in str(result.content)
    assert sink.records[0].status == "blocked"
    assert sink.records[0].tenant_id == "tenant-a"
    assert sink.records[0].tool_name == "danger"


def test_tool_governance_allows_and_audits_successful_call() -> None:
    registry = ToolRegistry()
    registry.register(ToolDescriptor(name="search", risk="read_only"))
    sink = InMemoryToolAuditSink()
    middleware = ToolGovernanceMiddleware(
        registry=registry,
        policy=ToolPolicy(allowed_risks=frozenset({"read_only"})),
        audit_sink=sink,
        defaults=ToolContextDefaults(tenant_id="fallback"),
    )

    result = middleware.wrap_tool_call(
        _request(),
        lambda _request: ToolMessage(content="found", tool_call_id="call-1", name="search"),
    )

    assert isinstance(result, ToolMessage)
    assert result.content == "found"
    assert sink.records[0].status == "success"
    assert sink.records[0].input_summary == '{"query": "postgres"}'
    assert sink.records[0].output_summary
