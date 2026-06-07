from __future__ import annotations

import time
from typing import TYPE_CHECKING

import deepagents.memory as memory_module
from deepagents.middleware.retrieval import RuntimeContextDefaults
from deepagents.runtime import AgentRuntimeConfig
from deepagents.runtime.evidence import EvidenceFinding, EvidencePackage, EvidenceSource
from deepagents.tasks import ContextBuilder, InMemoryTaskStore, LLMPlanner, TaskRuntime, TaskRuntimeLimits, TaskStepExecutor, TaskToolExecutor
from deepagents.tasks.runtime import TaskExecutionContext, _memory_handler
from deepagents.tasks.store import new_step_record
from deepagents.tools import InMemoryToolAuditSink, ToolDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest

    from deepagents.memory import MemoryScope
    from deepagents.tasks.evidence import EvidenceRequest


class FakePlanModel:
    """Model double that returns a plan without a final answer step."""

    def invoke(self, input_data: object) -> object:
        _ = input_data
        return """
        {
          "goal": "summarize the uploaded document",
          "summary": "Model forgot the answer step.",
          "steps": [
            {
              "kind": "think",
              "title": "Read notes",
              "instruction": "Organize the available context.",
              "tool_name": "",
              "input": {},
              "depends_on": [],
              "parallel_group": ""
            }
          ]
        }
        """


class FakeEvidenceAgent:
    """Evidence-agent double used by task step executor tests."""

    descriptor = ToolDescriptor(name="web_agent", description="Fake web evidence.", risk="external_read", source="runtime")

    def run(self, request: EvidenceRequest) -> EvidencePackage:
        return EvidencePackage(
            conclusion=f"evidence:{request.query}",
            findings=[EvidenceFinding(claim="found current source", source_indices=[0], confidence=0.8)],
            sources=[EvidenceSource(title="Example", url="https://example.com", source_type="web", quote="source quote")],
        )


class EvidencePlanModel:
    """Model double that returns an evidence-first task plan."""

    def invoke(self, input_data: object) -> object:
        _ = input_data
        return """
        {
          "goal": "research current events",
          "summary": "Use web evidence, then answer.",
          "steps": [
            {
              "kind": "web",
              "title": "Research web evidence",
              "instruction": "Find current web evidence.",
              "tool_name": "",
              "input": {"query": "current events"},
              "depends_on": [],
              "parallel_group": ""
            },
            {
              "kind": "answer",
              "title": "Answer",
              "instruction": "Answer with evidence.",
              "tool_name": "",
              "input": {},
              "depends_on": [],
              "parallel_group": ""
            }
          ]
        }
        """


def test_task_runtime_runs_plan_tool_and_answer_steps() -> None:
    """Task runtime should persist a small plan and execute available tools."""

    def search_handler(_context: object, input_data: Mapping[str, object]) -> str:
        return f"found:{input_data['query']}"

    runtime = TaskRuntime(
        store=InMemoryTaskStore(),
        planner=LLMPlanner(),
        executor=TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(
                handlers={"search_knowledge_base": search_handler},
                descriptors=(ToolDescriptor(name="search_knowledge_base", risk="read_only"),),
            ),
        ),
    )

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="summarize the uploaded document",
    )

    assert result.task.status == "succeeded"
    assert [step.kind for step in result.steps] == ["tool", "process", "answer"]
    assert result.steps[0].output == "found:summarize the uploaded document"
    assert "found:summarize the uploaded document" in result.final_answer
    assert [event.event_type for event in result.events] == [
        "created",
        "intent",
        "context",
        "planned",
        "validated",
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
        "step_started",
        "step_finished",
        "finished",
    ]


def test_task_runtime_executes_evidence_steps() -> None:
    """Evidence steps should run through evidence agents and store packages."""
    runtime = TaskRuntime(
        store=InMemoryTaskStore(),
        planner=LLMPlanner(model_factory=EvidencePlanModel),
        executor=TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(handlers={}, descriptors=()),
            evidence_agents={"web": FakeEvidenceAgent()},
        ),
    )

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="research current events",
    )

    assert result.task.status == "succeeded"
    assert [step.kind for step in result.steps] == ["web", "answer"]
    assert "<evidence_package>" in result.steps[0].output
    assert "evidence:current events" in result.steps[0].output
    assert "https://example.com" in result.final_answer


def test_task_step_executor_hides_raw_tools_when_evidence_agent_exists() -> None:
    """Planner should see high-level evidence agents instead of raw web tools."""
    executor = TaskStepExecutor(
        model_factory=None,
        tool_executor=TaskToolExecutor(
            handlers={},
            descriptors=(
                ToolDescriptor(name="web_search", risk="external_read", source="runtime"),
                ToolDescriptor(name="web_research", risk="external_read", source="runtime"),
                ToolDescriptor(name="search_memory", risk="read_only"),
            ),
        ),
        evidence_agents={"web": FakeEvidenceAgent()},
    )

    assert [descriptor.name for descriptor in executor.tool_descriptors] == ["search_memory", "web_agent"]


def test_heuristic_plan_prefers_evidence_steps() -> None:
    """Fallback planning should use evidence step kinds when available."""
    context = ContextBuilder().build(
        goal="research latest CS2 Major news online",
        intent="task",
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        tool_descriptors=(
            ToolDescriptor(name="rag_agent", risk="read_only", source="runtime"),
            ToolDescriptor(name="web_agent", risk="external_read", source="runtime"),
        ),
    )
    plan = LLMPlanner().plan(context)

    assert [step.kind for step in plan.steps] == ["rag", "web", "process", "answer"]


def test_task_runtime_hides_disabled_tools_from_planner() -> None:
    """Disabled tools should not appear in the generated fallback plan."""
    runtime = TaskRuntime(
        store=InMemoryTaskStore(),
        planner=LLMPlanner(),
        executor=TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(
                handlers={"search_knowledge_base": lambda _context, _input: "found"},
                descriptors=(ToolDescriptor(name="search_knowledge_base", risk="read_only"),),
            ),
        ),
    )

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="summarize the uploaded document",
        disabled_tools=("search_knowledge_base",),
    )

    assert result.task.status == "succeeded"
    assert [step.kind for step in result.steps] == ["process", "answer"]


def test_task_tool_executor_records_audit() -> None:
    """Task-mode tools should share the same audit surface as chat tools."""

    def search_handler(_context: object, input_data: Mapping[str, object]) -> str:
        return f"found:{input_data['query']}"

    sink = InMemoryToolAuditSink()
    executor = TaskToolExecutor(
        handlers={"web_search": search_handler},
        descriptors=(ToolDescriptor(name="web_search", risk="external_read", source="runtime"),),
        audit_sink=sink,
    )
    context = TaskExecutionContext(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="search web",
        defaults=RuntimeContextDefaults(tenant_id="tenant_1", user_id="user_1", thread_id="thread_1"),
    )
    step = new_step_record(
        task_id="task_1",
        step_index=0,
        kind="tool",
        title="Search web",
        instruction="Search.",
        tool_name="web_search",
        input={"query": "Kyuriagents"},
    )

    assert executor.execute(step, context) == "found:Kyuriagents"

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.tool_name == "web_search"
    assert record.thread_id == "thread_1"
    assert record.status == "success"
    assert "Kyuriagents" in record.input_summary
    assert "found:Kyuriagents" in record.output_summary
    assert record.metadata["task_mode"] is True


def test_task_runtime_adds_answer_step_when_model_plan_omits_it() -> None:
    """Runtime should repair model plans that forget the final answer."""
    runtime = TaskRuntime(store=InMemoryTaskStore(), planner=LLMPlanner(model_factory=FakePlanModel))

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="summarize the uploaded document",
    )

    assert result.task.status == "succeeded"
    assert [step.kind for step in result.steps] == ["think", "answer"]
    assert result.steps[-1].title == "Generate final answer"


def test_task_runtime_replans_after_tool_failure() -> None:
    """Tool failures should replan once and skip stale remaining steps."""

    def failing_search(_context: object, _input_data: Mapping[str, object]) -> str:
        msg = "Elasticsearch connection error"
        raise ConnectionError(msg)

    runtime = TaskRuntime(
        store=InMemoryTaskStore(),
        planner=LLMPlanner(),
        executor=TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(
                handlers={"search_knowledge_base": failing_search},
                descriptors=(ToolDescriptor(name="search_knowledge_base", risk="read_only"),),
            ),
        ),
        limits=TaskRuntimeLimits(max_step_retries=1, max_replans=1, max_same_error=2),
    )

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="summarize the uploaded document",
    )

    assert result.task.status == "succeeded"
    assert "replanned" in [event.event_type for event in result.events]
    assert [step.status for step in result.steps] == ["failed", "skipped", "skipped", "succeeded", "succeeded"]
    assert [step.kind for step in result.steps[-2:]] == ["process", "answer"]


def test_task_runtime_times_out_and_skips_read_only_tool() -> None:
    """Read-only tools should timeout, retry, then skip without looping forever."""

    def slow_search(_context: object, _input_data: Mapping[str, object]) -> str:
        time.sleep(0.05)
        return "late result"

    runtime = TaskRuntime(
        store=InMemoryTaskStore(),
        planner=LLMPlanner(),
        executor=TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(
                handlers={"search_knowledge_base": slow_search},
                descriptors=(ToolDescriptor(name="search_knowledge_base", risk="read_only"),),
                timeout_seconds=0.01,
            ),
        ),
        limits=TaskRuntimeLimits(max_step_retries=1, max_replans=0, tool_timeout_seconds=0.01),
    )

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="summarize the uploaded document",
    )

    assert result.task.status == "succeeded"
    assert result.steps[0].status == "skipped"
    assert result.steps[0].attempts == 2
    assert "timed out" in (result.steps[0].error_message or "")
    assert "retry" in [event.event_type for event in result.events]


def test_memory_handler_reuses_cached_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated memory tool calls should keep `MemoryScope` available."""

    class FakePostgresMemoryStore:
        def __init__(self, *, dsn: str) -> None:
            self.dsn = dsn

    class FakeMemoryService:
        def __init__(self, store: FakePostgresMemoryStore) -> None:
            self.store = store

        def build_context(self, query: str, *, scope: MemoryScope, limit: int) -> str:
            return f"{query}:{scope.user_id}:{limit}:{self.store.dsn}"

    monkeypatch.setattr(memory_module, "MemoryService", FakeMemoryService)
    monkeypatch.setattr(memory_module, "PostgresMemoryStore", FakePostgresMemoryStore)
    handler = _memory_handler(AgentRuntimeConfig(postgres_dsn="postgresql://example"))
    context = TaskExecutionContext(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="remember my preferences",
        defaults=RuntimeContextDefaults(tenant_id="tenant_1", user_id="user_1", thread_id="thread_1"),
    )

    assert handler(context, {"query": "first", "top_k": 2}) == "first:user_1:2:postgresql://example"
    assert handler(context, {"query": "second", "top_k": 3}) == "second:user_1:3:postgresql://example"
