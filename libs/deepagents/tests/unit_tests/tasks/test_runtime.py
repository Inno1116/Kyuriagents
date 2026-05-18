from __future__ import annotations

import time
from typing import TYPE_CHECKING

import deepagents.memory as memory_module
from deepagents.middleware.retrieval import RuntimeContextDefaults
from deepagents.runtime import AgentRuntimeConfig
from deepagents.tasks import InMemoryTaskStore, LLMPlanner, TaskRuntime, TaskRuntimeLimits, TaskStepExecutor, TaskToolExecutor
from deepagents.tasks.runtime import TaskExecutionContext, _memory_handler
from deepagents.tools import ToolDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest

    from deepagents.memory import MemoryScope


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
    assert [step.kind for step in result.steps] == ["tool", "think", "answer"]
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
    assert [step.kind for step in result.steps] == ["think", "answer"]


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
    assert [step.kind for step in result.steps[-2:]] == ["think", "answer"]


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
