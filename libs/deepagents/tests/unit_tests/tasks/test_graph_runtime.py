from __future__ import annotations

from typing import TYPE_CHECKING, cast

from deepagents.runtime import AgentRuntimeConfig
from deepagents.server.identity import MessageRecord
from deepagents.tasks import (
    ClarificationDecision,
    ClarificationJudge,
    ContextBuilder,
    GraphTaskRuntime,
    InMemoryTaskStore,
    LLMPlanner,
    TaskContext,
    TaskRuntime,
    TaskRuntimeLimits,
    TaskStepExecutor,
    TaskToolExecutor,
)
from deepagents.tools import ToolDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_task_context_keeps_recent_history_with_current_goal_authoritative() -> None:
    """Task planning context should keep history while preserving the current goal."""
    context = ContextBuilder().build(
        goal="Plan Chengdu for 3 days",
        intent="task",
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        messages=(
            MessageRecord(message_id="msg_1", tenant_id="tenant_1", thread_id="thread_1", role="user", content="Plan Osaka and Kyoto."),
            MessageRecord(
                message_id="msg_2",
                tenant_id="tenant_1",
                thread_id="thread_1",
                role="assistant",
                content="Long Osaka and Kyoto itinerary with temples, anime, and food.",
            ),
        ),
    )

    assert context.goal == "Plan Chengdu for 3 days"
    assert context.recent_messages == (
        {"role": "user", "content": "Plan Osaka and Kyoto."},
        {"role": "assistant", "content": "Long Osaka and Kyoto itinerary with temples, anime, and food."},
    )


def test_task_runtime_from_config_uses_graph_runtime_by_default() -> None:
    """Config factory should switch task mode to the graph runtime by default."""
    config = AgentRuntimeConfig(enable_rag=False, enable_memory=False)

    runtime = TaskRuntime.from_config(config)

    assert isinstance(runtime, GraphTaskRuntime)


def test_task_runtime_from_config_can_disable_graph_runtime() -> None:
    """Config factory should keep a rollback path to the legacy loop."""
    config = AgentRuntimeConfig(enable_rag=False, enable_memory=False, enable_task_graph_runtime=False)

    runtime = TaskRuntime.from_config(config)

    assert isinstance(runtime, TaskRuntime)
    assert not isinstance(runtime, GraphTaskRuntime)


def test_graph_task_runtime_runs_plan_tool_and_answer_steps() -> None:
    """Graph runtime should plan, execute, observe, and finish a small task."""

    def search_handler(_context: object, input_data: Mapping[str, object]) -> str:
        return f"found:{input_data['query']}"

    runtime = GraphTaskRuntime(
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


def test_graph_task_runtime_can_wait_for_user_clarification() -> None:
    """Graph runtime should stop before planning when the goal is underspecified."""

    class FakeClarificationJudge(ClarificationJudge):
        def judge(self, context: TaskContext) -> ClarificationDecision:
            assert context.goal == "帮我规划一下"
            return ClarificationDecision(need_user_input=True, question="你准备去哪里旅行?", reason="missing_destination")

    runtime = GraphTaskRuntime(
        store=InMemoryTaskStore(),
        planner=LLMPlanner(),
        clarification_judge=FakeClarificationJudge(),
    )

    result = runtime.run(
        tenant_id="tenant_1",
        user_id="user_1",
        thread_id="thread_1",
        goal="帮我规划一下",
    )

    assert result.task.status == "waiting_user"
    assert result.final_answer == "你准备去哪里旅行?"
    assert result.steps == ()
    assert result.task.metadata["clarification_reason"] == "missing_destination"
    assert [event.event_type for event in result.events] == ["created", "intent", "context", "hitl_requested"]
    assert result.task.metadata["original_goal"] == "帮我规划一下"
    hitl = cast("dict[str, object]", result.task.metadata["hitl"])
    assert hitl["status"] == "waiting_user"


def test_graph_task_runtime_replans_after_tool_failure() -> None:
    """Graph runtime should disable a failed tool path and continue with a new plan."""

    def failing_search(_context: object, _input_data: Mapping[str, object]) -> str:
        msg = "Elasticsearch connection error"
        raise ConnectionError(msg)

    runtime = GraphTaskRuntime(
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
