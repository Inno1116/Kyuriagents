"""Task-mode runtime with intent routing, planning, execution, and observation."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage

from deepagents.middleware.retrieval import RuntimeContextDefaults, format_rag_context
from deepagents.rag import ElasticsearchKeywordStore, HybridRAGRetriever, MilvusVectorStore, RetrievalScope
from deepagents.tasks.store import InMemoryTaskStore, PostgresTaskStore, TaskStore, new_step_record
from deepagents.tasks.types import (
    PlannedStep,
    StepObservation,
    TaskContext,
    TaskEventRecord,
    TaskIntent,
    TaskPlan,
    TaskRecord,
    TaskStepKind,
    TaskStepRecord,
    ValidationResult,
)
from deepagents.tools import ToolDescriptor, ToolRegistry, default_tool_registry

if TYPE_CHECKING:
    from deepagents.runtime import AgentRuntimeConfig
    from deepagents.server.identity import MessageRecord

_LOGGER = logging.getLogger(__name__)
_MAX_STEPS = 8
_TOOL_THREAD_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="deepagents-task-tool")
_TASK_HINT_RE = re.compile(
    "(?i)(help me|research|analyze|summarize|compare|draft|create|plan|investigate|"
    "\\u6574\\u7406|\\u8c03\\u7814|\\u5206\\u6790|\\u603b\\u7ed3|\\u751f\\u6210|\\u5b8c\\u6210|\\u8ba1\\u5212|\\u67e5\\u627e)"
)
_RAG_HINT_RE = re.compile(
    "(?i)(knowledge base|document|pdf|paper|file|\\u8d44\\u6599|\\u77e5\\u8bc6\\u5e93|\\u6587\\u6863|\\u8bba\\u6587|\\u6587\\u4ef6|\\u6839\\u636e)"
)
_MEMORY_HINT_RE = re.compile("(?i)(memory|remember|preference|my name|\\u8bb0\\u5fc6|\\u8bb0\\u4f4f|\\u504f\\u597d|\\u6211\\u7684\\u540d\\u5b57)")


class _Model(Protocol):
    def invoke(self, input_data: object) -> object:
        """Invoke a chat model."""
        ...


ModelFactory = Callable[[], _Model]
ToolHandler = Callable[["TaskExecutionContext", Mapping[str, object]], str]


@dataclass(frozen=True, kw_only=True)
class TaskRuntimeLimits:
    """Hard safety limits for task-mode planning and execution."""

    max_plan_steps: int = 8
    max_replans: int = 2
    max_total_steps: int = 12
    max_tool_calls: int = 8
    max_step_retries: int = 1
    max_runtime_seconds: float = 180.0
    max_same_error: int = 2
    max_step_output_chars: int = 4_000
    tool_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        """Validate limit values."""
        if self.max_plan_steps <= 0 or self.max_total_steps <= 0:
            msg = "Task step limits must be positive."
            raise ValueError(msg)
        if self.max_plan_steps > self.max_total_steps:
            msg = "`max_plan_steps` must not exceed `max_total_steps`."
            raise ValueError(msg)
        if self.max_step_retries < 0 or self.max_replans < 0:
            msg = "Retry and replan limits must not be negative."
            raise ValueError(msg)
        if self.max_tool_calls <= 0 or self.max_runtime_seconds <= 0 or self.tool_timeout_seconds <= 0:
            msg = "Runtime and tool limits must be positive."
            raise ValueError(msg)
        if self.max_same_error <= 0 or self.max_step_output_chars <= 0:
            msg = "Error and output limits must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class TaskExecutionContext:
    """Runtime context supplied to task tool handlers."""

    tenant_id: str
    user_id: str
    thread_id: str
    goal: str
    defaults: RuntimeContextDefaults


@dataclass(frozen=True, kw_only=True)
class TaskRunResult:
    """Result returned by a completed task run."""

    task: TaskRecord
    steps: tuple[TaskStepRecord, ...]
    events: tuple[TaskEventRecord, ...]
    final_answer: str


@dataclass(kw_only=True)
class _RunState:
    started_at: float
    replan_count: int = 0
    tool_calls: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class _StepOutcome:
    completed_step: TaskStepRecord | None = None
    replanned_steps: list[TaskStepRecord] = field(default_factory=list)
    final_answer: str = ""
    finish: bool = False


class IntentRouter:
    """Route a user input into chat, task, retrieval, or clarification intent."""

    def route(self, goal: str, *, forced_intent: TaskIntent | None = None) -> TaskIntent:
        """Route a goal.

        Args:
            goal: User goal text.
            forced_intent: Optional caller override.

        Returns:
            Routed intent.
        """
        if forced_intent is not None:
            return forced_intent
        text = goal.strip()
        if not text:
            return "clarify"
        if _TASK_HINT_RE.search(text):
            return "task"
        if _RAG_HINT_RE.search(text):
            return "rag_query"
        if _MEMORY_HINT_RE.search(text):
            return "memory_query"
        return "chat"


class ContextBuilder:
    """Build compact structured context for planning."""

    def __init__(self, *, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the builder.

        Args:
            tool_registry: Tool metadata source.
        """
        self._tool_registry = tool_registry or default_tool_registry()

    def build(
        self,
        *,
        goal: str,
        intent: TaskIntent,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        messages: Sequence[MessageRecord] = (),
        tool_descriptors: Sequence[ToolDescriptor] | None = None,
        constraints: Mapping[str, object] | None = None,
    ) -> TaskContext:
        """Build planner context."""
        descriptors = tuple(self._tool_registry.descriptors.values()) if tool_descriptors is None else tuple(tool_descriptors)
        return TaskContext(
            goal=goal,
            intent=intent,
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
            recent_messages=_recent_messages(messages),
            available_tools=tuple(_tool_dict(descriptor) for descriptor in descriptors),
            constraints={"language": "zh-CN", "max_steps": _MAX_STEPS, **dict(constraints or {})},
        )


class LLMPlanner:
    """Generate structured plans with an LLM and deterministic fallback."""

    def __init__(self, *, model_factory: ModelFactory | None = None) -> None:
        """Initialize the planner.

        Args:
            model_factory: Lazy chat model factory.
        """
        self._model_factory = model_factory

    def plan(self, context: TaskContext) -> TaskPlan:
        """Generate a task plan.

        Args:
            context: Structured planner context.

        Returns:
            Valid-looking plan, with semantic validation handled separately.
        """
        if self._model_factory is not None:
            try:
                payload = self._model_plan(context)
                plan = _plan_from_payload(payload, fallback_goal=context.goal)
                if plan.steps:
                    return plan
            except Exception as exc:  # noqa: BLE001  # planning falls back to deterministic heuristics
                _LOGGER.info("Task planner fell back to heuristic plan: %s", exc)
        return heuristic_plan(context)

    def _model_plan(self, context: TaskContext) -> Mapping[str, object]:
        if self._model_factory is None:
            msg = "Planner model factory is not configured."
            raise ValueError(msg)
        model = self._model_factory()
        prompt = (
            "You are the planner for Kyuriagents task mode. Return only JSON.\n"
            "Allowed step kinds: think, tool, answer.\n"
            "Only use tool names from available_tools. Prefer read-only tools before answering.\n"
            "Keep plans short and executable. Do not invent tools.\n"
            "Always include one final answer step as the last step.\n\n"
            f"Context:\n{json.dumps(_context_payload(context), ensure_ascii=False)}\n\n"
            "Return schema:\n"
            '{"goal": string, "summary": string, "steps": ['
            '{"kind": "think|tool|answer", "title": string, "instruction": string, '
            '"tool_name": string, "input": object, "depends_on": [string], "parallel_group": string}'
            "]}"
        )
        result = _invoke_model(model, [SystemMessage(content="Return valid JSON only."), HumanMessage(content=prompt)])
        payload = _json_payload(result)
        if not isinstance(payload, Mapping):
            msg = "Planner did not return a JSON object."
            raise TypeError(msg)
        return cast("Mapping[str, object]", payload)


def heuristic_plan(context: TaskContext) -> TaskPlan:
    """Build a conservative fallback plan from intent and tool availability."""
    tools = {str(tool.get("name")) for tool in context.available_tools}
    steps: list[PlannedStep] = []
    if context.intent in {"task", "rag_query"} and "search_knowledge_base" in tools:
        steps.append(
            PlannedStep(
                kind="tool",
                title="Search knowledge base",
                instruction="Search the user's uploaded and indexed knowledge base for relevant facts.",
                tool_name="search_knowledge_base",
                input={"query": context.goal, "top_k": 6},
                parallel_group="context_lookup",
            )
        )
    if context.intent in {"task", "memory_query"} and "search_memory" in tools:
        steps.append(
            PlannedStep(
                kind="tool",
                title="Search long-term memory",
                instruction="Search durable user and project memory for useful context.",
                tool_name="search_memory",
                input={"query": context.goal, "top_k": 5},
                parallel_group="context_lookup",
            )
        )
    if context.intent == "task":
        steps.append(
            PlannedStep(
                kind="think",
                title="Organize gathered information",
                instruction="Organize tool results and identify which facts support the final answer.",
            )
        )
    steps.append(
        PlannedStep(
            kind="answer",
            title="Generate final answer",
            instruction="Answer the user directly in Chinese, with evidence and any remaining uncertainty.",
        )
    )
    return TaskPlan(goal=context.goal, summary="Heuristic task plan.", steps=tuple(steps[:_MAX_STEPS]))


class PlanValidator:
    """Validate model-generated plans before execution."""

    def validate(self, plan: TaskPlan, context: TaskContext) -> ValidationResult:
        """Validate a task plan."""
        errors: list[str] = []
        max_steps = _int_value(context.constraints.get("max_steps"), default=_MAX_STEPS)
        if not plan.steps:
            errors.append("Plan must contain at least one step.")
        if len(plan.steps) > max_steps:
            errors.append(f"Plan has more than {max_steps} steps.")
        available = {str(tool.get("name")): tool for tool in context.available_tools}
        has_answer = False
        for index, step in enumerate(plan.steps):
            errors.extend(_validate_planned_step(step, index=index, available=available))
            if step.kind == "answer":
                has_answer = True
        if not has_answer:
            errors.append("Plan must end with or include an answer step.")
        return ValidationResult(valid=not errors, errors=tuple(errors))


def _validate_planned_step(step: PlannedStep, *, index: int, available: Mapping[str, Mapping[str, object]]) -> list[str]:
    errors: list[str] = []
    label = f"Step {index + 1}"
    if step.kind not in {"think", "tool", "answer"}:
        errors.append(f"{label} has unsupported kind `{step.kind}`.")
        return errors
    if step.kind != "tool":
        return errors
    if not step.tool_name:
        errors.append(f"{label} is missing `tool_name`.")
    elif step.tool_name not in available:
        errors.append(f"{label} references unavailable tool `{step.tool_name}`.")
    elif bool(available[step.tool_name].get("requires_confirmation")):
        errors.append(f"{label} requires confirmation, which task mode has not wired yet.")
    return errors


class TaskToolExecutor:
    """Execute task-mode tool steps through registered handlers."""

    def __init__(
        self,
        *,
        handlers: Mapping[str, ToolHandler],
        descriptors: Sequence[ToolDescriptor],
        timeout_seconds: float = 45.0,
    ) -> None:
        """Initialize the executor."""
        self._handlers = dict(handlers)
        self.descriptors = tuple(descriptors)
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> TaskToolExecutor:
        """Create default read-only task tools from runtime config."""
        handlers: dict[str, ToolHandler] = {}
        descriptors: list[ToolDescriptor] = []
        if config.enable_rag:
            handlers["search_knowledge_base"] = _rag_handler(config)
            descriptors.append(default_tool_registry().descriptor_for("search_knowledge_base"))
        if config.enable_memory and config.postgres_dsn:
            handlers["search_memory"] = _memory_handler(config)
            descriptors.append(default_tool_registry().descriptor_for("search_memory"))
        return cls(handlers=handlers, descriptors=descriptors)

    def execute(self, step: TaskStepRecord, context: TaskExecutionContext) -> str:
        """Execute one tool step."""
        handler = self._handlers.get(step.tool_name)
        if handler is None:
            msg = f"Task tool `{step.tool_name}` is not available."
            raise ValueError(msg)
        future: Future[str] = _TOOL_THREAD_POOL.submit(handler, context, step.input)
        try:
            return future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            msg = f"Task tool `{step.tool_name}` timed out after {self._timeout_seconds:.1f}s."
            raise TimeoutError(msg) from exc


class TaskStepExecutor:
    """Execute think, tool, and answer steps."""

    def __init__(self, *, model_factory: ModelFactory | None, tool_executor: TaskToolExecutor) -> None:
        """Initialize the executor."""
        self._model_factory = model_factory
        self._tool_executor = tool_executor

    @property
    def tool_descriptors(self) -> tuple[ToolDescriptor, ...]:
        """Return tool descriptors visible to the planner."""
        return self._tool_executor.descriptors

    def execute(self, step: TaskStepRecord, *, context: TaskExecutionContext, previous_steps: Sequence[TaskStepRecord]) -> str:
        """Execute one step."""
        if step.kind == "tool":
            return self._tool_executor.execute(step, context)
        if step.kind == "think":
            return self._model_step(
                step,
                context=context,
                previous_steps=previous_steps,
                purpose="Organize intermediate results into concise points.",
            )
        if step.kind == "answer":
            return self._model_step(
                step,
                context=context,
                previous_steps=previous_steps,
                purpose="Generate the final user-facing answer in Chinese.",
            )
        msg = f"Unsupported task step kind `{step.kind}`."
        raise ValueError(msg)

    def _model_step(
        self,
        step: TaskStepRecord,
        *,
        context: TaskExecutionContext,
        previous_steps: Sequence[TaskStepRecord],
        purpose: str,
    ) -> str:
        if self._model_factory is None:
            return _fallback_model_step(step, context=context, previous_steps=previous_steps)
        model = self._model_factory()
        prompt = (
            f"{purpose}\n\n"
            f"User goal: {context.goal}\n"
            f"Current step: {step.title}\n"
            f"Step instruction: {step.instruction}\n\n"
            f"Previous step results:\n{_previous_step_text(previous_steps)}"
        )
        messages = (
            SystemMessage(content="You are Kyuriagents task executor. Be concise and concrete."),
            HumanMessage(content=prompt),
        )
        return _invoke_model(model, list(messages))


class Observer:
    """Observe step outcomes and decide how execution should continue."""

    def observe(
        self,
        step: TaskStepRecord,
        *,
        error: Exception | None = None,
        limits: TaskRuntimeLimits | None = None,
        can_replan: bool = False,
        same_error_exceeded: bool = False,
    ) -> StepObservation:
        """Return the next execution decision."""
        resolved_limits = limits or TaskRuntimeLimits()
        if error is not None:
            if step.attempts <= resolved_limits.max_step_retries and not same_error_exceeded:
                return StepObservation(decision="retry", message=str(error))
            if step.kind == "tool" and can_replan and not same_error_exceeded:
                return StepObservation(decision="replan", message=str(error))
            if step.kind == "tool" and _can_skip_failed_tool(step):
                return StepObservation(decision="skip", message=str(error))
            return StepObservation(decision="fail", message=str(error))
        if step.kind == "answer":
            return StepObservation(decision="finish", message="Task has a final answer.")
        return StepObservation(decision="continue", message="Step succeeded.")


class TaskRuntime:
    """End-to-end task runtime."""

    def __init__(
        self,
        *,
        store: TaskStore | None = None,
        router: IntentRouter | None = None,
        context_builder: ContextBuilder | None = None,
        planner: LLMPlanner | None = None,
        validator: PlanValidator | None = None,
        executor: TaskStepExecutor | None = None,
        observer: Observer | None = None,
        limits: TaskRuntimeLimits | None = None,
    ) -> None:
        """Initialize the runtime."""
        self._limits = limits or TaskRuntimeLimits()
        self.store = store or InMemoryTaskStore()
        self._router = router or IntentRouter()
        self._context_builder = context_builder or ContextBuilder()
        self._planner = planner or LLMPlanner()
        self._validator = validator or PlanValidator()
        self._executor = executor or TaskStepExecutor(
            model_factory=None,
            tool_executor=TaskToolExecutor(handlers={}, descriptors=(), timeout_seconds=self._limits.tool_timeout_seconds),
        )
        self._observer = observer or Observer()

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> TaskRuntime:
        """Create the default runtime for deployed API servers."""

        def model_factory() -> _Model:
            from deepagents.runtime.dashscope import create_dashscope_model  # noqa: PLC0415

            return cast("_Model", create_dashscope_model(config))

        store: TaskStore = PostgresTaskStore(dsn=config.postgres_dsn) if config.postgres_dsn else InMemoryTaskStore()
        limits = TaskRuntimeLimits()
        tool_executor = TaskToolExecutor.from_config(config)
        return cls(
            store=store,
            planner=LLMPlanner(model_factory=model_factory),
            executor=TaskStepExecutor(model_factory=model_factory, tool_executor=tool_executor),
            context_builder=ContextBuilder(),
            limits=limits,
        )

    def run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        goal: str,
        title: str = "",
        messages: Sequence[MessageRecord] = (),
        forced_intent: TaskIntent | None = "task",
        disabled_tools: Sequence[str] = (),
    ) -> TaskRunResult:
        """Run a task synchronously through the full planning loop."""
        task = self.store.create_task(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id, goal=goal, title=title, intent="task")
        self.store.add_event(task_id=task.task_id, event_type="created", message="Task created.")
        return self.run_existing_task(task=task, messages=messages, forced_intent=forced_intent, disabled_tools=disabled_tools)

    def run_existing_task(
        self,
        *,
        task: TaskRecord,
        messages: Sequence[MessageRecord] = (),
        forced_intent: TaskIntent | None = "task",
        disabled_tools: Sequence[str] = (),
    ) -> TaskRunResult:
        """Run a previously created task through the planning loop.

        Args:
            task: Persisted task record that already has a `created` event.
            messages: Recent conversation messages used by the context builder.
            forced_intent: Optional intent override from the API layer.
            disabled_tools: Tool names hidden from the planner and executor.

        Returns:
            Final task run state, including steps, events, and final answer.
        """
        try:
            intent = self._router.route(task.goal, forced_intent=forced_intent)
            task = self.store.update_task(task.task_id, status="planning", intent=intent)
            self.store.add_event(task_id=task.task_id, event_type="intent", message=f"Intent routed as {intent}.")
            context = self._context_builder.build(
                goal=task.goal,
                intent=intent,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                thread_id=task.thread_id,
                messages=messages,
                tool_descriptors=_filter_descriptors(self._executor.tool_descriptors, disabled_tools=disabled_tools),
                constraints={"max_steps": self._limits.max_plan_steps, "max_tool_calls": self._limits.max_tool_calls},
            )
            self.store.add_event(task_id=task.task_id, event_type="context", message="Task context built.")
            plan = _normalize_plan(self._planner.plan(context), limits=self._limits)
            steps = self.store.add_steps(task_id=task.task_id, steps=_records_from_plan(task.task_id, plan))
            self.store.add_event(task_id=task.task_id, event_type="planned", message=plan.summary or "Plan generated.")
            validation = self._validator.validate(plan, context)
            if not validation.valid:
                message = "; ".join(validation.errors)
                task = self.store.update_task(task.task_id, status="failed", error_message=message, finished=True)
                self.store.add_event(task_id=task.task_id, event_type="failed", message=message)
                return self._result(task)
            self.store.add_event(task_id=task.task_id, event_type="validated", message="Plan validated.")
            task = self.store.update_task(task.task_id, status="running")
            final_answer = self._run_steps(
                task=task,
                context=context,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                thread_id=task.thread_id,
                goal=task.goal,
                steps=steps,
            )
            task = self.store.update_task(task.task_id, status="succeeded", final_answer=final_answer, finished=True)
            self.store.add_event(task_id=task.task_id, event_type="finished", message="Task completed.")
        except Exception as exc:  # noqa: BLE001  # task state must record unexpected runtime errors
            task = self.store.update_task(task.task_id, status="failed", error_message=str(exc), finished=True)
            self.store.add_event(task_id=task.task_id, event_type="failed", message=str(exc))
        return self._result(task)

    def _run_steps(
        self,
        *,
        task: TaskRecord,
        context: TaskContext,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        goal: str,
        steps: Sequence[TaskStepRecord],
    ) -> str:
        execution_context = TaskExecutionContext(
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
            goal=goal,
            defaults=RuntimeContextDefaults(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id),
        )
        state = _RunState(started_at=time.monotonic())
        final_answer = ""
        completed: list[TaskStepRecord] = []
        queue = list(steps)
        index = 0
        while index < len(queue):
            self._check_runtime_budget(state)
            step = queue[index]
            if step.kind == "tool" and state.tool_calls >= self._limits.max_tool_calls:
                skipped = self._skip_step_for_tool_limit(task=task, step=step)
                completed.append(skipped)
                index += 1
                continue
            outcome = self._execute_step(
                task=task,
                context=context,
                execution_context=execution_context,
                state=state,
                queue=queue,
                index=index,
                completed=completed,
            )
            if outcome.completed_step is not None:
                completed.append(outcome.completed_step)
            if outcome.final_answer:
                final_answer = outcome.final_answer
            if outcome.replanned_steps:
                self._skip_remaining(task=task, steps=queue[index + 1 :])
                queue = [*queue[: index + 1], *outcome.replanned_steps]
            if outcome.finish:
                break
            index += 1
        return final_answer or _previous_step_text(completed)

    def _execute_step(
        self,
        *,
        task: TaskRecord,
        context: TaskContext,
        execution_context: TaskExecutionContext,
        state: _RunState,
        queue: Sequence[TaskStepRecord],
        index: int,
        completed: Sequence[TaskStepRecord],
    ) -> _StepOutcome:
        current = self.store.update_step(queue[index].step_id, status="running", attempts=queue[index].attempts + 1, started=True)
        self.store.add_event(task_id=task.task_id, step_id=current.step_id, event_type="step_started", message=current.title)
        while True:
            try:
                output = self._execute_current_step(current, execution_context=execution_context, state=state, completed=completed)
            except Exception as exc:  # noqa: BLE001  # step failures must be observed and persisted
                outcome = self._handle_step_error(
                    task=task,
                    context=context,
                    execution_context=execution_context,
                    state=state,
                    current=current,
                    queue=queue,
                    index=index,
                    completed=completed,
                    error=exc,
                )
                if outcome is None:
                    current = self.store.update_step(current.step_id, attempts=current.attempts + 1, error_message=str(exc))
                    continue
                return outcome
            output = _truncate_text(output, max_chars=self._limits.max_step_output_chars)
            current = self.store.update_step(current.step_id, status="succeeded", output=output, error_message=None, finished=True)
            self.store.add_event(task_id=task.task_id, step_id=current.step_id, event_type="step_finished", message=current.title)
            observation = self._observer.observe(current)
            return _StepOutcome(
                completed_step=current, final_answer=output if observation.decision == "finish" else "", finish=observation.decision == "finish"
            )

    def _execute_current_step(
        self,
        current: TaskStepRecord,
        *,
        execution_context: TaskExecutionContext,
        state: _RunState,
        completed: Sequence[TaskStepRecord],
    ) -> str:
        self._check_runtime_budget(state)
        if current.kind == "tool":
            if state.tool_calls >= self._limits.max_tool_calls:
                return _raise_tool_call_limit(self._limits.max_tool_calls)
            state.tool_calls += 1
        return self._executor.execute(current, context=execution_context, previous_steps=completed)

    def _handle_step_error(
        self,
        *,
        task: TaskRecord,
        context: TaskContext,
        execution_context: TaskExecutionContext,
        state: _RunState,
        current: TaskStepRecord,
        queue: Sequence[TaskStepRecord],
        index: int,
        completed: Sequence[TaskStepRecord],
        error: Exception,
    ) -> _StepOutcome | None:
        error_count = self._record_error(state, error)
        observation = self._observer.observe(
            current,
            error=error,
            limits=self._limits,
            can_replan=self._can_replan(state, current=current, queue=queue, index=index),
            same_error_exceeded=error_count > self._limits.max_same_error,
        )
        if observation.decision == "retry":
            self.store.add_event(task_id=task.task_id, step_id=current.step_id, event_type="retry", message=observation.message)
            return None
        if observation.decision == "replan":
            failed = self.store.update_step(current.step_id, status="failed", error_message=str(error), finished=True)
            self.store.add_event(task_id=task.task_id, step_id=failed.step_id, event_type="failed", message=str(error))
            replanned = self._replan(state=state, task=task, context=context, failed_step=failed, completed=completed, error=error)
            if replanned:
                return _StepOutcome(replanned_steps=replanned)
            current = failed
        if observation.decision == "skip":
            return _StepOutcome(completed_step=self._skip_failed_step(task=task, step=current, message=observation.message, error=error))
        if current.kind == "answer":
            return self._fallback_answer_after_error(task=task, step=current, execution_context=execution_context, completed=completed, error=error)
        failed = self.store.update_step(current.step_id, status="failed", error_message=str(error), finished=True)
        self.store.add_event(task_id=task.task_id, step_id=failed.step_id, event_type="failed", message=str(error))
        raise error

    def _skip_step_for_tool_limit(self, *, task: TaskRecord, step: TaskStepRecord) -> TaskStepRecord:
        output = f"Skipped because the task reached the tool-call limit of {self._limits.max_tool_calls}."
        skipped = self.store.update_step(step.step_id, status="skipped", output=output, finished=True)
        self.store.add_event(task_id=task.task_id, step_id=skipped.step_id, event_type="skipped", message=skipped.output)
        return skipped

    def _skip_failed_step(self, *, task: TaskRecord, step: TaskStepRecord, message: str, error: Exception) -> TaskStepRecord:
        output = _truncate_text(f"Skipped after failure: {message}", max_chars=self._limits.max_step_output_chars)
        skipped = self.store.update_step(
            step.step_id,
            status="skipped",
            output=output,
            error_message=str(error),
            finished=True,
        )
        self.store.add_event(task_id=task.task_id, step_id=skipped.step_id, event_type="skipped", message=output)
        return skipped

    def _fallback_answer_after_error(
        self,
        *,
        task: TaskRecord,
        step: TaskStepRecord,
        execution_context: TaskExecutionContext,
        completed: Sequence[TaskStepRecord],
        error: Exception,
    ) -> _StepOutcome:
        output = _fallback_model_step(step, context=execution_context, previous_steps=completed)
        recovered = self.store.update_step(
            step.step_id,
            status="succeeded",
            output=_truncate_text(output, max_chars=self._limits.max_step_output_chars),
            error_message=str(error),
            finished=True,
        )
        self.store.add_event(
            task_id=task.task_id,
            step_id=recovered.step_id,
            event_type="step_finished",
            message="Fallback answer generated after executor failure.",
        )
        return _StepOutcome(completed_step=recovered, final_answer=recovered.output, finish=True)

    def _check_runtime_budget(self, state: _RunState) -> None:
        elapsed = time.monotonic() - state.started_at
        if elapsed > self._limits.max_runtime_seconds:
            msg = f"Task exceeded runtime limit of {self._limits.max_runtime_seconds:.1f}s."
            raise TimeoutError(msg)

    def _record_error(self, state: _RunState, error: Exception) -> int:
        key = _error_key(error)
        count = state.error_counts.get(key, 0) + 1
        state.error_counts[key] = count
        return count

    def _can_replan(self, state: _RunState, *, current: TaskStepRecord, queue: Sequence[TaskStepRecord], index: int) -> bool:
        if current.kind != "tool" or not current.tool_name:
            return False
        if state.replan_count >= self._limits.max_replans:
            return False
        existing_steps = len(self.store.list_steps(task_id=current.task_id))
        if existing_steps >= self._limits.max_total_steps:
            return False
        return any(step.status == "pending" for step in queue[index + 1 :])

    def _replan(
        self,
        *,
        state: _RunState,
        task: TaskRecord,
        context: TaskContext,
        failed_step: TaskStepRecord,
        completed: Sequence[TaskStepRecord],
        error: Exception,
    ) -> list[TaskStepRecord]:
        existing_steps = self.store.list_steps(task_id=task.task_id)
        remaining_budget = self._limits.max_total_steps - len(existing_steps)
        if remaining_budget <= 0:
            return []
        state.replan_count += 1
        replan_context = _replan_context(context, failed_step=failed_step, completed=completed, error=error)
        plan = _normalize_plan(self._planner.plan(replan_context), limits=self._limits, max_steps=remaining_budget)
        validation = self._validator.validate(plan, replan_context)
        if not validation.valid:
            message = "; ".join(validation.errors)
            self.store.add_event(task_id=task.task_id, event_type="failed", message=f"Replan rejected: {message}")
            return []
        start_index = len(existing_steps)
        records = self.store.add_steps(task_id=task.task_id, steps=_records_from_plan(task.task_id, plan, start_index=start_index))
        self.store.add_event(
            task_id=task.task_id,
            step_id=failed_step.step_id,
            event_type="replanned",
            message=plan.summary or f"Replanned after `{failed_step.title}` failed.",
            payload={"failed_step_id": failed_step.step_id, "failed_tool": failed_step.tool_name, "error": str(error)},
        )
        return records

    def _skip_remaining(self, *, task: TaskRecord, steps: Sequence[TaskStepRecord]) -> None:
        for step in steps:
            if step.status != "pending":
                continue
            skipped = self.store.update_step(
                step.step_id,
                status="skipped",
                output="Skipped because the task was replanned.",
                finished=True,
            )
            self.store.add_event(task_id=task.task_id, step_id=skipped.step_id, event_type="skipped", message=skipped.output)

    def _result(self, task: TaskRecord) -> TaskRunResult:
        return TaskRunResult(
            task=task,
            steps=tuple(self.store.list_steps(task_id=task.task_id)),
            events=tuple(self.store.list_events(task_id=task.task_id)),
            final_answer=task.final_answer,
        )


def _records_from_plan(task_id: str, plan: TaskPlan, *, start_index: int = 0) -> list[TaskStepRecord]:
    records: list[TaskStepRecord] = []
    registry = default_tool_registry()
    for index, step in enumerate(plan.steps):
        descriptor = registry.descriptor_for(step.tool_name) if step.tool_name else None
        records.append(
            new_step_record(
                task_id=task_id,
                step_index=start_index + index,
                kind=step.kind,
                title=step.title,
                instruction=step.instruction,
                tool_name=step.tool_name,
                input=step.input,
                depends_on=step.depends_on,
                parallel_group=step.parallel_group,
                risk=descriptor.risk if descriptor is not None else "read_only",
                requires_confirmation=descriptor.requires_confirmation if descriptor is not None else False,
            )
        )
    return records


def _normalize_plan(plan: TaskPlan, *, limits: TaskRuntimeLimits, max_steps: int | None = None) -> TaskPlan:
    step_limit = max(1, min(limits.max_plan_steps, max_steps or limits.max_plan_steps))
    non_answer = tuple(step for step in plan.steps if step.kind != "answer")
    answer = next((step for step in reversed(plan.steps) if step.kind == "answer"), _default_answer_step())
    return TaskPlan(
        goal=plan.goal,
        summary=plan.summary,
        steps=(*non_answer[: step_limit - 1], answer),
        metadata=plan.metadata,
    )


def _default_answer_step() -> PlannedStep:
    return PlannedStep(
        kind="answer",
        title="Generate final answer",
        instruction="Answer the user directly in Chinese, using completed step outputs and noting uncertainty.",
    )


def _replan_context(
    context: TaskContext,
    *,
    failed_step: TaskStepRecord,
    completed: Sequence[TaskStepRecord],
    error: Exception,
) -> TaskContext:
    disabled_tools = {failed_step.tool_name} if failed_step.tool_name else set()
    available_tools = tuple(tool for tool in context.available_tools if str(tool.get("name")) not in disabled_tools)
    metadata = {
        **context.metadata,
        "replan": True,
        "failed_step": {
            "title": failed_step.title,
            "kind": failed_step.kind,
            "tool_name": failed_step.tool_name,
            "error": str(error),
        },
        "completed_step_outputs": _previous_step_text(completed),
    }
    constraints = {
        **context.constraints,
        "avoid_tools": sorted(disabled_tools),
        "replan_reason": str(error),
    }
    return TaskContext(
        goal=context.goal,
        intent=context.intent,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        thread_id=context.thread_id,
        recent_messages=context.recent_messages,
        available_tools=available_tools,
        constraints=constraints,
        metadata=metadata,
    )


def _can_skip_failed_tool(step: TaskStepRecord) -> bool:
    return step.risk in {"read_only", "external_read", "network"}


def _error_key(error: Exception) -> str:
    return f"{type(error).__name__}:{str(error)[:160]}"


def _raise_tool_call_limit(max_tool_calls: int) -> str:
    msg = f"Task reached tool-call limit of {max_tool_calls}."
    raise RuntimeError(msg)


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated after {max_chars} characters]"


def _rag_handler(config: AgentRuntimeConfig) -> ToolHandler:
    retriever: HybridRAGRetriever | None = None

    def handler(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        nonlocal retriever
        if retriever is None:
            from deepagents.runtime.dashscope import create_dashscope_embed_query  # noqa: PLC0415

            embed_query = create_dashscope_embed_query(config)
            retriever = HybridRAGRetriever(
                vector_searcher=MilvusVectorStore(
                    collection_name=config.rag_milvus_collection,
                    uri=config.rag_milvus_uri,
                    token=config.rag_milvus_token,
                    db_name=config.rag_milvus_db,
                    embed_query=embed_query,
                ),
                keyword_searcher=ElasticsearchKeywordStore(index=config.rag_es_index, url=config.rag_es_url),
            )
        query = str(input_data.get("query") or context.goal)
        top_k = _int_value(input_data.get("top_k"), default=6)
        scope = RetrievalScope(tenant_id=context.tenant_id, user_id=context.user_id, kb_ids=config.rag_kb_ids)
        chunks = retriever.retrieve(query, scope=scope, top_k=top_k)
        return format_rag_context(chunks) if chunks else "No relevant knowledge-base chunks found."

    return handler


def _memory_handler(config: AgentRuntimeConfig) -> ToolHandler:
    service = None

    def handler(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        nonlocal service
        from deepagents.memory import MemoryScope  # noqa: PLC0415

        if service is None:
            from deepagents.memory import MemoryService, PostgresMemoryStore  # noqa: PLC0415

            service = MemoryService(PostgresMemoryStore(dsn=config.postgres_dsn or ""))
        query = str(input_data.get("query") or context.goal)
        top_k = _int_value(input_data.get("top_k"), default=5)
        memory_scope = MemoryScope(tenant_id=context.tenant_id, user_id=context.user_id, scope_types=("user",), scope_ids=(context.user_id,))
        return service.build_context(query, scope=memory_scope, limit=top_k)

    return handler


def _recent_messages(messages: Sequence[MessageRecord], *, limit: int = 12) -> tuple[dict[str, str], ...]:
    recent = messages[-limit:]
    return tuple({"role": message.role, "content": message.content[:1200]} for message in recent)


def _tool_dict(descriptor: ToolDescriptor) -> dict[str, object]:
    return {
        "name": descriptor.name,
        "description": descriptor.description,
        "risk": descriptor.risk,
        "source": descriptor.source,
        "requires_confirmation": descriptor.requires_confirmation,
        "tags": list(descriptor.tags),
    }


def _filter_descriptors(descriptors: Sequence[ToolDescriptor], *, disabled_tools: Sequence[str]) -> tuple[ToolDescriptor, ...]:
    disabled = set(disabled_tools)
    return tuple(descriptor for descriptor in descriptors if descriptor.name not in disabled)


def _context_payload(context: TaskContext) -> dict[str, object]:
    return {
        "goal": context.goal,
        "intent": context.intent,
        "recent_messages": list(context.recent_messages),
        "available_tools": list(context.available_tools),
        "constraints": context.constraints,
        "metadata": context.metadata,
    }


def _plan_from_payload(payload: Mapping[str, object], *, fallback_goal: str) -> TaskPlan:
    raw_steps = payload.get("steps")
    steps: list[PlannedStep] = []
    if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, str | bytes):
        for item in raw_steps:
            if not isinstance(item, Mapping):
                continue
            raw = cast("Mapping[str, object]", item)
            steps.append(
                PlannedStep(
                    kind=_step_kind(raw.get("kind")),
                    title=str(raw.get("title") or raw.get("kind") or "Step"),
                    instruction=str(raw.get("instruction") or ""),
                    tool_name=str(raw.get("tool_name") or ""),
                    input=_dict_value(raw.get("input")),
                    depends_on=tuple(str(value) for value in _list_value(raw.get("depends_on"))),
                    parallel_group=str(raw.get("parallel_group") or ""),
                )
            )
    return TaskPlan(
        goal=str(payload.get("goal") or fallback_goal),
        summary=str(payload.get("summary") or ""),
        steps=tuple(steps[:_MAX_STEPS]),
        metadata=_dict_value(payload.get("metadata")),
    )


def _invoke_model(model: _Model, messages: object) -> str:
    result = model.invoke(messages)
    content = getattr(result, "content", result)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _json_payload(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _fallback_model_step(step: TaskStepRecord, *, context: TaskExecutionContext, previous_steps: Sequence[TaskStepRecord]) -> str:
    if step.kind == "answer":
        previous = _previous_step_text(previous_steps)
        if previous:
            return f"Task results organized from current steps:\n\n{previous}"
        return f"Task goal: {context.goal}"
    return _previous_step_text(previous_steps) or step.instruction or step.title


def _previous_step_text(steps: Sequence[TaskStepRecord]) -> str:
    parts = []
    for step in steps:
        output = step.output.strip()
        if output:
            parts.append(f"[{step.title}]\n{output}")
    return "\n\n".join(parts)


def _int_value(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value if value > 0 else default
    if not isinstance(value, str | bytes | bytearray):
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _dict_value(value: object) -> dict[str, object]:
    return dict(cast("Mapping[str, object]", value)) if isinstance(value, Mapping) else {}


def _list_value(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []


def _step_kind(value: object) -> TaskStepKind:
    if value in {"think", "tool", "answer"}:
        return cast("TaskStepKind", value)
    return "think"


__all__ = [
    "ContextBuilder",
    "IntentRouter",
    "LLMPlanner",
    "Observer",
    "PlanValidator",
    "TaskExecutionContext",
    "TaskRunResult",
    "TaskRuntime",
    "TaskRuntimeLimits",
    "TaskStepExecutor",
    "TaskToolExecutor",
    "heuristic_plan",
]
