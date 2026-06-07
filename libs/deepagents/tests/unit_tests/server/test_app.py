from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from deepagents.ingestion import KnowledgeBaseService
from deepagents.ingestion.store import InMemoryKnowledgeBaseStore
from deepagents.runtime import AgentRuntimeConfig
from deepagents.server.app import (
    APIKeyCreateRequest,
    ChatRequest,
    LoginRequest,
    RegisterRequest,
    TaskCreateRequest,
    TaskResumeRequest,
    TenantCreateRequest,
    UserCreateRequest,
    create_app,
)
from deepagents.server.identity import AuthContext, InMemoryUserCenter, MessageRecord
from deepagents.server.pending import InMemoryPendingTurnStore
from deepagents.tasks import InMemoryTaskStore, TaskRuntime


class FakeAgent:
    def __init__(self, *, stream_items: list[object] | None = None) -> None:
        self.input_data: dict[str, object] | None = None
        self.config: dict[str, object] | None = None
        self.stream_items = stream_items or []

    def invoke(self, input_data: dict[str, object], config: dict[str, object] | None = None) -> dict[str, list[AIMessage]]:
        self.input_data = input_data
        self.config = config
        return {"messages": [AIMessage(content="pong")]}

    def stream(
        self,
        input_data: dict[str, object],
        *,
        config: dict[str, object] | None = None,
        stream_mode: object | None = None,
    ) -> Iterator[object]:
        _ = stream_mode
        self.input_data = input_data
        self.config = config
        yield from self.stream_items


class QuotaFailingAgent(FakeAgent):
    def invoke(self, input_data: dict[str, object], config: dict[str, object] | None = None) -> dict[str, list[AIMessage]]:
        self.input_data = input_data
        self.config = config
        msg = "Error code: 429 - insufficient_quota"
        raise RuntimeError(msg)


class FakeToolChunk:
    type = "AIMessageChunk"
    content = "I will search first."

    def __init__(self, tool_name: str) -> None:
        self.tool_calls = [{"name": tool_name}]
        self.tool_call_chunks: list[dict[str, object]] = []


class FakeSummaryService:
    """Summary service double used to verify rolling summary persistence."""

    def __init__(self, summary: str = "persisted project summary") -> None:
        """Initialize the fake service."""
        self.summary = summary
        self.calls: list[tuple[str, list[str]]] = []

    def summarize(self, *, existing_summary: str, messages: Sequence[MessageRecord]) -> str:
        """Return a deterministic summary and record compacted messages."""
        self.calls.append((existing_summary, [message.content for message in messages]))
        return self.summary


SCHEMA_DRIFT_ERROR = type("UndefinedColumn", (Exception,), {})


class SchemaDriftKnowledgeService:
    """Knowledge service double that simulates an old PostgreSQL schema."""

    def ensure_identity(self, **kwargs: object) -> None:
        """No-op identity setup for the test double."""
        _ = kwargs

    def create_knowledge_base(self, **kwargs: object) -> None:
        """Raise the same class name psycopg uses for missing columns."""
        _ = kwargs
        msg = 'column "metadata" of relation "rag_knowledge_bases" does not exist'
        raise SCHEMA_DRIFT_ERROR(msg)


class FakeIndexer:
    """Indexer double used by knowledge-base API tests."""

    def __init__(self) -> None:
        """Initialize deletion call tracking."""
        self.chunks: list[object] = []
        self.deleted_kbs: list[tuple[str, str]] = []
        self.deleted_docs: list[tuple[str, str, str]] = []

    def index(self, chunks: list[object]) -> None:
        """Record indexed chunks."""
        self.chunks.extend(chunks)

    def delete_knowledge_base(self, *, tenant_id: str, kb_id: str) -> None:
        """Record knowledge-base deletion."""
        self.deleted_kbs.append((tenant_id, kb_id))

    def delete_document(self, *, tenant_id: str, kb_id: str, doc_id: str) -> None:
        """Record document deletion."""
        self.deleted_docs.append((tenant_id, kb_id, doc_id))


def _agent_message_contents(agent: FakeAgent) -> list[str]:
    input_data = agent.input_data
    assert isinstance(input_data, dict)
    messages = input_data["messages"]
    assert isinstance(messages, list)
    return [str(message.content) for message in messages if isinstance(message, BaseMessage)]


def test_api_chat_flow_creates_authenticated_thread_messages():
    center = InMemoryUserCenter()
    agent = FakeAgent()
    factory_kwargs: dict[str, object] = {}

    def factory(_config, **kwargs: object):
        factory_kwargs.update(kwargs)
        return agent

    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        agent_factory=factory,
    )

    tenant_response = _endpoint(app, "/v1/admin/tenants", "POST")(TenantCreateRequest(tenant_id="tenant_1", name="Tenant One"))
    assert tenant_response.data["tenant_id"] == "tenant_1"

    user_response = _endpoint(app, "/v1/admin/users", "POST")(
        UserCreateRequest(tenant_id="tenant_1", user_id="user_1", email="user@example.test", display_name="User One")
    )
    assert user_response.data["user_id"] == "user_1"

    key_response = _endpoint(app, "/v1/admin/api-keys", "POST")(APIKeyCreateRequest(tenant_id="tenant_1", user_id="user_1", name="local"))
    api_key = key_response.api_key
    context = center.authenticate_api_key(api_key)
    assert context is not None

    chat_response = _endpoint(app, "/v1/chat", "POST")(ChatRequest(message="hello", title="First thread"), context)
    assert chat_response.content == "pong"
    assert chat_response.thread_id.startswith("thread_")

    messages_response = _endpoint(app, "/v1/threads/{thread_id}/messages", "GET")(chat_response.thread_id, context)
    messages = messages_response.data["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hello"
    assert messages[1]["content"] == "pong"
    assert isinstance(agent.config, dict)
    configurable = cast("dict[str, object]", agent.config["configurable"])
    assert configurable["tenant_id"] == "tenant_1"
    system_prompt = factory_kwargs["system_prompt"]
    assert isinstance(system_prompt, str)
    assert "For comparison questions" in system_prompt
    assert _agent_message_contents(agent) == ["hello"]


def test_api_chat_returns_public_message_for_quota_errors():
    center = InMemoryUserCenter()
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: QuotaFailingAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"message": "hello", "title": "quota"},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "模型额度可能已用尽" in detail
    assert "210825684@qq.com" in detail


def test_api_chat_rejects_oversized_input_without_messages() -> None:
    center = InMemoryUserCenter()
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", max_user_input_tokens=1),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"message": "hello " * 100, "title": "too large"},
    )

    assert response.status_code == 413
    assert center.list_threads(tenant_id=tenant.tenant_id, user_id=user.user_id) == []


def test_api_thread_delete_hides_thread_from_recent_list() -> None:
    center = InMemoryUserCenter()
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))
    thread = center.create_thread(tenant_id=tenant.tenant_id, user_id=user.user_id, title="Remove me")

    response = _endpoint(app, "/v1/threads/{thread_id}", "DELETE")(thread.thread_id, context)

    assert response.data == {"deleted": True, "thread_id": thread.thread_id}
    assert center.list_threads(tenant_id=tenant.tenant_id, user_id=user.user_id) == []
    assert center.get_thread(tenant_id=tenant.tenant_id, user_id=user.user_id, thread_id=thread.thread_id) is None


def test_api_chat_sends_history_when_checkpointer_is_disabled():
    center = InMemoryUserCenter()
    agent = FakeAgent()
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", enable_checkpointer=False),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: agent,
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    thread = center.create_thread(tenant_id=tenant.tenant_id, user_id=user.user_id, thread_id="thread_1")
    center.append_message(tenant_id=tenant.tenant_id, thread_id=thread.thread_id, user_id=user.user_id, role="user", content="old user")
    center.append_message(tenant_id=tenant.tenant_id, thread_id=thread.thread_id, user_id=user.user_id, role="assistant", content="old assistant")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))

    _endpoint(app, "/v1/chat", "POST")(ChatRequest(message="new user", thread_id=thread.thread_id), context)

    assert _agent_message_contents(agent) == ["old user", "old assistant", "new user"]


def test_api_chat_persists_rolling_summary_with_successful_turn() -> None:
    center = InMemoryUserCenter()
    agent = FakeAgent()
    summary = FakeSummaryService()
    app = _create_app(
        config=AgentRuntimeConfig(
            api_admin_key="admin-key",
            context_summary_trigger_tokens=0,
            context_summary_trigger_messages=2,
            context_summary_keep_messages=1,
        ),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: agent,
        thread_summary_service=summary,
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    thread = center.create_thread(tenant_id=tenant.tenant_id, user_id=user.user_id, thread_id="thread_1")
    center.append_message(tenant_id=tenant.tenant_id, thread_id=thread.thread_id, user_id=user.user_id, role="user", content="old user")
    old_assistant = center.append_message(
        tenant_id=tenant.tenant_id,
        thread_id=thread.thread_id,
        user_id=user.user_id,
        role="assistant",
        content="old assistant",
    )
    center.append_message(
        tenant_id=tenant.tenant_id,
        thread_id=thread.thread_id,
        user_id=user.user_id,
        role="user",
        content="old followup",
    )
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))

    _endpoint(app, "/v1/chat", "POST")(ChatRequest(message="new user", thread_id=thread.thread_id), context)

    stored = center.get_thread_summary(tenant_id=tenant.tenant_id, user_id=user.user_id, thread_id=thread.thread_id)
    assert stored is not None
    assert stored.summary == "persisted project summary"
    assert stored.summarized_until_message_seq == old_assistant.message_seq
    assert summary.calls == [("", ["old user", "old assistant"])]
    contents = _agent_message_contents(agent)
    assert contents[0].startswith("Persistent conversation summary")
    assert "persisted project summary" in contents[0]
    assert contents[1:] == ["old followup", "new user"]


def test_api_chat_can_disable_rag_per_request():
    center = InMemoryUserCenter()
    agent = FakeAgent()
    configs: list[AgentRuntimeConfig] = []

    def factory(config: AgentRuntimeConfig, **_kwargs: object) -> FakeAgent:
        configs.append(config)
        return agent

    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", rag_mode="tool"),
        user_center=center,
        agent_factory=factory,
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))

    _endpoint(app, "/v1/chat", "POST")(ChatRequest(message="hello", rag_enabled=False), context)

    assert configs[0].rag_mode == "off"
    assert configs[0].enable_rag is True


def test_api_chat_can_disable_web_search_per_request():
    center = InMemoryUserCenter()
    agent = FakeAgent()
    configs: list[AgentRuntimeConfig] = []

    def factory(config: AgentRuntimeConfig, **_kwargs: object) -> FakeAgent:
        configs.append(config)
        return agent

    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", enable_web_search=True),
        user_center=center,
        agent_factory=factory,
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))

    _endpoint(app, "/v1/chat", "POST")(ChatRequest(message="hello", web_search_enabled=False), context)

    assert configs[0].enable_web_search is False


def test_api_task_flow_creates_task_steps_and_messages() -> None:
    center = InMemoryUserCenter()
    runtime = TaskRuntime(store=InMemoryTaskStore())
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        task_runtime=runtime,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))

    response = _endpoint(app, "/v1/tasks", "POST")(
        TaskCreateRequest(goal="summarize the uploaded document", title="Task test"),
        context,
    )

    task = cast("dict[str, object]", response.data["task"])
    steps = cast("list[dict[str, object]]", response.data["steps"])
    events = cast("list[dict[str, object]]", response.data["events"])
    assert task["status"] == "succeeded"
    assert task["title"] == "Task test"
    assert [step["kind"] for step in steps] == ["process", "answer"]
    assert events[-1]["event_type"] == "finished"
    thread_id = cast("str", task["thread_id"])
    messages = center.list_messages(tenant_id=tenant.tenant_id, thread_id=thread_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].metadata == {"task_id": task["task_id"], "task_mode": True}
    assert messages[1].metadata == {"task_id": task["task_id"], "task_mode": True}


def test_api_task_stream_emits_progress_and_persists_messages() -> None:
    center = InMemoryUserCenter()
    runtime = TaskRuntime(store=InMemoryTaskStore())
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        task_runtime=runtime,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)

    response = TestClient(app).post(
        "/v1/tasks/stream",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"goal": "summarize the uploaded document", "title": "Task stream test"},
    )

    assert response.status_code == 200
    assert "event: task_start" in response.text
    assert "event: task_event" in response.text
    assert "event: task_snapshot" in response.text
    assert "event: done" in response.text
    tasks = runtime.store.list_tasks(tenant_id=tenant.tenant_id, user_id=user.user_id)
    assert len(tasks) == 1
    messages = center.list_messages(tenant_id=tenant.tenant_id, thread_id=tasks[0].thread_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].metadata == {"task_id": tasks[0].task_id, "task_mode": True}


def test_api_task_resume_continues_waiting_task() -> None:
    center = InMemoryUserCenter()
    runtime = TaskRuntime(store=InMemoryTaskStore())
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        task_runtime=runtime,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    context = _require_context(center.authenticate_api_key(key.raw_key))
    thread = center.create_thread(tenant_id=tenant.tenant_id, user_id=user.user_id, title="Trip")
    task = runtime.store.create_task(
        tenant_id=tenant.tenant_id,
        user_id=user.user_id,
        thread_id=thread.thread_id,
        goal="Plan a trip.",
        title="Trip",
        intent="task",
        metadata={"original_goal": "Plan a trip.", "hitl": {"question": "Where do you want to go?", "answers": []}},
    )
    task = runtime.store.update_task(task.task_id, status="waiting_user", final_answer="Where do you want to go?")

    response = _endpoint(app, "/v1/tasks/{task_id}/resume", "POST")(
        task.task_id,
        TaskResumeRequest(message="I want to go to Beijing for 7 days."),
        context,
    )

    resumed = cast("dict[str, object]", response.data["task"])
    assert resumed["task_id"] == task.task_id
    assert resumed["status"] == "succeeded"
    assert "Beijing" in cast("str", resumed["goal"])
    assert [stored.task_id for stored in runtime.store.list_tasks(tenant_id=tenant.tenant_id, user_id=user.user_id)] == [task.task_id]
    metadata = cast("dict[str, object]", resumed["metadata"])
    hitl = cast("dict[str, object]", metadata["hitl"])
    answers = cast("list[dict[str, object]]", hitl["answers"])
    assert metadata["original_goal"] == "Plan a trip."
    assert hitl["status"] == "resumed"
    assert answers[-1]["question"] == "Where do you want to go?"
    assert answers[-1]["content"] == "I want to go to Beijing for 7 days."
    events = runtime.store.list_events(task_id=task.task_id)
    assert "hitl_resumed" in [event.event_type for event in events]
    messages = center.list_messages(tenant_id=tenant.tenant_id, thread_id=thread.thread_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "I want to go to Beijing for 7 days."
    assert messages[0].metadata == {"task_id": task.task_id, "task_mode": True, "task_resume": True}
    assert messages[1].metadata == {"task_id": task.task_id, "task_mode": True, "task_resume": True}


def test_api_task_resume_stream_continues_waiting_task() -> None:
    center = InMemoryUserCenter()
    runtime = TaskRuntime(store=InMemoryTaskStore())
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        task_runtime=runtime,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    thread = center.create_thread(tenant_id=tenant.tenant_id, user_id=user.user_id, title="Trip")
    task = runtime.store.create_task(
        tenant_id=tenant.tenant_id,
        user_id=user.user_id,
        thread_id=thread.thread_id,
        goal="Plan a trip.",
        title="Trip",
        intent="task",
        metadata={"original_goal": "Plan a trip.", "hitl": {"question": "Where do you want to go?", "answers": []}},
    )
    task = runtime.store.update_task(task.task_id, status="waiting_user", final_answer="Where do you want to go?")

    response = TestClient(app).post(
        f"/v1/tasks/{task.task_id}/resume/stream",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"message": "I want to go to Beijing for 7 days."},
    )

    assert response.status_code == 200
    assert "event: task_start" in response.text
    assert "event: done" in response.text
    assert task.task_id in response.text
    assert [stored.task_id for stored in runtime.store.list_tasks(tenant_id=tenant.tenant_id, user_id=user.user_id)] == [task.task_id]
    messages = center.list_messages(tenant_id=tenant.tenant_id, thread_id=thread.thread_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].metadata == {"task_id": task.task_id, "task_mode": True, "task_resume": True}
    assert messages[1].metadata == {"task_id": task.task_id, "task_mode": True, "task_resume": True}


def test_api_chat_stream_emits_status_delta_and_persists_message():
    center = InMemoryUserCenter()
    agent = FakeAgent(
        stream_items=[
            ("messages", (FakeToolChunk("search_knowledge_base"), {})),
            ("updates", {"tools": {"messages": [ToolMessage(content="found it", name="search_knowledge_base", tool_call_id="call-1")]}}),
            ("messages", (AIMessageChunk(content="po"), {})),
            ("messages", (AIMessageChunk(content="ng"), {})),
            ("updates", {"agent": {"messages": [AIMessage(content="pong")]}}),
        ]
    )
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: agent,
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)

    response = TestClient(app).post(
        "/v1/chat/stream",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"message": "hello", "title": "Streaming test", "rag_enabled": True},
    )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "正在检索知识库" in response.text
    assert "I will search first" not in response.text
    assert 'data: {"text":"po"}' in response.text
    assert 'data: {"text":"ng"}' in response.text
    assert '"replace":false' in response.text
    assert "event: done" in response.text
    messages = center.list_messages(tenant_id=tenant.tenant_id, thread_id=response.text.split('"thread_id":"')[1].split('"')[0])
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].content == "pong"


def test_email_password_register_login_issues_bearer_tokens():
    center = InMemoryUserCenter()
    sample = "correct horse"
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", tenant_id="default"),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )

    register_response = _endpoint(app, "/v1/auth/register", "POST")(RegisterRequest(email="User@Example.Test", password=sample, display_name="User"))
    assert register_response.token_type == "bearer"  # noqa: S105  # OAuth token type label, not a credential.
    assert register_response.access_token.startswith("kya_")
    assert register_response.expires_at is not None
    assert register_response.data["user"]["email"] == "user@example.test"

    register_context = center.authenticate_api_key(register_response.access_token)
    assert register_context is not None
    assert register_context.tenant.tenant_id == "default"

    login_response = _endpoint(app, "/v1/auth/login", "POST")(LoginRequest(email="user@example.test", password=sample))
    assert login_response.access_token.startswith("kya_")
    assert login_response.access_token != register_response.access_token

    login_context = center.authenticate_api_key(login_response.access_token)
    assert login_context is not None
    assert login_context.user.user_id == register_context.user.user_id


def test_knowledge_base_upload_queues_document(tmp_path):
    center = InMemoryUserCenter()
    indexer = FakeIndexer()
    knowledge = KnowledgeBaseService(
        config=AgentRuntimeConfig(api_admin_key="admin-key", upload_dir=str(tmp_path)),
        store=InMemoryKnowledgeBaseStore(),
        indexer=cast("Any", indexer),
    )
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", upload_dir=str(tmp_path)),
        user_center=center,
        knowledge_service=knowledge,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    client = TestClient(app)

    kb_response = client.post(
        "/v1/knowledge-bases",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"name": "PDF KB"},
    )
    assert kb_response.status_code == 200
    kb_id = kb_response.json()["data"]["kb_id"]

    upload_response = client.post(
        f"/v1/knowledge-bases/{kb_id}/documents?filename=sample.pdf",
        headers={"Authorization": f"Bearer {key.raw_key}", "Content-Type": "application/pdf"},
        content=b"%PDF fake content",
    )

    assert upload_response.status_code == 200
    data = upload_response.json()["data"]
    assert data["document"]["status"] == "processing"
    assert data["job"]["status"] == "queued"

    documents_response = client.get(
        f"/v1/knowledge-bases/{kb_id}/documents",
        headers={"Authorization": f"Bearer {key.raw_key}"},
    )
    assert documents_response.status_code == 200
    doc_id = documents_response.json()["data"]["documents"][0]["doc_id"]
    assert documents_response.json()["data"]["documents"][0]["file_name"] == "sample.pdf"

    delete_document_response = client.delete(
        f"/v1/knowledge-bases/{kb_id}/documents/{doc_id}",
        headers={"Authorization": f"Bearer {key.raw_key}"},
    )
    assert delete_document_response.status_code == 200
    assert delete_document_response.json()["data"]["status"] == "deleted"
    assert indexer.deleted_docs == [("tenant_1", kb_id, doc_id)]

    delete_kb_response = client.delete(
        f"/v1/knowledge-bases/{kb_id}",
        headers={"Authorization": f"Bearer {key.raw_key}"},
    )
    assert delete_kb_response.status_code == 200
    assert delete_kb_response.json()["data"]["status"] == "archived"
    assert indexer.deleted_kbs == [("tenant_1", kb_id)]


def test_knowledge_base_schema_drift_returns_actionable_error():
    center = InMemoryUserCenter()
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=center,
        knowledge_service=cast("Any", SchemaDriftKnowledgeService()),
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )
    tenant = center.create_tenant(name="Tenant One", tenant_id="tenant_1")
    user = center.create_user(tenant_id=tenant.tenant_id, user_id="user_1", email="user@example.test")
    key = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id)
    client = TestClient(app)

    response = client.post(
        "/v1/knowledge-bases",
        headers={"Authorization": f"Bearer {key.raw_key}"},
        json={"name": "PDF KB"},
    )

    assert response.status_code == 503
    assert "PostgreSQL schema is out of date" in response.json()["detail"]
    assert "bootstrap_runtime.py" in response.json()["detail"]


def test_logout_and_revoke_token_disable_bearer_tokens():
    center = InMemoryUserCenter()
    sample = "correct horse"
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key", tenant_id="default"),
        user_center=center,
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )

    register_response = _endpoint(app, "/v1/auth/register", "POST")(RegisterRequest(email="user@example.test", password=sample))
    first = _require_context(center.authenticate_api_key(register_response.access_token))

    logout_response = _endpoint(app, "/v1/auth/logout", "POST")(first)
    assert logout_response.data == {"revoked": True, "key_id": first.api_key.key_id}
    assert center.authenticate_api_key(register_response.access_token) is None

    login_response = _endpoint(app, "/v1/auth/login", "POST")(LoginRequest(email="user@example.test", password=sample))
    second = _require_context(center.authenticate_api_key(login_response.access_token))
    revoke_response = _endpoint(app, "/v1/auth/tokens/{key_id}/revoke", "POST")(second.api_key.key_id, second)

    assert revoke_response.data == {"revoked": True, "key_id": second.api_key.key_id}
    assert center.authenticate_api_key(login_response.access_token) is None


def test_auth_context_is_not_exposed_as_query_parameter():
    app = _create_app(
        config=AgentRuntimeConfig(api_admin_key="admin-key"),
        user_center=InMemoryUserCenter(),
        agent_factory=lambda _config, **_kwargs: FakeAgent(),
    )

    for path, method in [
        ("/v1/me", "GET"),
        ("/v1/auth/logout", "POST"),
        ("/v1/auth/tokens/{key_id}/revoke", "POST"),
        ("/v1/threads", "GET"),
        ("/v1/threads", "POST"),
        ("/v1/threads/{thread_id}", "DELETE"),
        ("/v1/threads/{thread_id}/messages", "GET"),
        ("/v1/knowledge-bases", "GET"),
        ("/v1/knowledge-bases", "POST"),
        ("/v1/knowledge-bases/{kb_id}", "DELETE"),
        ("/v1/knowledge-bases/{kb_id}/documents", "GET"),
        ("/v1/knowledge-bases/{kb_id}/documents", "POST"),
        ("/v1/knowledge-bases/{kb_id}/documents/{doc_id}", "DELETE"),
        ("/v1/ingestion/jobs/{job_id}", "GET"),
        ("/v1/tasks", "GET"),
        ("/v1/tasks", "POST"),
        ("/v1/tasks/stream", "POST"),
        ("/v1/tasks/{task_id}/resume", "POST"),
        ("/v1/tasks/{task_id}/resume/stream", "POST"),
        ("/v1/tasks/{task_id}", "GET"),
        ("/v1/tasks/{task_id}/events", "GET"),
        ("/v1/tasks/{task_id}/cancel", "POST"),
        ("/v1/chat", "POST"),
    ]:
        route = _route(app, path, method)
        query_names = {param.name for param in route.dependant.query_params}
        assert "context" not in query_names


def _endpoint(app, path, method):
    return _route(app, path, method).endpoint


def _create_app(**kwargs: Any):
    return create_app(pending_turn_store=InMemoryPendingTurnStore(), **kwargs)


def _route(app, path, method):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    msg = f"Missing route {method} {path}"
    raise AssertionError(msg)


def _require_context(context: AuthContext | None) -> AuthContext:
    assert context is not None
    return context
