from typing import cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage

pytest.importorskip("fastapi")

from deepagents.runtime import AgentRuntimeConfig
from deepagents.server.app import (
    APIKeyCreateRequest,
    ChatRequest,
    LoginRequest,
    RegisterRequest,
    TenantCreateRequest,
    UserCreateRequest,
    create_app,
)
from deepagents.server.identity import AuthContext, InMemoryUserCenter


class FakeAgent:
    def __init__(self) -> None:
        self.input_data: dict[str, object] | None = None
        self.config: dict[str, object] | None = None

    def invoke(self, input_data: dict[str, object], config: dict[str, object] | None = None) -> dict[str, list[AIMessage]]:
        self.input_data = input_data
        self.config = config
        return {"messages": [AIMessage(content="pong")]}


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

    app = create_app(
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


def test_api_chat_sends_history_when_checkpointer_is_disabled():
    center = InMemoryUserCenter()
    agent = FakeAgent()
    app = create_app(
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


def test_email_password_register_login_issues_bearer_tokens():
    center = InMemoryUserCenter()
    sample = "correct horse"
    app = create_app(
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


def test_logout_and_revoke_token_disable_bearer_tokens():
    center = InMemoryUserCenter()
    sample = "correct horse"
    app = create_app(
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
    app = create_app(
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
        ("/v1/threads/{thread_id}/messages", "GET"),
        ("/v1/chat", "POST"),
    ]:
        route = _route(app, path, method)
        query_names = {param.name for param in route.dependant.query_params}
        assert "context" not in query_names


def _endpoint(app, path, method):
    return _route(app, path, method).endpoint


def _route(app, path, method):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    msg = f"Missing route {method} {path}"
    raise AssertionError(msg)


def _require_context(context: AuthContext | None) -> AuthContext:
    assert context is not None
    return context
