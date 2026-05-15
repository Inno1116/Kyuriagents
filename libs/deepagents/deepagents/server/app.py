"""FastAPI application factory for deployed Deep Agents runtimes."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from deepagents.runtime import AgentRuntimeConfig, create_kyuri_agent
from deepagents.server.identity import AuthContext, DuplicateUserError, MessageRecord, PostgresUserCenter, ThreadRecord, UserCenter

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class _Agent(Protocol):
    """Minimal runnable agent contract used by the API layer."""

    def invoke(self, input_data: object, /, config: Mapping[str, object] | None = None) -> object:
        """Invoke the agent with LangGraph-style input."""
        ...


class ThreadNotFoundError(LookupError):
    """Raised when a user cannot access the requested thread."""


_MIN_PASSWORD_LENGTH = 8
_DEFAULT_API_SYSTEM_PROMPT = """You are Kyuriagents, a careful agent with access to tools, RAG, and long-term memory.

Answer the user's question directly and in the user's language unless they ask otherwise.
When retrieved knowledge-base or memory context is available, ground your answer in that context and prefer concrete dates, names, and facts.
For comparison questions, state the conclusion first, then give the supporting facts.
Do not narrate internal reasoning, do not say what the user "appears to be asking", and do not mention search mechanics unless the user asks.
If the available context is insufficient, say so briefly and name the missing information.
"""


class RegisterRequest(BaseModel):
    """Request body for email/password registration."""

    email: str
    password: str = Field(min_length=_MIN_PASSWORD_LENGTH)
    display_name: str = ""
    tenant_id: str | None = None
    tenant_name: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class LoginRequest(BaseModel):
    """Request body for email/password login."""

    email: str
    password: str
    tenant_id: str | None = None


class TenantCreateRequest(BaseModel):
    """Request body for creating a tenant."""

    name: str
    tenant_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class UserCreateRequest(BaseModel):
    """Request body for creating a user."""

    tenant_id: str
    email: str
    display_name: str = ""
    user_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class APIKeyCreateRequest(BaseModel):
    """Request body for creating an API key."""

    tenant_id: str
    user_id: str
    name: str = ""
    expires_at: str | None = None


class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""

    title: str = ""
    thread_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request body for sending a user message."""

    message: str
    thread_id: str | None = None
    title: str = ""


class RecordResponse(BaseModel):
    """Generic JSON response for record-shaped objects."""

    data: dict[str, object]


class APIKeyResponse(BaseModel):
    """Response body for API key creation."""

    data: dict[str, object]
    api_key: str


class AuthTokenResponse(BaseModel):
    """Response body for email/password auth endpoints."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105  # OAuth token type label, not a credential.
    expires_at: str | None = None
    data: dict[str, object]


class ChatResponse(BaseModel):
    """Response body for chat calls."""

    thread_id: str
    message_id: str
    content: str


def create_app(
    *,
    config: AgentRuntimeConfig | None = None,
    user_center: UserCenter | None = None,
    agent_factory: Callable[..., object] | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        config: Runtime configuration. Defaults to `AgentRuntimeConfig.from_env()`.
        user_center: Optional user center store. Defaults to PostgreSQL.
        agent_factory: Optional agent factory for tests.

    Returns:
        FastAPI app instance.

    Raises:
        ValueError: If PostgreSQL DSN is missing when no user center is supplied.
    """
    resolved_config = config or AgentRuntimeConfig.from_env()
    resolved_center = user_center or _postgres_user_center(resolved_config)
    resolved_agent_factory = agent_factory or create_kyuri_agent
    app = FastAPI(title="Deep Agents API", version="0.1.0")
    _configure_cors(app, resolved_config)
    _register_auth_routes(app=app, config=resolved_config, user_center=resolved_center)
    _register_admin_routes(app=app, config=resolved_config, user_center=resolved_center)
    _register_user_routes(app=app, config=resolved_config, user_center=resolved_center, agent_factory=resolved_agent_factory)
    return app


def _configure_cors(app: FastAPI, config: AgentRuntimeConfig) -> None:
    """Allow configured browser origins to call the API."""
    if not config.api_cors_origins:
        return
    app.add_middleware(
        cast("Any", CORSMiddleware),
        allow_origins=list(config.api_cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_auth_routes(*, app: FastAPI, config: AgentRuntimeConfig, user_center: UserCenter) -> None:
    """Register public email/password authentication routes."""

    @app.post("/v1/auth/register", response_model=AuthTokenResponse)
    def register(request: RegisterRequest) -> AuthTokenResponse:
        tenant_id = request.tenant_id or config.tenant_id
        user_center.ensure_tenant(name=request.tenant_name or tenant_id, tenant_id=tenant_id)
        try:
            user = user_center.create_user_with_password(
                tenant_id=tenant_id,
                email=request.email,
                password=request.password,
                display_name=request.display_name,
                metadata=request.metadata,
            )
        except DuplicateUserError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A user with this email already exists.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        created = user_center.create_api_key(
            tenant_id=tenant_id,
            user_id=user.user_id,
            name="password-login",
            expires_at=_token_expires_at(config),
        )
        return _auth_token_response(user_center, created.raw_key)

    @app.post("/v1/auth/login", response_model=AuthTokenResponse)
    def login(request: LoginRequest) -> AuthTokenResponse:
        tenant_id = request.tenant_id or config.tenant_id
        try:
            user = user_center.authenticate_password(tenant_id=tenant_id, email=request.email, password=request.password)
        except ValueError:
            user = None
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
        created = user_center.create_api_key(
            tenant_id=tenant_id,
            user_id=user.user_id,
            name="password-login",
            expires_at=_token_expires_at(config),
        )
        return _auth_token_response(user_center, created.raw_key)


def _register_admin_routes(*, app: FastAPI, config: AgentRuntimeConfig, user_center: UserCenter) -> None:
    """Register tenant and API key administration routes."""

    def require_admin(x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None) -> None:
        if not config.api_admin_key:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin API is not configured.")
        if x_admin_key != config.api_admin_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key.")

    @app.post("/v1/admin/tenants", response_model=RecordResponse, dependencies=[Depends(require_admin)])
    def create_tenant(request: TenantCreateRequest) -> RecordResponse:
        tenant = user_center.create_tenant(name=request.name, tenant_id=request.tenant_id, metadata=request.metadata)
        return RecordResponse(data=_record_dict(tenant))

    @app.post("/v1/admin/users", response_model=RecordResponse, dependencies=[Depends(require_admin)])
    def create_user(request: UserCreateRequest) -> RecordResponse:
        user = user_center.create_user(
            tenant_id=request.tenant_id,
            email=request.email,
            display_name=request.display_name,
            user_id=request.user_id,
            metadata=request.metadata,
        )
        return RecordResponse(data=_record_dict(user))

    @app.post("/v1/admin/api-keys", response_model=APIKeyResponse, dependencies=[Depends(require_admin)])
    def create_api_key(request: APIKeyCreateRequest) -> APIKeyResponse:
        created = user_center.create_api_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            name=request.name,
            expires_at=request.expires_at,
        )
        return APIKeyResponse(data=_record_dict(created.record), api_key=created.raw_key)


def _register_user_routes(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    agent_factory: Callable[..., object],
) -> None:
    """Register authenticated user and chat routes."""
    require_auth = _make_auth_dependency(user_center)
    _register_user_metadata_routes(app=app, user_center=user_center, require_auth=require_auth)
    _register_chat_route(app=app, config=config, user_center=user_center, agent_factory=agent_factory, require_auth=require_auth)


def _make_auth_dependency(user_center: UserCenter) -> Callable[..., AuthContext]:
    """Build the FastAPI dependency that authenticates API keys."""

    def require_auth(
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> AuthContext:
        raw_key = _raw_api_key(authorization=authorization, x_api_key=x_api_key)
        if raw_key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key.")
        context = user_center.authenticate_api_key(raw_key)
        if context is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
        return context

    return require_auth


def _register_user_metadata_routes(
    *,
    app: FastAPI,
    user_center: UserCenter,
    require_auth: Callable[..., AuthContext],
) -> None:
    """Register authenticated user, thread, and message metadata routes."""
    auth_dependency = cast("AuthContext", Depends(require_auth))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/me", response_model=RecordResponse)
    def me(context: AuthContext = auth_dependency) -> RecordResponse:
        return RecordResponse(
            data={
                "tenant": _record_dict(context.tenant),
                "user": _record_dict(context.user),
                "api_key": {"key_id": context.api_key.key_id, "name": context.api_key.name, "key_prefix": context.api_key.key_prefix},
            }
        )

    @app.post("/v1/auth/logout", response_model=RecordResponse)
    def logout(context: AuthContext = auth_dependency) -> RecordResponse:
        revoked = user_center.revoke_api_key(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            key_id=context.api_key.key_id,
        )
        return RecordResponse(data={"revoked": revoked, "key_id": context.api_key.key_id})

    @app.post("/v1/auth/tokens/{key_id}/revoke", response_model=RecordResponse)
    def revoke_token(key_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        revoked = user_center.revoke_api_key(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            key_id=key_id,
        )
        if not revoked:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found.")
        return RecordResponse(data={"revoked": True, "key_id": key_id})

    @app.post("/v1/threads", response_model=RecordResponse)
    def create_thread(request: ThreadCreateRequest, context: AuthContext = auth_dependency) -> RecordResponse:
        thread = user_center.create_thread(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            title=request.title,
            thread_id=request.thread_id,
            metadata=request.metadata,
        )
        return RecordResponse(data=_record_dict(thread))

    @app.get("/v1/threads", response_model=RecordResponse)
    def list_threads(context: AuthContext = auth_dependency, limit: int = 50) -> RecordResponse:
        threads = user_center.list_threads(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, limit=limit)
        return RecordResponse(data={"threads": [_record_dict(thread) for thread in threads]})

    @app.get("/v1/threads/{thread_id}/messages", response_model=RecordResponse)
    def list_messages(thread_id: str, context: AuthContext = auth_dependency, limit: int = 100) -> RecordResponse:
        try:
            thread = _require_thread(user_center, context=context, thread_id=thread_id)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        messages = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=limit)
        return RecordResponse(data={"messages": [_record_dict(message) for message in messages]})


def _register_chat_route(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    agent_factory: Callable[..., object],
    require_auth: Callable[..., AuthContext],
) -> None:
    """Register the chat endpoint."""
    auth_dependency = cast("AuthContext", Depends(require_auth))

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, context: AuthContext = auth_dependency) -> ChatResponse:
        try:
            thread = _resolve_thread(user_center, context=context, request=request)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        history = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=100)
        user_message = user_center.append_message(
            tenant_id=context.tenant.tenant_id,
            thread_id=thread.thread_id,
            user_id=context.user.user_id,
            role="user",
            content=request.message,
        )
        agent_config = replace(
            config,
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            thread_id=thread.thread_id,
        )
        agent = cast("_Agent", agent_factory(agent_config, system_prompt=_DEFAULT_API_SYSTEM_PROMPT))
        result = agent.invoke(
            {
                "messages": _agent_messages(
                    history,
                    user_message=user_message,
                    use_checkpointer=agent_config.enable_checkpointer,
                )
            },
            config={
                "configurable": {
                    "tenant_id": context.tenant.tenant_id,
                    "user_id": context.user.user_id,
                    "thread_id": thread.thread_id,
                    "tool_thread_id": thread.thread_id,
                    "memory_scope_types": ["user"],
                    "memory_scope_ids": [context.user.user_id],
                }
            },
        )
        content = _assistant_text(cast("Mapping[str, object]", result))
        assistant_message = user_center.append_message(
            tenant_id=context.tenant.tenant_id,
            thread_id=thread.thread_id,
            user_id=context.user.user_id,
            role="assistant",
            content=content,
        )
        return ChatResponse(thread_id=thread.thread_id, message_id=assistant_message.message_id, content=content)


def _postgres_user_center(config: AgentRuntimeConfig) -> PostgresUserCenter:
    if not config.postgres_dsn:
        msg = "Set `DEEPAGENTS_POSTGRES_DSN` before starting the API server."
        raise ValueError(msg)
    return PostgresUserCenter(dsn=config.postgres_dsn)


def _raw_api_key(*, authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def _auth_token_response(user_center: UserCenter, raw_key: str) -> AuthTokenResponse:
    context = user_center.authenticate_api_key(raw_key)
    if context is None:
        msg = "Created API key could not be authenticated."
        raise RuntimeError(msg)
    return AuthTokenResponse(
        access_token=raw_key,
        expires_at=context.api_key.expires_at,
        data={
            "tenant": _record_dict(context.tenant),
            "user": _record_dict(context.user),
            "api_key": {"key_id": context.api_key.key_id, "name": context.api_key.name, "key_prefix": context.api_key.key_prefix},
        },
    )


def _token_expires_at(config: AgentRuntimeConfig) -> str | None:
    if config.auth_token_ttl_days <= 0:
        return None
    return (datetime.now(tz=UTC) + timedelta(days=config.auth_token_ttl_days)).isoformat()


def _resolve_thread(
    user_center: UserCenter,
    *,
    context: AuthContext,
    request: ChatRequest,
) -> ThreadRecord:
    if request.thread_id is None:
        return user_center.create_thread(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            title=request.title,
        )
    return _require_thread(user_center, context=context, thread_id=request.thread_id)


def _require_thread(
    user_center: UserCenter,
    *,
    context: AuthContext,
    thread_id: str,
) -> ThreadRecord:
    thread = user_center.get_thread(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, thread_id=thread_id)
    if thread is None:
        raise ThreadNotFoundError
    return thread


def _to_langchain_messages(messages: Sequence[MessageRecord]) -> list[object]:
    converted: list[object] = []
    for message in messages:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content, id=message.message_id))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content, id=message.message_id))
        elif message.role == "system":
            converted.append(SystemMessage(content=message.content, id=message.message_id))
    return converted


def _agent_messages(
    history: Sequence[MessageRecord],
    *,
    user_message: MessageRecord,
    use_checkpointer: bool,
) -> list[object]:
    current = HumanMessage(content=user_message.content, id=user_message.message_id)
    if use_checkpointer:
        return [current]
    return [*_to_langchain_messages(history), current]


def _assistant_text(result: Mapping[str, object]) -> str:
    messages = result.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = getattr(message, "text", "")
            if isinstance(text, str):
                return text
            content = getattr(message, "content", "")
            return str(content)
    return ""


def _record_dict(record: object) -> dict[str, object]:
    data = getattr(record, "__dict__", {})
    if not isinstance(data, dict):
        return {}
    hidden = {"key_hash", "password_hash"}
    return {key: value for key, value in data.items() if key not in hidden}


__all__ = ["create_app"]
