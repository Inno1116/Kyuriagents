"""Middleware that connects RAG and long-term memory to the main agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast, get_args

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime  # noqa: TC002  # StructuredTool detects runtime injection from this annotation.
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

from deepagents.memory import (
    MemoryScope,
    MemoryService,
    MemoryWriteCandidate,
    format_memory_context,
)
from deepagents.memory.types import MemoryScopeType, MemoryType, MemoryVisibility
from deepagents.middleware._utils import append_to_system_message
from deepagents.rag import HybridRAGRetriever, RetrievalScope, RetrievedChunk

RetrievalMode = Literal["off", "auto", "tool", "hybrid"]
"""Runtime modes for automatic context injection and explicit tools."""

RagScopeResolver = Callable[[object], RetrievalScope]
"""Callable that derives a RAG scope from LangGraph runtime objects."""

MemoryScopeResolver = Callable[[object], MemoryScope]
"""Callable that derives a memory scope from LangGraph runtime objects."""

_MEMORY_TYPES = frozenset(get_args(MemoryType))
_MEMORY_SCOPE_TYPES = frozenset(get_args(MemoryScopeType))
_MEMORY_VISIBILITIES = frozenset(get_args(MemoryVisibility))


@dataclass(frozen=True, kw_only=True)
class RuntimeContextDefaults:
    """Default tenant and user values for retrieval scopes.

    Args:
        tenant_id: Fallback tenant identifier when runtime config does not
            provide one.
        user_id: Optional fallback user identifier.
        kb_ids: Optional fallback knowledge-base identifiers for RAG.
    """

    tenant_id: str = "default"
    user_id: str | None = None
    kb_ids: tuple[str, ...] = ()


def resolve_retrieval_scope(
    runtime: object,
    *,
    defaults: RuntimeContextDefaults | None = None,
) -> RetrievalScope:
    """Resolve a RAG retrieval scope from runtime context and config.

    Looks for `tenant_id`, `user_id`, and `kb_ids` in runtime context,
    metadata, and configurable values. `rag_tenant_id`, `rag_user_id`, and
    `rag_kb_ids` override the generic names when provided.

    Args:
        runtime: LangGraph runtime or tool runtime object.
        defaults: Fallback scope values.

    Returns:
        `RetrievalScope` for the current request.
    """
    resolved_defaults = defaults or RuntimeContextDefaults()
    values = _runtime_values(runtime)
    tenant_id = _string_value(values.get("rag_tenant_id") or values.get("tenant_id")) or resolved_defaults.tenant_id
    user_id = _optional_string(values.get("rag_user_id") or values.get("user_id")) or resolved_defaults.user_id
    kb_ids = _string_tuple(values.get("rag_kb_ids") or values.get("kb_ids")) or resolved_defaults.kb_ids
    doc_ids = _string_tuple(values.get("rag_doc_ids") or values.get("doc_ids"))
    tags = _string_tuple(values.get("rag_tags") or values.get("tags"))
    languages = _string_tuple(values.get("rag_languages") or values.get("languages"))
    source_types = _string_tuple(values.get("rag_source_types") or values.get("source_types"))

    return RetrievalScope(
        tenant_id=tenant_id,
        user_id=user_id,
        kb_ids=kb_ids,
        doc_ids=doc_ids,
        tags=tags,
        languages=languages,
        source_types=source_types,
    )


def resolve_memory_scope(
    runtime: object,
    *,
    defaults: RuntimeContextDefaults | None = None,
) -> MemoryScope:
    """Resolve a long-term memory scope from runtime context and config.

    Looks for `tenant_id`, `user_id`, `memory_scope_types`, `memory_scope_ids`,
    `memory_types`, and `memory_tags`.

    Args:
        runtime: LangGraph runtime or tool runtime object.
        defaults: Fallback scope values.

    Returns:
        `MemoryScope` for the current request.
    """
    resolved_defaults = defaults or RuntimeContextDefaults()
    values = _runtime_values(runtime)
    tenant_id = _string_value(values.get("memory_tenant_id") or values.get("tenant_id")) or resolved_defaults.tenant_id
    user_id = _optional_string(values.get("memory_user_id") or values.get("user_id")) or resolved_defaults.user_id
    scope_types = _memory_scope_types(values.get("memory_scope_types"))
    scope_ids = _string_tuple(values.get("memory_scope_ids"))
    memory_types = _memory_types(values.get("memory_types"))
    tags = _string_tuple(values.get("memory_tags") or values.get("tags"))

    return MemoryScope(
        tenant_id=tenant_id,
        user_id=user_id,
        scope_types=scope_types,
        scope_ids=scope_ids,
        memory_types=memory_types,
        tags=tags,
    )


def format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved knowledge-base chunks for prompt injection.

    Args:
        chunks: Ranked retrieved chunks.

    Returns:
        XML-like context block with source metadata and snippets.
    """
    if not chunks:
        return "<retrieved_knowledge_base>\n(No relevant knowledge-base chunks found.)\n</retrieved_knowledge_base>"

    lines = ["<retrieved_knowledge_base>"]
    for chunk in chunks:
        metadata = chunk.metadata
        score = _chunk_score(chunk)
        title = metadata.title or metadata.doc_id
        source = metadata.source_uri or metadata.doc_id
        lines.append(f"- [{chunk.chunk_id}; score={score:.4f}; title={title}; source={source}] {chunk.text}")
    lines.append("</retrieved_knowledge_base>")
    return "\n".join(lines)


class RetrievalMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Connect hybrid RAG and long-term memory to a Deep Agent.

    The middleware can automatically inject small Top-K context blocks before
    model calls, expose explicit `search_knowledge_base` and `search_memory`
    tools, and expose simple memory write/delete tools backed by `MemoryService`.

    Args:
        rag_retriever: Optional hybrid RAG retriever.
        memory_service: Optional long-term memory service.
        rag_mode: Whether RAG runs automatically, as a tool, both, or neither.
        memory_mode: Whether memory runs automatically, as tools, both, or neither.
        defaults: Fallback tenant/user/kb values.
        rag_scope: Optional static RAG scope.
        memory_scope: Optional static memory scope.
        rag_scope_resolver: Optional callable for dynamic RAG scope resolution.
        memory_scope_resolver: Optional callable for dynamic memory scope resolution.
        rag_auto_top_k: Maximum chunks to inject automatically.
        rag_tool_top_k: Default chunks returned by the RAG tool.
        memory_auto_top_k: Maximum memories to inject automatically.
        memory_tool_top_k: Default memories returned by the memory search tool.
        min_rag_score: Minimum RAG score for automatic injection.
        min_memory_score: Minimum memory score for automatic injection.
    """

    def __init__(
        self,
        *,
        rag_retriever: HybridRAGRetriever | None = None,
        memory_service: MemoryService | None = None,
        rag_mode: RetrievalMode = "tool",
        memory_mode: RetrievalMode = "hybrid",
        defaults: RuntimeContextDefaults | None = None,
        rag_scope: RetrievalScope | None = None,
        memory_scope: MemoryScope | None = None,
        rag_scope_resolver: RagScopeResolver | None = None,
        memory_scope_resolver: MemoryScopeResolver | None = None,
        rag_auto_top_k: int = 4,
        rag_tool_top_k: int = 8,
        memory_auto_top_k: int = 3,
        memory_tool_top_k: int = 10,
        min_rag_score: float = 0.0,
        min_memory_score: float = 0.0,
    ) -> None:
        """Initialize the retrieval middleware."""
        self._rag_retriever = rag_retriever
        self._memory_service = memory_service
        self._rag_mode = rag_mode
        self._memory_mode = memory_mode
        self._defaults = defaults or RuntimeContextDefaults()
        self._rag_scope = rag_scope
        self._memory_scope = memory_scope
        self._rag_scope_resolver = rag_scope_resolver
        self._memory_scope_resolver = memory_scope_resolver
        self._rag_auto_top_k = _positive("rag_auto_top_k", rag_auto_top_k)
        self._rag_tool_top_k = _positive("rag_tool_top_k", rag_tool_top_k)
        self._memory_auto_top_k = _positive("memory_auto_top_k", memory_auto_top_k)
        self._memory_tool_top_k = _positive("memory_tool_top_k", memory_tool_top_k)
        self._min_rag_score = min_rag_score
        self._min_memory_score = min_memory_score
        self.tools = self._build_tools()

    def _build_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        if self._rag_retriever is not None and _has_tool(self._rag_mode):
            tools.append(self._build_search_knowledge_base_tool())
        if self._memory_service is not None and _has_tool(self._memory_mode):
            tools.extend(
                [
                    self._build_search_memory_tool(),
                    self._build_save_memory_tool(),
                    self._build_delete_memory_tool(),
                ]
            )
        return tools

    def _resolve_rag_scope(self, runtime: object) -> RetrievalScope:
        if self._rag_scope is not None:
            return self._rag_scope
        if self._rag_scope_resolver is not None:
            return self._rag_scope_resolver(runtime)
        return resolve_retrieval_scope(runtime, defaults=self._defaults)

    def _resolve_memory_scope(self, runtime: object) -> MemoryScope:
        if self._memory_scope is not None:
            return self._memory_scope
        if self._memory_scope_resolver is not None:
            return self._memory_scope_resolver(runtime)
        return resolve_memory_scope(runtime, defaults=self._defaults)

    def _retrieve_rag_context(self, query: str, runtime: object, *, limit: int) -> str:
        if self._rag_retriever is None:
            return ""
        chunks = self._rag_retriever.retrieve(query, scope=self._resolve_rag_scope(runtime), top_k=limit)
        filtered = [chunk for chunk in chunks if _chunk_score(chunk) >= self._min_rag_score]
        if not filtered:
            return ""
        return format_rag_context(filtered)

    def _retrieve_memory_context(self, query: str, runtime: object, *, limit: int) -> str:
        if self._memory_service is None:
            return ""
        results = self._memory_service.search(query, scope=self._resolve_memory_scope(runtime), limit=limit)
        filtered = [result for result in results if result.score >= self._min_memory_score]
        if not filtered:
            return ""
        return format_memory_context(filtered)

    def _build_auto_context(self, request: ModelRequest[ContextT]) -> str:
        query = _latest_human_text(request.messages)
        if not query:
            return ""

        sections: list[str] = []
        if self._rag_retriever is not None and _has_auto(self._rag_mode):
            rag_context = self._retrieve_rag_context(query, request.runtime, limit=self._rag_auto_top_k)
            if rag_context:
                sections.append(rag_context)
        if self._memory_service is not None and _has_auto(self._memory_mode):
            memory_context = self._retrieve_memory_context(query, request.runtime, limit=self._memory_auto_top_k)
            if memory_context:
                sections.append(memory_context)
        return "\n\n".join(sections)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject relevant RAG and memory context into a model request.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with relevant retrieval context in the system
            message. If no context is retrieved, returns the original request.
        """
        context = self._build_auto_context(request)
        if not context:
            return request
        system_message = append_to_system_message(request.system_message, context)
        return request.override(system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject retrieval context before a synchronous model call."""
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Inject retrieval context before an asynchronous model call."""
        return await handler(self.modify_request(request))

    def _build_search_knowledge_base_tool(self) -> BaseTool:
        middleware = self

        def search_knowledge_base(query: str, runtime: ToolRuntime, top_k: int | None = None) -> str:
            """Search the configured knowledge base for relevant document chunks."""
            limit = _positive("top_k", top_k or middleware._rag_tool_top_k)
            context = middleware._retrieve_rag_context(query, runtime, limit=limit)
            return context or "No relevant knowledge-base chunks found."

        return StructuredTool.from_function(
            name="search_knowledge_base",
            func=search_knowledge_base,
            description=(
                "Search the configured RAG knowledge base. Use this when the "
                "answer may depend on indexed documents, uploaded files, or "
                "domain knowledge outside the current conversation."
            ),
        )

    def _build_search_memory_tool(self) -> BaseTool:
        middleware = self

        def search_memory(query: str, runtime: ToolRuntime, top_k: int | None = None) -> str:
            """Search long-term memory for relevant user or project context."""
            limit = _positive("top_k", top_k or middleware._memory_tool_top_k)
            context = middleware._retrieve_memory_context(query, runtime, limit=limit)
            return context or "No relevant long-term memory found."

        return StructuredTool.from_function(
            name="search_memory",
            func=search_memory,
            description=("Search long-term memory for user preferences, durable facts, project rules, prior decisions, corrections, and workflows."),
        )

    def _build_save_memory_tool(self) -> BaseTool:
        middleware = self

        def save_memory(
            content: str,
            runtime: ToolRuntime,
            memory_type: str = "fact",
            scope_type: str = "user",
            scope_id: str = "",
            summary: str = "",
            importance: float = 0.5,
            confidence: float = 0.8,
            visibility: str = "private",
        ) -> str:
            """Save a durable memory item with tenant and user metadata."""
            if middleware._memory_service is None:
                return "Long-term memory is not configured."
            resolved_scope = middleware._resolve_memory_scope(runtime)
            resolved_memory_type = cast("MemoryType", _literal_value("memory_type", memory_type, _MEMORY_TYPES))
            resolved_scope_type = cast("MemoryScopeType", _literal_value("scope_type", scope_type, _MEMORY_SCOPE_TYPES))
            resolved_visibility = cast("MemoryVisibility", _literal_value("visibility", visibility, _MEMORY_VISIBILITIES))
            resolved_scope_id = scope_id or _default_memory_scope_id(resolved_scope, resolved_scope_type)
            record = middleware._memory_service.save_candidate(
                MemoryWriteCandidate(
                    content=content,
                    memory_type=resolved_memory_type,
                    scope_type=resolved_scope_type,
                    scope_id=resolved_scope_id,
                    summary=summary,
                    importance=importance,
                    confidence=confidence,
                    visibility=resolved_visibility,
                    source_thread_id=_thread_id(runtime),
                ),
                tenant_id=resolved_scope.tenant_id,
                user_id=resolved_scope.user_id,
            )
            return f"Saved memory `{record.memory_id}`."

        return StructuredTool.from_function(
            name="save_memory",
            func=save_memory,
            description=(
                "Save durable long-term memory. Use only for stable preferences, "
                "facts, project rules, decisions, corrections, or workflows. "
                "Never save secrets or one-time transient details."
            ),
        )

    def _build_delete_memory_tool(self) -> BaseTool:
        middleware = self

        def delete_memory(memory_id: str, runtime: ToolRuntime) -> str:
            """Delete a visible long-term memory item."""
            if middleware._memory_service is None:
                return "Long-term memory is not configured."
            deleted = middleware._memory_service.delete(memory_id, scope=middleware._resolve_memory_scope(runtime))
            if not deleted:
                return f"Memory `{memory_id}` was not found or is not visible."
            return f"Deleted memory `{memory_id}`."

        return StructuredTool.from_function(
            name="delete_memory",
            func=delete_memory,
            description="Soft-delete a long-term memory item when it is wrong, outdated, or no longer useful.",
        )


def _runtime_values(runtime: object) -> dict[str, object]:
    values: dict[str, object] = {}
    state = getattr(runtime, "state", None)
    context = getattr(runtime, "context", None)
    config = getattr(runtime, "config", None)

    if isinstance(state, Mapping):
        _merge_mapping(values, state)
    if isinstance(context, Mapping):
        _merge_mapping(values, context)
    if isinstance(config, Mapping):
        metadata = config.get("metadata")
        configurable = config.get("configurable")
        if isinstance(metadata, Mapping):
            _merge_mapping(values, metadata)
        if isinstance(configurable, Mapping):
            _merge_mapping(values, configurable)
    return values


def _merge_mapping(target: dict[str, object], source: Mapping[object, object]) -> None:
    target.update({key: value for key, value in source.items() if isinstance(key, str)})


def _string_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if item is not None and str(item))
    return (str(value),)


def _memory_types(value: object) -> tuple[MemoryType, ...]:
    return tuple(cast("MemoryType", _literal_value("memory_type", item, _MEMORY_TYPES)) for item in _string_tuple(value))


def _memory_scope_types(value: object) -> tuple[MemoryScopeType, ...]:
    return tuple(cast("MemoryScopeType", _literal_value("scope_type", item, _MEMORY_SCOPE_TYPES)) for item in _string_tuple(value))


def _literal_value(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        msg = f"`{name}` must be one of: {allowed_values}."
        raise ValueError(msg)
    return value


def _positive(name: str, value: int) -> int:
    if value <= 0:
        msg = f"`{name}` must be positive."
        raise ValueError(msg)
    return value


def _has_auto(mode: RetrievalMode) -> bool:
    return mode in {"auto", "hybrid"}


def _has_tool(mode: RetrievalMode) -> bool:
    return mode in {"tool", "hybrid"}


def _chunk_score(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is not None:
        return chunk.rerank_score
    if chunk.fused_score:
        return chunk.fused_score
    if chunk.vector_score is not None:
        return chunk.vector_score
    if chunk.keyword_score is not None:
        return chunk.keyword_score
    return 0.0


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.text)
    return ""


def _default_memory_scope_id(scope: MemoryScope, scope_type: MemoryScopeType) -> str:
    if scope.scope_ids:
        return scope.scope_ids[0]
    if scope_type == "user" and scope.user_id is not None:
        return scope.user_id
    return scope.tenant_id


def _thread_id(runtime: object) -> str | None:
    values = _runtime_values(runtime)
    return _optional_string(values.get("thread_id"))


__all__ = [
    "MemoryScopeResolver",
    "RagScopeResolver",
    "RetrievalMiddleware",
    "RetrievalMode",
    "RuntimeContextDefaults",
    "format_rag_context",
    "resolve_memory_scope",
    "resolve_retrieval_scope",
]
