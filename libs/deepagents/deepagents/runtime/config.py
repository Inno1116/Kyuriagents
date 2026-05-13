"""Configuration model for assembling a runnable Deep Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from deepagents.middleware.retrieval import RetrievalMode, RuntimeContextDefaults
from deepagents.tools import (
    DEFAULT_ALLOWED_RISKS,
    DEFAULT_CONFIRMATION_RISKS,
    ToolContextDefaults,
    ToolPolicy,
    ToolRisk,
    parse_tool_names,
    parse_tool_risks,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True, kw_only=True)
class AgentRuntimeConfig:
    """Runtime configuration for wiring model, RAG, memory, and persistence.

    Args:
        tenant_id: Tenant or organization identifier.
        user_id: Optional user identifier.
        thread_id: Optional default thread identifier.
        dashscope_api_key: DashScope API key.
        dashscope_base_url: OpenAI-compatible DashScope base URL.
        chat_model: Chat model name.
        embedding_model: Embedding model name.
        embedding_dimensions: Optional embedding dimension override.
        postgres_dsn: Application PostgreSQL DSN.
        postgres_admin_dsn: Optional admin PostgreSQL DSN for database creation.
        postgres_database: Database name to create during bootstrap.
        enable_rag: Whether to wire RAG when dependencies are available.
        enable_memory: Whether to wire long-term memory.
        enable_checkpointer: Whether to wire LangGraph PostgreSQL persistence.
        rag_mode: RAG middleware mode.
        memory_mode: Long-term memory middleware mode.
        rag_es_url: Elasticsearch URL.
        rag_es_index: Elasticsearch index.
        rag_milvus_uri: Milvus URI.
        rag_milvus_collection: Milvus collection name.
        rag_milvus_db: Optional Milvus database.
        rag_milvus_token: Optional Milvus token.
        rag_kb_ids: Optional default knowledge-base filters.
        enable_tools: Whether to enable tool governance middleware.
        enable_mcp: Whether to load MCP tools during runtime startup.
        tool_allowed_risks: Risk classes allowed to execute.
        tool_confirmation_risks: Risk classes that require confirmation.
        tool_allow_requires_confirmation: Whether confirmation-gated tools may execute.
        tool_allowed_names: Optional tool allow-list.
        tool_denied_names: Explicit tool deny-list.
        enable_tool_audit: Whether tool calls should be audited.
        mcp_config_path: Optional MCP JSON config path.
        mcp_tool_name_prefix: Whether descriptor matching expects server-prefixed MCP tool names.
    """

    tenant_id: str = "default"
    user_id: str | None = None
    thread_id: str | None = None
    dashscope_api_key: str | None = None
    dashscope_base_url: str = _DASHSCOPE_BASE_URL
    chat_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int | None = None
    postgres_dsn: str | None = None
    postgres_admin_dsn: str | None = None
    postgres_database: str = "deepagents"
    enable_rag: bool = True
    enable_memory: bool = True
    enable_checkpointer: bool = True
    rag_mode: RetrievalMode = "tool"
    memory_mode: RetrievalMode = "hybrid"
    rag_es_url: str = "http://localhost:9200"
    rag_es_index: str = "rag_chunks"
    rag_milvus_uri: str = "http://localhost:19530"
    rag_milvus_collection: str = "rag_chunks"
    rag_milvus_db: str | None = None
    rag_milvus_token: str | None = None
    rag_kb_ids: tuple[str, ...] = ()
    enable_tools: bool = True
    enable_mcp: bool = False
    tool_allowed_risks: frozenset[ToolRisk] = DEFAULT_ALLOWED_RISKS
    tool_confirmation_risks: frozenset[ToolRisk] = DEFAULT_CONFIRMATION_RISKS
    tool_allow_requires_confirmation: bool = False
    tool_allowed_names: frozenset[str] = frozenset()
    tool_denied_names: frozenset[str] = frozenset()
    enable_tool_audit: bool = True
    mcp_config_path: str | None = None
    mcp_tool_name_prefix: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AgentRuntimeConfig:
        """Create runtime config from environment variables.

        Args:
            tenant_id: Optional explicit tenant override.
            user_id: Optional explicit user override.
            thread_id: Optional explicit thread override.
            env: Environment mapping for tests or custom loaders.

        Returns:
            Parsed runtime configuration.
        """
        source = env if env is not None else os.environ
        return cls(
            tenant_id=tenant_id or _env(source, "DEEPAGENTS_TENANT_ID", "TENANT_ID", default="default"),
            user_id=user_id or _optional_env(source, "DEEPAGENTS_USER_ID", "USER_ID"),
            thread_id=thread_id or _optional_env(source, "DEEPAGENTS_THREAD_ID", "THREAD_ID"),
            dashscope_api_key=_optional_env(source, "DASHSCOPE_API_KEY"),
            dashscope_base_url=_env(source, "DASHSCOPE_BASE_URL", default=_DASHSCOPE_BASE_URL),
            chat_model=_env(source, "DASHSCOPE_CHAT_MODEL", "DEEPAGENTS_CHAT_MODEL", default="qwen-plus"),
            embedding_model=_env(source, "DASHSCOPE_EMBEDDING_MODEL", "DEEPAGENTS_EMBEDDING_MODEL", default="text-embedding-v3"),
            embedding_dimensions=_optional_int_env(source, "DASHSCOPE_EMBEDDING_DIMENSIONS", "DEEPAGENTS_EMBEDDING_DIMENSIONS"),
            postgres_dsn=_optional_env(source, "DEEPAGENTS_POSTGRES_DSN", "MEMORY_POSTGRES_DSN", "RAG_POSTGRES_DSN"),
            postgres_admin_dsn=_optional_env(source, "DEEPAGENTS_POSTGRES_ADMIN_DSN", "POSTGRES_ADMIN_DSN"),
            postgres_database=_env(source, "DEEPAGENTS_POSTGRES_DATABASE", "POSTGRES_DATABASE", default="deepagents"),
            enable_rag=_bool_env(source, "DEEPAGENTS_ENABLE_RAG", default=True),
            enable_memory=_bool_env(source, "DEEPAGENTS_ENABLE_MEMORY", default=True),
            enable_checkpointer=_bool_env(source, "DEEPAGENTS_ENABLE_CHECKPOINTER", default=True),
            rag_mode=_retrieval_mode(_env(source, "DEEPAGENTS_RAG_MODE", default="tool")),
            memory_mode=_retrieval_mode(_env(source, "DEEPAGENTS_MEMORY_MODE", default="hybrid")),
            rag_es_url=_env(source, "RAG_ES_URL", default="http://localhost:9200"),
            rag_es_index=_env(source, "RAG_ES_INDEX", default="rag_chunks"),
            rag_milvus_uri=_env(source, "RAG_MILVUS_URI", default="http://localhost:19530"),
            rag_milvus_collection=_env(source, "RAG_MILVUS_COLLECTION", default="rag_chunks"),
            rag_milvus_db=_optional_env(source, "RAG_MILVUS_DB"),
            rag_milvus_token=_optional_env(source, "RAG_MILVUS_TOKEN"),
            rag_kb_ids=_tuple_env(source, "RAG_KB_IDS", "DEEPAGENTS_RAG_KB_IDS"),
            enable_tools=_bool_env(source, "DEEPAGENTS_ENABLE_TOOLS", default=True),
            enable_mcp=_bool_env(source, "DEEPAGENTS_ENABLE_MCP", default=False),
            tool_allowed_risks=parse_tool_risks(source.get("DEEPAGENTS_TOOL_ALLOWED_RISKS"), default=DEFAULT_ALLOWED_RISKS),
            tool_confirmation_risks=parse_tool_risks(source.get("DEEPAGENTS_TOOL_CONFIRMATION_RISKS"), default=DEFAULT_CONFIRMATION_RISKS),
            tool_allow_requires_confirmation=_bool_env(source, "DEEPAGENTS_TOOL_ALLOW_REQUIRES_CONFIRMATION", default=False),
            tool_allowed_names=parse_tool_names(source.get("DEEPAGENTS_TOOL_ALLOWED_NAMES")),
            tool_denied_names=parse_tool_names(source.get("DEEPAGENTS_TOOL_DENIED_NAMES")),
            enable_tool_audit=_bool_env(source, "DEEPAGENTS_ENABLE_TOOL_AUDIT", default=True),
            mcp_config_path=_optional_env(source, "DEEPAGENTS_MCP_CONFIG_PATH"),
            mcp_tool_name_prefix=_bool_env(source, "DEEPAGENTS_MCP_TOOL_NAME_PREFIX", default=False),
        )

    def retrieval_defaults(self) -> RuntimeContextDefaults:
        """Return defaults consumed by `RetrievalMiddleware`.

        Returns:
            Runtime context defaults for tenant, user, and knowledge bases.
        """
        return RuntimeContextDefaults(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            kb_ids=self.rag_kb_ids,
        )

    def missing_for_model(self) -> tuple[str, ...]:
        """Return missing settings required to construct the DashScope model."""
        if self.dashscope_api_key:
            return ()
        return ("DASHSCOPE_API_KEY",)

    def missing_for_rag(self) -> tuple[str, ...]:
        """Return missing settings required to construct RAG components."""
        missing = list(self.missing_for_model())
        if not self.rag_es_url:
            missing.append("RAG_ES_URL")
        if not self.rag_milvus_uri:
            missing.append("RAG_MILVUS_URI")
        return tuple(missing)

    def missing_for_memory(self) -> tuple[str, ...]:
        """Return missing settings required to construct memory components."""
        if self.postgres_dsn:
            return ()
        return ("DEEPAGENTS_POSTGRES_DSN",)

    def tool_policy(self) -> ToolPolicy:
        """Return the configured tool policy."""
        return ToolPolicy(
            allowed_risks=self.tool_allowed_risks,
            confirmation_risks=self.tool_confirmation_risks,
            allow_requires_confirmation=self.tool_allow_requires_confirmation,
            allowed_tools=self.tool_allowed_names,
            denied_tools=self.tool_denied_names,
        )

    def tool_defaults(self) -> ToolContextDefaults:
        """Return defaults consumed by `ToolGovernanceMiddleware`."""
        return ToolContextDefaults(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
        )


def _env(source: Mapping[str, str], *names: str, default: str) -> str:
    for name in names:
        value = source.get(name)
        if value:
            return value
    return default


def _optional_env(source: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = source.get(name)
        if value:
            return value
    return None


def _bool_env(source: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = source.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _optional_int_env(source: Mapping[str, str], *names: str) -> int | None:
    value = _optional_env(source, *names)
    if value is None:
        return None
    return int(value)


def _tuple_env(source: Mapping[str, str], *names: str) -> tuple[str, ...]:
    value = _optional_env(source, *names)
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _retrieval_mode(value: str) -> RetrievalMode:
    if value not in {"off", "auto", "tool", "hybrid"}:
        msg = "`RetrievalMode` must be one of: off, auto, tool, hybrid."
        raise ValueError(msg)
    return cast("RetrievalMode", value)
