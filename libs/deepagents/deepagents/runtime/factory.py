"""Factory for assembling a runnable Deep Agent from runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from deepagents.graph import create_deep_agent
from deepagents.memory import MemoryService, PostgresMemoryStore
from deepagents.middleware.retrieval import RetrievalMiddleware
from deepagents.rag import ElasticsearchKeywordStore, HybridRAGRetriever, MilvusVectorStore
from deepagents.runtime.dashscope import EmbedQuery, create_dashscope_embed_query, create_dashscope_model

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from deepagents.runtime.config import AgentRuntimeConfig


def create_kyuri_agent(
    config: AgentRuntimeConfig,
    *,
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    rag_retriever: HybridRAGRetriever | None = None,
    memory_service: MemoryService | None = None,
    embed_query: EmbedQuery | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    system_prompt: str | None = None,
    debug: bool = False,
    name: str | None = None,
) -> object:
    """Create a Deep Agent with runtime RAG and memory wiring.

    Args:
        config: Runtime configuration.
        model: Optional prebuilt model. When omitted, DashScope is used.
        tools: Additional user tools.
        middleware: Additional middleware before retrieval wiring.
        rag_retriever: Optional prebuilt RAG retriever.
        memory_service: Optional prebuilt memory service.
        embed_query: Optional query embedding function.
        checkpointer: Optional LangGraph checkpointer.
        store: Optional LangGraph store.
        system_prompt: Optional user system prompt.
        debug: Whether to enable LangGraph debug mode.
        name: Optional agent name.

    Returns:
        Compiled Deep Agent graph.
    """
    resolved_model = model if model is not None else create_dashscope_model(config)
    resolved_embed_query = embed_query
    if config.enable_rag and rag_retriever is None:
        resolved_embed_query = resolved_embed_query or create_dashscope_embed_query(config)
        rag_retriever = _create_rag_retriever(config, resolved_embed_query)
    if config.enable_memory and memory_service is None:
        memory_service = _create_memory_service(config)

    resolved_checkpointer = checkpointer
    resolved_store = store
    if config.enable_checkpointer and (resolved_checkpointer is None or resolved_store is None):
        pg_checkpointer, pg_store = _create_langgraph_postgres(config)
        resolved_checkpointer = resolved_checkpointer or pg_checkpointer
        resolved_store = resolved_store or pg_store

    retrieval = RetrievalMiddleware(
        rag_retriever=rag_retriever if config.enable_rag else None,
        memory_service=memory_service if config.enable_memory else None,
        rag_mode=config.rag_mode,
        memory_mode=config.memory_mode,
        defaults=config.retrieval_defaults(),
    )

    return create_deep_agent(
        model=resolved_model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[*middleware, retrieval],
        checkpointer=resolved_checkpointer,
        store=resolved_store,
        debug=debug,
        name=name,
    )


def _create_rag_retriever(config: AgentRuntimeConfig, embed_query: EmbedQuery) -> HybridRAGRetriever:
    return HybridRAGRetriever(
        vector_searcher=MilvusVectorStore(
            collection_name=config.rag_milvus_collection,
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token,
            db_name=config.rag_milvus_db,
            embed_query=embed_query,
        ),
        keyword_searcher=ElasticsearchKeywordStore(
            index=config.rag_es_index,
            url=config.rag_es_url,
        ),
    )


def _create_memory_service(config: AgentRuntimeConfig) -> MemoryService:
    if not config.postgres_dsn:
        missing = ", ".join(config.missing_for_memory())
        msg = f"Missing settings for memory runtime: {missing}."
        raise ValueError(msg)
    return MemoryService(PostgresMemoryStore(dsn=config.postgres_dsn))


def _create_langgraph_postgres(config: AgentRuntimeConfig) -> tuple[Checkpointer, BaseStore]:
    if not config.postgres_dsn:
        missing = ", ".join(config.missing_for_memory())
        msg = f"Missing settings for LangGraph PostgreSQL runtime: {missing}."
        raise ValueError(msg)
    try:
        import psycopg  # noqa: PLC0415
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
        from langgraph.store.postgres import PostgresStore  # noqa: PLC0415
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `deepagents[memory]` to use PostgreSQL checkpointer/store."
        raise ImportError(msg) from exc

    connect = cast("Any", psycopg.connect)
    checkpointer_connection = connect(config.postgres_dsn, autocommit=True, row_factory=dict_row)
    store_connection = connect(config.postgres_dsn, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(checkpointer_connection)
    runtime_store = PostgresStore(store_connection)
    return checkpointer, runtime_store


__all__ = ["create_kyuri_agent"]
