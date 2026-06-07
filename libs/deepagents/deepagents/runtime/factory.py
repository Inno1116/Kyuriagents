"""Factory for assembling a runnable Deep Agent from runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from deepagents.graph import create_deep_agent
from deepagents.memory import ElasticsearchMilvusMemoryIndexer, MemoryHybridSearcher, MemoryService, PostgresMemoryStore
from deepagents.middleware.retrieval import RetrievalMiddleware
from deepagents.profiles.harness.harness_profiles import GeneralPurposeSubagentProfile
from deepagents.rag import DashScopeTextReranker, ElasticsearchKeywordStore, HybridRAGRetriever, MilvusVectorStore, PostgresChunkTextHydrator
from deepagents.runtime.dashscope import EmbedQuery, create_dashscope_embed_query, create_dashscope_model
from deepagents.runtime.evidence import EvidencePackage
from deepagents.runtime.mcp import LoadedMCPTools, load_mcp_tools
from deepagents.tools import (
    PostgresToolAuditSink,
    ToolAuditSink,
    ToolDescriptor,
    ToolGovernanceMiddleware,
    ToolPolicy,
    ToolRegistry,
    default_tool_registry,
    merge_tool_sequences,
)
from deepagents.websearch import create_web_agent_tools, create_web_search_tools, web_search_tool_descriptors

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from deepagents.middleware.subagents import SubAgent
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
    tool_registry: ToolRegistry | None = None,
    tool_policy: ToolPolicy | None = None,
    tool_audit_sink: ToolAuditSink | None = None,
    mcp_tools: Sequence[BaseTool | Callable | dict[str, Any]] | LoadedMCPTools | None = None,
    mcp_descriptors: Sequence[ToolDescriptor] = (),
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
        tool_registry: Optional registry for tool descriptors.
        tool_policy: Optional policy for tool calls.
        tool_audit_sink: Optional audit sink.
        mcp_tools: Optional preloaded MCP tools. When omitted and MCP is
            enabled, tools are loaded from `config.mcp_config_path`.
        mcp_descriptors: Optional descriptors for preloaded MCP tools.
        checkpointer: Optional LangGraph checkpointer.
        store: Optional LangGraph store.
        system_prompt: Optional user system prompt.
        debug: Whether to enable LangGraph debug mode.
        name: Optional agent name.

    Returns:
        Compiled Deep Agent graph.
    """
    resolved_model = model if model is not None else create_dashscope_model(config)
    summarization_model = _create_summarization_model(config)
    resolved_embed_query = embed_query
    if config.enable_rag and rag_retriever is None:
        resolved_embed_query = resolved_embed_query or create_dashscope_embed_query(config)
        rag_retriever = _create_rag_retriever(config, resolved_embed_query)
    if config.enable_memory and memory_service is None:
        memory_service = _create_memory_service(
            config,
            hybrid_retriever=rag_retriever if config.enable_rag else None,
            embed_text=resolved_embed_query,
        )

    resolved_checkpointer = checkpointer
    resolved_store = store
    if config.enable_checkpointer and (resolved_checkpointer is None or resolved_store is None):
        pg_checkpointer, pg_store = _create_langgraph_postgres(config)
        resolved_checkpointer = resolved_checkpointer or pg_checkpointer
        resolved_store = resolved_store or pg_store

    use_information_subagents = config.enable_subagents and (config.enable_rag or config.enable_web_search)
    main_rag_mode = "off" if use_information_subagents and config.enable_rag else config.rag_mode
    retrieval = RetrievalMiddleware(
        rag_retriever=rag_retriever if config.enable_rag else None,
        memory_service=memory_service if config.enable_memory else None,
        rag_mode=main_rag_mode,
        memory_mode=config.memory_mode,
        defaults=config.retrieval_defaults(),
        memory_checkpoint_interval=config.memory_checkpoint_interval,
        memory_checkpoint_max_chars=config.memory_checkpoint_max_chars,
    )
    rag_subagent_tools: Sequence[BaseTool] = ()
    if use_information_subagents and config.enable_rag and rag_retriever is not None:
        rag_subagent_retrieval = RetrievalMiddleware(
            rag_retriever=rag_retriever,
            memory_service=None,
            rag_mode="tool",
            memory_mode="off",
            defaults=config.retrieval_defaults(),
            memory_checkpoint_interval=0,
        )
        rag_subagent_tools = rag_subagent_retrieval.tools
    web_subagent_tools = create_web_agent_tools(config) if use_information_subagents and config.enable_web_search else ()
    runtime_web_tools = create_web_search_tools(config) if config.enable_web_search and not use_information_subagents else ()
    resolved_tools, governance = _build_tool_runtime(
        config,
        native_tools=tools,
        runtime_tools=runtime_web_tools,
        runtime_descriptors=web_search_tool_descriptors(
            timeout_seconds=max(1, int(max(config.web_search_timeout_seconds, config.web_fetch_timeout_seconds, config.web_render_timeout_seconds)))
        )
        if config.enable_web_search
        else (),
        middleware_tools=(*retrieval.tools, *rag_subagent_tools),
        tool_registry=tool_registry,
        tool_policy=tool_policy,
        tool_audit_sink=tool_audit_sink,
        mcp_tools=mcp_tools,
        mcp_descriptors=mcp_descriptors,
    )
    information_subagents = _build_information_subagents(
        config,
        rag_tools=rag_subagent_tools,
        web_tools=web_subagent_tools,
        governance=governance,
    )
    resolved_middleware = [*middleware, retrieval]
    if governance is not None:
        resolved_middleware.append(governance)

    return create_deep_agent(
        model=resolved_model,
        tools=resolved_tools,
        system_prompt=system_prompt,
        middleware=resolved_middleware,
        subagents=information_subagents or None,
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False) if use_information_subagents else None,
        checkpointer=resolved_checkpointer,
        store=resolved_store,
        enable_summarization=config.enable_context_summarization,
        summarization_model=summarization_model,
        summarization_trigger=config.context_summary_trigger(),
        summarization_keep=config.context_summary_keep(),
        debug=debug,
        name=name,
    )


def _build_information_subagents(
    config: AgentRuntimeConfig,
    *,
    rag_tools: Sequence[BaseTool],
    web_tools: Sequence[BaseTool],
    governance: ToolGovernanceMiddleware | None,
) -> list[SubAgent]:
    subagents: list[SubAgent] = []
    if rag_tools:
        rag_agent: SubAgent = {
            "name": "rag-agent",
            "description": (
                "Use only for uploaded documents, private knowledge bases, or indexed local corpora. "
                "Do not use for public web research, current events, URLs, official websites, anime databases, "
                "Bangumi, Douban, Bilibili, MoeGirl, or other online facts; use web-agent for those."
            ),
            "system_prompt": _rag_agent_prompt(),
            "tools": rag_tools,
            "middleware": _subagent_governance_middleware(governance),
            "response_format": EvidencePackage,
        }
        subagents.append(rag_agent)
    if web_tools:
        web_agent: SubAgent = {
            "name": "web-agent",
            "description": (
                "Use for public internet research, current or recently changed facts, official web sources, URLs, "
                "web pages, Chinese web sources, anime/media databases, community encyclopedias, Bangumi, Douban, "
                "Bilibili, MoeGirl, and any task that asks whether something exists online."
            ),
            "system_prompt": _web_agent_prompt(config),
            "tools": web_tools,
            "middleware": _subagent_governance_middleware(governance),
            "response_format": EvidencePackage,
        }
        subagents.append(web_agent)
    return subagents


def _subagent_governance_middleware(governance: ToolGovernanceMiddleware | None) -> list[AgentMiddleware]:
    return cast("list[AgentMiddleware]", [governance] if governance is not None else [])


def _rag_agent_prompt() -> str:
    return """You are Kyuriagents' RAG evidence agent.

Your job is to verify the delegated question against the configured knowledge base and return a compact evidence package.

Rules:
- Use `search_knowledge_base` before making factual claims from uploaded or indexed documents.
- If the delegated task asks for public web research, URLs, current events, online communities, or public websites,
  do not pretend the local knowledge base can answer it. Record that the task should be delegated to `web-agent`
  in `missing` or `failures`.
- Rewrite the query when useful, but keep searches focused.
- Prefer concise, source-backed findings over broad summaries.
- Do not invent sources. If retrieval is empty or ambiguous, record that in `missing` or `failures`.
- Return only the structured evidence package requested by the response schema.

Evidence guidance:
- `conclusion` should be the shortest answer supported by the retrieved chunks.
- `findings` should be atomic claims.
- `sources` should include the document title/source URI or chunk identifier when available.
- `quote` should be a short supporting excerpt, not a full chunk."""


def _web_agent_prompt(config: AgentRuntimeConfig) -> str:
    return f"""You are Kyuriagents' web evidence agent.

Your job is to search the public web, decide which pages are worth opening, and return a compact evidence package for the main agent.

Research policy:
- Use `web_search` for search result discovery. Spend at most {config.web_agent_max_search_calls} search calls for one delegated task.
- Preserve the user's original language in at least one search query. For Chinese questions, search Chinese terms first,
  then add English queries only when they improve source coverage.
- Deduplicate URLs before fetching pages.
- Use `web_fetch_static` first for page reading. Open at most {config.web_fetch_max_pages} pages with static fetch.
- Static and rendered page tools return Markdown with `Quality flags`, text length, truncation status, and extracted content.
- Use `web_render_page` only when static fetch reports `empty_text`, `too_short`, `maybe_js_required`, `blocked_or_verification`,
  or when the extracted content clearly misses dynamic content needed for the delegated question.
- Render at most {config.web_render_max_pages} pages.
- Each fetched page is already capped at about {config.web_fetch_max_chars} extracted characters; summarize and quote only the useful parts.
- Prefer official, primary, or high-authority sources. Avoid low-value mirrors when better sources exist.
- Do not treat generic homepages, topic landing pages, dictionaries, or unrelated portal pages as evidence for a specific claim.
- Record blocked pages, timeouts, and weak evidence in `failures` or `missing`.

Output rules:
- Return only the structured evidence package requested by the response schema.
- `conclusion` should directly answer the delegated research task.
- Do not claim that no public information exists unless multiple targeted searches and fetched pages directly support that absence.
  Otherwise say that the search was inconclusive and list what is missing.
- `findings` should be evidence-backed and concise.
- `sources` should include title, URL, source_type=`web`, and a short quote.
- Do not include raw page dumps or long excerpts."""


def _create_summarization_model(config: AgentRuntimeConfig) -> BaseChatModel | None:
    if not config.enable_context_summarization or not config.context_summary_model:
        return None
    return create_dashscope_model(config, model_name=config.context_summary_model)


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
        chunk_hydrator=_create_rag_chunk_hydrator(config),
        reranker=_create_rag_reranker(config),
    )


def _create_rag_chunk_hydrator(config: AgentRuntimeConfig) -> PostgresChunkTextHydrator | None:
    if not config.postgres_dsn:
        return None
    return PostgresChunkTextHydrator(dsn=config.postgres_dsn)


def _create_rag_reranker(config: AgentRuntimeConfig) -> DashScopeTextReranker | None:
    if not config.rag_rerank_model:
        return None
    return DashScopeTextReranker(
        api_key=config.dashscope_api_key or "",
        model=config.rag_rerank_model,
        endpoint=config.rag_rerank_url,
        timeout_seconds=config.rag_rerank_timeout_seconds,
    )


def _create_memory_service(
    config: AgentRuntimeConfig,
    *,
    hybrid_retriever: HybridRAGRetriever | None = None,
    embed_text: EmbedQuery | None = None,
) -> MemoryService:
    if not config.postgres_dsn:
        missing = ", ".join(config.missing_for_memory())
        msg = f"Missing settings for memory runtime: {missing}."
        raise ValueError(msg)
    store = PostgresMemoryStore(dsn=config.postgres_dsn)
    hybrid_searcher = None
    indexer = None
    if hybrid_retriever is not None and embed_text is not None:
        memory_retriever = _create_memory_retriever(config, embed_text)
        hybrid_searcher = MemoryHybridSearcher(retriever=memory_retriever, store=store)
        indexer = ElasticsearchMilvusMemoryIndexer(
            es_index=config.memory_es_index,
            es_url=config.rag_es_url,
            milvus_collection=config.memory_milvus_collection,
            milvus_uri=config.rag_milvus_uri,
            milvus_token=config.rag_milvus_token,
            milvus_db=config.rag_milvus_db,
            embed_text=embed_text,
        )
    return MemoryService(store, hybrid_searcher=hybrid_searcher, indexer=indexer)


def _create_memory_retriever(config: AgentRuntimeConfig, embed_query: EmbedQuery) -> HybridRAGRetriever:
    return HybridRAGRetriever(
        vector_searcher=MilvusVectorStore(
            collection_name=config.memory_milvus_collection,
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token,
            db_name=config.rag_milvus_db,
            embed_query=embed_query,
        ),
        keyword_searcher=ElasticsearchKeywordStore(
            index=config.memory_es_index,
            url=config.rag_es_url,
        ),
    )


def _build_tool_runtime(
    config: AgentRuntimeConfig,
    *,
    native_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    runtime_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    runtime_descriptors: Sequence[ToolDescriptor],
    middleware_tools: Sequence[BaseTool],
    tool_registry: ToolRegistry | None,
    tool_policy: ToolPolicy | None,
    tool_audit_sink: ToolAuditSink | None,
    mcp_tools: Sequence[BaseTool | Callable | dict[str, Any]] | LoadedMCPTools | None,
    mcp_descriptors: Sequence[ToolDescriptor],
) -> tuple[list[BaseTool | Callable | dict[str, Any]], ToolGovernanceMiddleware | None]:
    registry = tool_registry.copy() if tool_registry is not None else default_tool_registry()
    resolved_mcp_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None
    resolved_mcp_descriptors: Sequence[ToolDescriptor] = mcp_descriptors
    if config.enable_mcp:
        loaded = load_mcp_tools(config) if mcp_tools is None else mcp_tools
        if isinstance(loaded, LoadedMCPTools):
            resolved_mcp_tools = loaded.tools
            resolved_mcp_descriptors = (*resolved_mcp_descriptors, *loaded.descriptors)
        else:
            resolved_mcp_tools = loaded

    for tool in native_tools or ():
        _register_tool_if_missing(registry, tool)
    for tool in runtime_tools:
        _register_tool_if_missing(registry, tool, source="runtime")
    for tool in middleware_tools:
        _register_tool_if_missing(registry, tool, source="runtime")
    registry.register_many(runtime_descriptors, replace_existing=True)
    registry.register_many(resolved_mcp_descriptors, replace_existing=True)

    governance = None
    if config.enable_tools:
        resolved_audit_sink = tool_audit_sink
        if resolved_audit_sink is None and config.enable_tool_audit and config.postgres_dsn:
            resolved_audit_sink = PostgresToolAuditSink(dsn=config.postgres_dsn)
        governance = ToolGovernanceMiddleware(
            registry=registry,
            policy=tool_policy or config.tool_policy(),
            audit_sink=resolved_audit_sink,
            defaults=config.tool_defaults(),
        )

    return merge_tool_sequences(native_tools, runtime_tools, resolved_mcp_tools), governance


def _register_tool_if_missing(
    registry: ToolRegistry,
    tool: BaseTool | Callable | dict[str, Any],
    *,
    source: str = "native",
) -> None:
    try:
        registry.register_tool(tool, source=cast("Any", source))
    except ValueError:
        return


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
