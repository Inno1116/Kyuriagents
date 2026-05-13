from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.memory import InMemoryMemoryStore, MemoryRecord, MemoryService
from deepagents.middleware.retrieval import (
    RetrievalMiddleware,
    RuntimeContextDefaults,
    format_rag_context,
    resolve_memory_scope,
    resolve_retrieval_scope,
)
from deepagents.rag import (
    ChunkMetadata,
    DocumentChunk,
    HybridRAGRetriever,
    HybridSearchConfig,
    InMemoryKeywordStore,
    InMemoryVectorStore,
    RetrievalScope,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


class _Runtime:
    def __init__(self, *, config: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.context = context or {}


def _metadata(chunk_id: str, *, tenant_id: str = "tenant-a", user_id: str | None = None) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        user_id=user_id,
        kb_id="kb-main",
        doc_id=f"doc-{chunk_id}",
        doc_version=f"version-{chunk_id}",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
        source_type="md",
        source_uri=f"/docs/{chunk_id}.md",
        title=f"title {chunk_id}",
        section_path="Runtime / Retrieval",
        tags=("runtime",),
        embedding_model="test-embedding",
        embedding_version="v1",
    )


def _embed_query(query: str) -> tuple[float, float]:
    lowered = query.lower()
    return (
        1.0 if "auth" in lowered or "login" in lowered else 0.0,
        1.0 if "postgres" in lowered or "metadata" in lowered else 0.0,
    )


def _rag_retriever() -> HybridRAGRetriever:
    chunks = [
        DocumentChunk(
            text="Authentication and login tokens are configured here.",
            metadata=_metadata("auth-1"),
            embedding=(1.0, 0.0),
            keywords=("auth", "login"),
        ),
        DocumentChunk(
            text="PostgreSQL stores relational metadata for deployments.",
            metadata=_metadata("postgres-1"),
            embedding=(0.0, 1.0),
            keywords=("postgres", "metadata"),
        ),
        DocumentChunk(
            text="Other tenants must not leak into retrieval.",
            metadata=_metadata("other-tenant", tenant_id="tenant-b"),
            embedding=(1.0, 1.0),
            keywords=("auth", "postgres"),
        ),
    ]
    return HybridRAGRetriever(
        vector_searcher=InMemoryVectorStore(chunks, embed_query=_embed_query),
        keyword_searcher=InMemoryKeywordStore(chunks),
        config=HybridSearchConfig(top_k=3),
    )


def _memory_service() -> MemoryService:
    return MemoryService(
        InMemoryMemoryStore(
            [
                MemoryRecord(
                    memory_id="mem-1",
                    tenant_id="tenant-a",
                    user_id="user-1",
                    scope_type="user",
                    scope_id="user-1",
                    memory_type="preference",
                    content="The user prefers concise PostgreSQL deployment notes.",
                    importance=0.9,
                    tags=("postgres",),
                ),
                MemoryRecord(
                    memory_id="mem-hidden",
                    tenant_id="tenant-a",
                    user_id="user-2",
                    scope_type="user",
                    scope_id="user-2",
                    memory_type="preference",
                    content="This other user's preference must stay hidden.",
                ),
            ]
        )
    )


def test_resolve_scopes_from_runtime_config() -> None:
    runtime = _Runtime(
        config={
            "configurable": {
                "tenant_id": "tenant-a",
                "user_id": "user-1",
                "kb_ids": ["kb-main"],
                "memory_scope_types": ["user"],
            }
        }
    )

    rag_scope = resolve_retrieval_scope(runtime)
    memory_scope = resolve_memory_scope(runtime)

    assert rag_scope == RetrievalScope(tenant_id="tenant-a", user_id="user-1", kb_ids=("kb-main",))
    assert memory_scope.tenant_id == "tenant-a"
    assert memory_scope.user_id == "user-1"
    assert memory_scope.scope_types == ("user",)


def test_retrieval_middleware_auto_injects_rag_and_memory_context() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Done.")]))
    middleware = RetrievalMiddleware(
        rag_retriever=_rag_retriever(),
        memory_service=_memory_service(),
        rag_mode="auto",
        memory_mode="auto",
        defaults=RuntimeContextDefaults(tenant_id="tenant-a", user_id="user-1"),
    )
    agent = create_agent(model=model, middleware=[middleware])

    agent.invoke({"messages": [HumanMessage(content="How should auth and postgres metadata work?")]})

    system_message = model.call_history[0]["messages"][0]
    content = system_message.text
    assert "<retrieved_knowledge_base>" in content
    assert "Authentication and login tokens" in content
    assert "<agent_long_term_memory>" in content
    assert "concise PostgreSQL deployment notes" in content
    assert "other user's preference" not in content


def test_search_knowledge_base_tool_uses_rag_retriever() -> None:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge_base",
                            "args": {"query": "auth login", "top_k": 2},
                            "id": "call_search_kb",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
    )
    middleware = RetrievalMiddleware(
        rag_retriever=_rag_retriever(),
        rag_mode="tool",
        memory_mode="off",
        defaults=RuntimeContextDefaults(tenant_id="tenant-a"),
    )
    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage(content="Search the knowledge base")]})

    tool_messages = [message for message in result["messages"] if message.type == "tool"]
    assert len(tool_messages) == 1
    assert "<retrieved_knowledge_base>" in tool_messages[0].text
    assert "Authentication and login tokens" in tool_messages[0].text
    assert "Other tenants" not in tool_messages[0].text


def test_memory_tools_search_save_and_delete() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save_memory",
                            "args": {
                                "content": "The user wants OmniEval results reported with exact dates.",
                                "memory_type": "preference",
                            },
                            "id": "call_save_memory",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_memory",
                            "args": {"query": "OmniEval dates"},
                            "id": "call_search_memory",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
    )
    middleware = RetrievalMiddleware(
        memory_service=service,
        rag_mode="off",
        memory_mode="tool",
        defaults=RuntimeContextDefaults(tenant_id="tenant-a", user_id="user-1"),
    )
    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage(content="Remember this preference")]})

    tool_messages = [message for message in result["messages"] if message.type == "tool"]
    assert "Saved memory" in tool_messages[0].text
    assert "OmniEval results reported with exact dates" in tool_messages[1].text


def test_format_rag_context_handles_empty_results() -> None:
    assert "No relevant knowledge-base chunks" in format_rag_context([])
