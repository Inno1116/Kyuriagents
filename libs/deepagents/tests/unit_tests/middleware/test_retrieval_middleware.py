from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.memory import InMemoryMemoryStore, MemoryRecord, MemoryScope, MemoryService
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

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


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


def test_retrieval_middleware_saves_memory_checkpoint_every_interval() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    middleware = RetrievalMiddleware(
        memory_service=service,
        rag_mode="off",
        memory_mode="tool",
        defaults=RuntimeContextDefaults(tenant_id="tenant-a", user_id="user-1", thread_id="thread-1"),
        memory_checkpoint_interval=2,
        memory_checkpoint_max_chars=800,
    )
    runtime = _Runtime(
        config={
            "configurable": {
                "tenant_id": "tenant-a",
                "user_id": "user-1",
                "thread_id": "thread-1",
                "memory_scope_types": ["user"],
                "memory_scope_ids": ["user-1"],
            }
        }
    )
    state: AgentState = {
        "messages": [
            HumanMessage(content="Remember that password=super-secret must not be saved raw.", id="human-1"),
            AIMessage(content="I will keep secrets out of long-term memory.", id="ai-1"),
            HumanMessage(content="Also summarize the RAG testing plan.", id="human-2"),
            AIMessage(content="We will test RAG, then Memory, then tools.", id="ai-2"),
        ]
    }

    middleware.after_agent(state, runtime)
    middleware.after_agent(state, runtime)

    memories = service.list_memories(scope=MemoryScope(tenant_id="tenant-a", user_id="user-1", active_only=False), limit=10)
    assert len(memories) == 1
    memory = memories[0]
    assert memory.memory_type == "summary"
    assert memory.scope_type == "user"
    assert memory.scope_id == "user-1"
    assert memory.source_thread_id == "thread-1"
    assert memory.source_message_ids == ("human-1", "ai-1", "human-2", "ai-2")
    assert "Automatic conversation checkpoint for user turns 1-2." in memory.content
    assert "Recent user goals:" in memory.content
    assert "Assistant outcomes:" in memory.content
    assert "User asked/needed: Remember that password=[redacted] must not be saved raw." in memory.content
    assert "password=[redacted]" in memory.content
    assert "super-secret" not in memory.content
    assert "- user:" not in memory.content
    assert "- assistant:" not in memory.content


def test_retrieval_middleware_checkpoint_deduplicates_repeated_assistant_messages() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    middleware = RetrievalMiddleware(
        memory_service=service,
        rag_mode="off",
        memory_mode="tool",
        defaults=RuntimeContextDefaults(tenant_id="tenant-a", user_id="user-1", thread_id="thread-1"),
        memory_checkpoint_interval=2,
        memory_checkpoint_max_chars=800,
    )
    runtime = _Runtime(
        config={
            "configurable": {
                "tenant_id": "tenant-a",
                "user_id": "user-1",
                "thread_id": "thread-1",
                "memory_scope_types": ["user"],
                "memory_scope_ids": ["user-1"],
            }
        }
    )
    state: AgentState = {
        "messages": [
            HumanMessage(content="My name is Jack could you remember me?", id="human-1"),
            AIMessage(content="Got it, your name is Jack.", id="ai-1"),
            AIMessage(content="Got it, your name is Jack.", id="ai-1-duplicate"),
            HumanMessage(content="Do you remember my name?", id="human-2"),
            AIMessage(content="Yes, your name is Jack.", id="ai-2"),
            AIMessage(content="Yes, your name is Jack.", id="ai-2-duplicate"),
        ]
    }

    middleware.after_agent(state, runtime)

    memories = service.list_memories(scope=MemoryScope(tenant_id="tenant-a", user_id="user-1", active_only=False), limit=10)
    assert len(memories) == 1
    content = memories[0].content
    assert "User identified themselves as Jack." in content
    assert content.count("Got it, your name is Jack.") == 1
    assert content.count("Yes, your name is Jack.") == 1


def test_retrieval_middleware_skips_checkpoint_before_interval() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store)
    middleware = RetrievalMiddleware(
        memory_service=service,
        rag_mode="off",
        memory_mode="tool",
        defaults=RuntimeContextDefaults(tenant_id="tenant-a", user_id="user-1"),
        memory_checkpoint_interval=2,
    )

    middleware.after_agent({"messages": [HumanMessage(content="Only one user turn.")]}, _Runtime())

    memories = service.list_memories(scope=MemoryScope(tenant_id="tenant-a", user_id="user-1", active_only=False), limit=10)
    assert memories == []


def test_format_rag_context_handles_empty_results() -> None:
    assert "No relevant knowledge-base chunks" in format_rag_context([])
