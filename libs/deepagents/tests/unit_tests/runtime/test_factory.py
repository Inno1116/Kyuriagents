from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from deepagents.memory import InMemoryMemoryStore, MemoryRecord, MemoryService
from deepagents.rag import ChunkMetadata, DocumentChunk, HybridRAGRetriever, InMemoryKeywordStore, InMemoryVectorStore
from deepagents.runtime import AgentRuntimeConfig, create_kyuri_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def _embed_query(query: str) -> tuple[float]:
    return (1.0 if "postgres" in query.lower() else 0.0,)


def _retriever() -> HybridRAGRetriever:
    chunk = DocumentChunk(
        text="PostgreSQL stores durable memory metadata.",
        metadata=ChunkMetadata(
            chunk_id="chunk-1",
            tenant_id="tenant-a",
            kb_id="kb-main",
            doc_id="doc-1",
            doc_version="version-1",
            chunk_index=0,
            content_hash="hash-1",
            source_type="md",
            source_uri="/docs/postgres.md",
        ),
        embedding=(1.0,),
        keywords=("postgres", "memory"),
    )
    return HybridRAGRetriever(
        vector_searcher=InMemoryVectorStore([chunk], embed_query=_embed_query),
        keyword_searcher=InMemoryKeywordStore([chunk]),
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
                    content="The user wants PostgreSQL setup steps kept concrete.",
                    tags=("postgres",),
                )
            ]
        )
    )


def test_create_kyuri_agent_wires_retrieval_middleware_with_injected_components() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Done.")]))
    config = AgentRuntimeConfig(
        tenant_id="tenant-a",
        user_id="user-1",
        enable_checkpointer=False,
        rag_mode="auto",
        memory_mode="auto",
    )

    agent = create_kyuri_agent(
        config,
        model=model,
        rag_retriever=_retriever(),
        memory_service=_memory_service(),
    )
    agent.invoke({"messages": [HumanMessage(content="How do we set up postgres memory?")]})

    content = model.call_history[0]["messages"][0].text
    assert "<retrieved_knowledge_base>" in content
    assert "PostgreSQL stores durable memory metadata" in content
    assert "<agent_long_term_memory>" in content
    assert "PostgreSQL setup steps kept concrete" in content
