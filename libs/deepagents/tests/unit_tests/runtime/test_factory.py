from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from deepagents.memory import InMemoryMemoryStore, MemoryRecord, MemoryService
from deepagents.rag import ChunkMetadata, DocumentChunk, HybridRAGRetriever, InMemoryKeywordStore, InMemoryVectorStore
from deepagents.runtime import AgentRuntimeConfig, create_kyuri_agent
from deepagents.tools import ToolDescriptor
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

    agent = cast(
        "Any",
        create_kyuri_agent(
            config,
            model=model,
            rag_retriever=_retriever(),
            memory_service=_memory_service(),
        ),
    )
    agent.invoke({"messages": [HumanMessage(content="How do we set up postgres memory?")]})

    content = model.call_history[0]["messages"][0].text
    assert "<retrieved_knowledge_base>" in content
    assert "PostgreSQL stores durable memory metadata" in content
    assert "<agent_long_term_memory>" in content
    assert "PostgreSQL setup steps kept concrete" in content


def test_create_kyuri_agent_adds_preloaded_mcp_tools() -> None:
    def lookup_status(query: str) -> str:
        """Look up status."""
        return f"status: {query}"

    model = GenericFakeChatModel(messages=iter([AIMessage(content="Done.")]))
    config = AgentRuntimeConfig(
        enable_rag=False,
        enable_memory=False,
        enable_checkpointer=False,
        enable_mcp=True,
    )

    agent = cast(
        "Any",
        create_kyuri_agent(
            config,
            model=model,
            mcp_tools=[StructuredTool.from_function(name="status_lookup", func=lookup_status)],
            mcp_descriptors=[ToolDescriptor(name="status_lookup", source="mcp", risk="read_only")],
        ),
    )

    assert "status_lookup" in agent.nodes["tools"].bound._tools_by_name


def test_create_kyuri_agent_passes_context_summarization_settings() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Done.")]))
    config = AgentRuntimeConfig(
        enable_rag=False,
        enable_memory=False,
        enable_checkpointer=False,
        enable_context_summarization=True,
        context_summary_trigger_messages=32,
        context_summary_keep_messages=10,
    )

    with patch("deepagents.runtime.factory.create_deep_agent", return_value=MagicMock()) as mock_create:
        create_kyuri_agent(config, model=model)

    kwargs = mock_create.call_args.kwargs
    assert kwargs["enable_summarization"] is True
    assert kwargs["summarization_trigger"] == ("messages", 32)
    assert kwargs["summarization_keep"] == ("messages", 10)


def test_create_kyuri_agent_uses_dedicated_context_summary_model() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Done.")]))
    summary_model = GenericFakeChatModel(messages=iter([AIMessage(content="Summary.")]))
    config = AgentRuntimeConfig(
        enable_rag=False,
        enable_memory=False,
        enable_checkpointer=False,
        enable_context_summarization=True,
        context_summary_model="qwen-turbo",
    )

    with (
        patch("deepagents.runtime.factory.create_dashscope_model", return_value=summary_model) as mock_dashscope,
        patch("deepagents.runtime.factory.create_deep_agent", return_value=MagicMock()) as mock_create,
    ):
        create_kyuri_agent(config, model=model)

    mock_dashscope.assert_called_once_with(config, model_name="qwen-turbo")
    assert mock_create.call_args.kwargs["model"] is model
    assert mock_create.call_args.kwargs["summarization_model"] is summary_model
