# Deep Agents Dynamic Memory

This package contains storage-neutral primitives for long-term agent memory.

The intended production layering is:

1. LangGraph short-term memory keeps thread state and checkpoints in PostgreSQL.
2. LangMem extracts and manages durable memory candidates.
3. Deep Agents stores authoritative memory metadata, audit events, and access rules in PostgreSQL.
4. Searchable memory text is indexed as `source_type="memory"` through the same Milvus and Elasticsearch path used by RAG.

The existing `deepagents.middleware.memory.MemoryMiddleware` still loads
human-edited AGENTS.md files. This package is for dynamic memories learned from
conversations and project work.

Use `MemoryRecord.to_document_chunk()` when indexing memory into the hybrid RAG
pipeline. Use `MemoryService.build_context()` to inject only relevant Top-K
memories into a prompt. Use `create_langmem_memory_tools()` when wiring LangMem
tools into a LangGraph agent.
