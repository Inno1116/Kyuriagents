"""Experimental dynamic memory primitives for Deep Agents.

This package defines storage-neutral contracts for short-term summaries and
long-term user/project memory. LangMem can be layered on top for extraction and
maintenance, while Deep Agents keeps tenant isolation, audit metadata, and
deployment schemas under its own control.
"""

from deepagents.memory.compression import CompressedMemoryContext, MemoryContextBudget, MemoryContextCompressor
from deepagents.memory.in_memory import InMemoryMemoryStore
from deepagents.memory.indexing import (
    ElasticsearchMilvusMemoryIndexer,
    MemoryHybridSearcher,
    MemoryIndexer,
    memory_records_to_chunks,
)
from deepagents.memory.langmem import build_langmem_namespace, create_langmem_memory_tools
from deepagents.memory.maintenance import MemoryCompactionConfig, MemoryCompactionResult, MemoryMaintenanceService
from deepagents.memory.postgres import PostgresMemoryStore
from deepagents.memory.service import MemoryService, format_memory_context
from deepagents.memory.types import (
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
    MemoryWriteCandidate,
)

__all__ = [
    "CompressedMemoryContext",
    "ElasticsearchMilvusMemoryIndexer",
    "InMemoryMemoryStore",
    "MemoryCompactionConfig",
    "MemoryCompactionResult",
    "MemoryContextBudget",
    "MemoryContextCompressor",
    "MemoryHybridSearcher",
    "MemoryIndexer",
    "MemoryMaintenanceService",
    "MemoryRecord",
    "MemoryScope",
    "MemorySearchResult",
    "MemoryService",
    "MemoryStore",
    "MemoryWriteCandidate",
    "PostgresMemoryStore",
    "build_langmem_namespace",
    "create_langmem_memory_tools",
    "format_memory_context",
    "memory_records_to_chunks",
]
