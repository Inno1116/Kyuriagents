"""Experimental dynamic memory primitives for Deep Agents.

This package defines storage-neutral contracts for short-term summaries and
long-term user/project memory. LangMem can be layered on top for extraction and
maintenance, while Deep Agents keeps tenant isolation, audit metadata, and
deployment schemas under its own control.
"""

from deepagents.memory.in_memory import InMemoryMemoryStore
from deepagents.memory.langmem import build_langmem_namespace, create_langmem_memory_tools
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
    "InMemoryMemoryStore",
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
]
