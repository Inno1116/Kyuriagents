"""Document ingestion utilities for user knowledge bases."""

from deepagents.ingestion.parsers import (
    AutoDocumentParser,
    DocumentParser,
    LocalDocumentParser,
    LocalDocxTextParser,
    LocalPdfTextParser,
    LocalPlainTextParser,
    MCPDocumentParser,
    ParsedDocument,
    ParsedSection,
    ParseRequest,
    build_document_parser,
)
from deepagents.ingestion.service import KnowledgeBaseService
from deepagents.ingestion.store import (
    DocumentRecord,
    IngestionJobRecord,
    KnowledgeBaseRecord,
)

__all__ = [
    "AutoDocumentParser",
    "DocumentParser",
    "DocumentRecord",
    "IngestionJobRecord",
    "KnowledgeBaseRecord",
    "KnowledgeBaseService",
    "LocalDocumentParser",
    "LocalDocxTextParser",
    "LocalPdfTextParser",
    "LocalPlainTextParser",
    "MCPDocumentParser",
    "ParseRequest",
    "ParsedDocument",
    "ParsedSection",
    "build_document_parser",
]
