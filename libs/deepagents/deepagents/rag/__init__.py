"""Experimental RAG primitives for Deep Agents.

This package contains the online retrieval path for a hybrid RAG system:
query rewriting, keyword/vector retrieval, rank fusion, reranking, and Top-K
selection. Offline document parsing and embedding jobs should write chunks into
Milvus, Elasticsearch, and the PostgreSQL metadata tables described in
`deepagents/rag/schemas`.
"""

from deepagents.rag.elasticsearch import ElasticsearchKeywordStore
from deepagents.rag.hybrid import HybridRAGRetriever, HybridSearchConfig
from deepagents.rag.in_memory import InMemoryKeywordStore, InMemoryVectorStore
from deepagents.rag.metadata import ChunkMetadata, RetrievalScope
from deepagents.rag.milvus import MilvusVectorStore
from deepagents.rag.query import IdentityQueryRewriter, QueryRewrite, QueryRewriter
from deepagents.rag.rerank import FusedScoreReranker, LexicalReranker
from deepagents.rag.types import (
    DocumentChunk,
    KeywordSearcher,
    Reranker,
    RetrievedChunk,
    VectorSearcher,
)

__all__ = [
    "ChunkMetadata",
    "DocumentChunk",
    "ElasticsearchKeywordStore",
    "FusedScoreReranker",
    "HybridRAGRetriever",
    "HybridSearchConfig",
    "IdentityQueryRewriter",
    "InMemoryKeywordStore",
    "InMemoryVectorStore",
    "KeywordSearcher",
    "LexicalReranker",
    "MilvusVectorStore",
    "QueryRewrite",
    "QueryRewriter",
    "Reranker",
    "RetrievalScope",
    "RetrievedChunk",
    "VectorSearcher",
]
