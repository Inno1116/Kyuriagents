from __future__ import annotations

from deepagents.memory import (
    ElasticsearchMilvusMemoryIndexer,
    InMemoryMemoryStore,
    MemoryHybridSearcher,
    MemoryRecord,
    MemoryScope,
    memory_records_to_chunks,
)
from deepagents.rag import HybridRAGRetriever, HybridSearchConfig, InMemoryKeywordStore, InMemoryVectorStore


class _FakeElasticsearch:
    def __init__(self) -> None:
        self.bulk_calls: list[dict[str, object]] = []

    def bulk(self, *, operations, refresh):
        self.bulk_calls.append({"operations": operations, "refresh": refresh})
        return {"errors": False}


class _FakeMilvus:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []

    def upsert(self, *, collection_name, data):
        self.upsert_calls.append({"collection_name": collection_name, "data": data})

    def delete(self, **kwargs: object):
        self.delete_calls.append(kwargs)


def _embed(text: str) -> tuple[float, ...]:
    lowered = text.lower()
    return (
        1.0 if "milvus" in lowered else 0.0,
        1.0 if "dashscope" in lowered else 0.0,
    )


def _memory(memory_id: str = "mem-1") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        tenant_id="tenant-a",
        user_id="user-1",
        scope_type="user",
        scope_id="user-1",
        memory_type="fact",
        content="Kyuriagents uses DashScope and Milvus for RAG.",
        tags=("rag",),
    )


def test_memory_records_to_chunks_preserves_memory_scope_metadata() -> None:
    chunk = memory_records_to_chunks([_memory()], embed_text=_embed)[0]

    assert chunk.metadata.source_type == "memory"
    assert chunk.metadata.kb_id == "memory:user:user-1"
    assert chunk.metadata.doc_id == "mem-1"
    assert chunk.embedding == (1.0, 1.0)


def test_memory_hybrid_searcher_loads_source_of_truth_records() -> None:
    store = InMemoryMemoryStore([_memory()])
    chunks = memory_records_to_chunks([_memory()], embed_text=_embed)
    retriever = HybridRAGRetriever(
        vector_searcher=InMemoryVectorStore(chunks, embed_query=_embed),
        keyword_searcher=InMemoryKeywordStore(chunks),
        config=HybridSearchConfig(top_k=3),
    )
    searcher = MemoryHybridSearcher(retriever=retriever, store=store)

    results = searcher.search("Which Milvus database does Kyuriagents use?", scope=MemoryScope(tenant_id="tenant-a", user_id="user-1"), limit=3)

    assert results[0].memory_id == "mem-1"
    assert results[0].semantic_score is not None


def test_elasticsearch_milvus_memory_indexer_writes_and_deletes_memory_chunks() -> None:
    es = _FakeElasticsearch()
    milvus = _FakeMilvus()
    indexer = ElasticsearchMilvusMemoryIndexer(
        es_index="memory_chunks",
        milvus_collection="memory_chunks",
        embed_text=_embed,
        es_client=es,
        milvus_client=milvus,
        refresh=True,
    )

    indexer.upsert([_memory()])

    assert es.bulk_calls[0]["refresh"] is True
    assert es.bulk_calls[0]["operations"][0] == {"index": {"_index": "memory_chunks", "_id": "memory:mem-1"}}
    assert es.bulk_calls[0]["operations"][1]["source_type"] == "memory"
    assert es.bulk_calls[0]["operations"][1]["chunk_text"] == "Kyuriagents uses DashScope and Milvus for RAG."
    assert milvus.upsert_calls[0]["collection_name"] == "memory_chunks"
    assert milvus.upsert_calls[0]["data"][0]["chunk_id"] == "memory:mem-1"
    assert milvus.upsert_calls[0]["data"][0]["embedding"] == [1.0, 1.0]

    indexer.delete(["mem-1"])

    assert es.bulk_calls[1]["operations"][0] == {"delete": {"_index": "memory_chunks", "_id": "memory:mem-1"}}
    assert milvus.delete_calls[0] == {
        "collection_name": "memory_chunks",
        "filter": 'chunk_id in ["memory:mem-1"]',
    }
