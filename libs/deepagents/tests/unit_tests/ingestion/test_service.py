from dataclasses import replace

from deepagents.ingestion.parsers import ParsedDocument, ParsedSection, ParseRequest
from deepagents.ingestion.service import InMemoryKnowledgeBaseStore, KnowledgeBaseService
from deepagents.runtime import AgentRuntimeConfig


class FakeParser:
    name = "fake_parser"
    version = "fake_parser:v1"

    def supports(self, request: ParseRequest) -> bool:
        return request.filename.endswith(".pdf")

    def parse(self, request: ParseRequest) -> ParsedDocument:
        return ParsedDocument(
            title="Uploaded PDF",
            sections=(ParsedSection(text="Kyuriagents can parse uploaded PDFs for RAG.", page_start=1, page_end=1),),
            language="en",
        )


class FakeIndexer:
    def __init__(self) -> None:
        self.chunks = []
        self.deleted_kbs = []
        self.deleted_docs = []

    def index(self, chunks):
        self.chunks.extend(chunks)

    def delete_knowledge_base(self, *, tenant_id: str, kb_id: str) -> None:
        self.deleted_kbs.append((tenant_id, kb_id))

    def delete_document(self, *, tenant_id: str, kb_id: str, doc_id: str) -> None:
        self.deleted_docs.append((tenant_id, kb_id, doc_id))


def test_upload_document_creates_job_and_worker_indexes_chunks(tmp_path):
    store = InMemoryKnowledgeBaseStore()
    indexer = FakeIndexer()
    service = KnowledgeBaseService(
        config=AgentRuntimeConfig(
            tenant_id="tenant_1",
            user_id="user_1",
            upload_dir=str(tmp_path),
            ingestion_parser_mode="local",
            embedding_model="text-embedding-v4",
            embedding_dimensions=1024,
        ),
        store=store,
        parser=FakeParser(),
        indexer=indexer,
    )
    kb = service.create_knowledge_base(tenant_id="tenant_1", user_id="user_1", name="PDF KB")

    document, job = service.upload_document(
        tenant_id="tenant_1",
        user_id="user_1",
        kb_id=kb.kb_id,
        filename="sample.pdf",
        mime_type="application/pdf",
        content=b"%PDF fake text pdf",
    )

    assert document.status == "processing"
    assert job.status == "queued"

    processed = service.process_next_job()

    assert processed is not None
    assert len(indexer.chunks) == 1
    documents = service.list_documents(tenant_id="tenant_1", user_id="user_1", kb_id=kb.kb_id)
    assert documents[0].status == "active"
    assert documents[0].title == "Uploaded PDF"


def test_delete_document_hides_it_and_removes_indexed_chunks(tmp_path):
    store = InMemoryKnowledgeBaseStore()
    indexer = FakeIndexer()
    service = KnowledgeBaseService(
        config=AgentRuntimeConfig(
            tenant_id="tenant_1",
            user_id="user_1",
            upload_dir=str(tmp_path),
            ingestion_parser_mode="local",
            embedding_model="text-embedding-v4",
            embedding_dimensions=1024,
        ),
        store=store,
        parser=FakeParser(),
        indexer=indexer,
    )
    kb = service.create_knowledge_base(tenant_id="tenant_1", user_id="user_1", name="PDF KB")
    document, _job = service.upload_document(
        tenant_id="tenant_1",
        user_id="user_1",
        kb_id=kb.kb_id,
        filename="sample.pdf",
        mime_type="application/pdf",
        content=b"%PDF fake text pdf",
    )

    deleted = service.delete_document(tenant_id="tenant_1", user_id="user_1", kb_id=kb.kb_id, doc_id=document.doc_id)

    assert deleted.status == "deleted"
    assert service.list_documents(tenant_id="tenant_1", user_id="user_1", kb_id=kb.kb_id) == []
    assert indexer.deleted_docs == [("tenant_1", kb.kb_id, document.doc_id)]


def test_fail_stale_jobs_marks_running_documents_failed(tmp_path):
    store = InMemoryKnowledgeBaseStore()
    service = KnowledgeBaseService(
        config=AgentRuntimeConfig(
            tenant_id="tenant_1",
            user_id="user_1",
            upload_dir=str(tmp_path),
            ingestion_parser_mode="local",
            ingestion_job_timeout_seconds=1,
            embedding_model="text-embedding-v4",
            embedding_dimensions=1024,
        ),
        store=store,
        parser=FakeParser(),
        indexer=FakeIndexer(),
    )
    kb = service.create_knowledge_base(tenant_id="tenant_1", user_id="user_1", name="PDF KB")
    _document, job = service.upload_document(
        tenant_id="tenant_1",
        user_id="user_1",
        kb_id=kb.kb_id,
        filename="sample.pdf",
        mime_type="application/pdf",
        content=b"%PDF fake text pdf",
    )
    claimed = store.claim_next_job()
    assert claimed is not None
    store._jobs[job.job_id] = replace(claimed, started_at="2000-01-01T00:00:00+00:00")

    failed = service.fail_stale_jobs(max_age_seconds=1)

    assert failed == 1
    updated = store.get_job(tenant_id="tenant_1", user_id="user_1", job_id=job.job_id)
    assert updated is not None
    assert updated.status == "failed"
    documents = service.list_documents(tenant_id="tenant_1", user_id="user_1", kb_id=kb.kb_id)
    assert documents[0].status == "failed"
