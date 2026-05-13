-- Deep Agents RAG metadata schema for MySQL 8.0+.
-- Vector data lives in Milvus and searchable text lives in Elasticsearch.
-- MySQL is the source of truth for users, tenants, knowledge bases, document
-- lifecycle, chunk manifests, access rules, and offline ingestion jobs.

CREATE TABLE IF NOT EXISTS rag_tenants (
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    status ENUM('active', 'disabled') NOT NULL DEFAULT 'active',
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (tenant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_users (
    user_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    external_user_id VARCHAR(255) NULL,
    display_name VARCHAR(255) NULL,
    email VARCHAR(320) NULL,
    status ENUM('active', 'disabled') NOT NULL DEFAULT 'active',
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (user_id),
    UNIQUE KEY uq_rag_users_tenant_external (tenant_id, external_user_id),
    KEY idx_rag_users_tenant_status (tenant_id, status),
    CONSTRAINT fk_rag_users_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_knowledge_bases (
    kb_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    owner_user_id VARCHAR(64) NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT NULL,
    visibility ENUM('private', 'team', 'public') NOT NULL DEFAULT 'private',
    status ENUM('active', 'archived', 'disabled') NOT NULL DEFAULT 'active',
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (kb_id),
    KEY idx_rag_kb_tenant_visibility (tenant_id, visibility, status),
    KEY idx_rag_kb_owner (owner_user_id),
    CONSTRAINT fk_rag_kb_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id),
    CONSTRAINT fk_rag_kb_owner
        FOREIGN KEY (owner_user_id) REFERENCES rag_users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    kb_id VARCHAR(64) NOT NULL,
    owner_user_id VARCHAR(64) NULL,
    source_type VARCHAR(32) NOT NULL,
    source_uri VARCHAR(2048) NOT NULL,
    title VARCHAR(512) NOT NULL DEFAULT '',
    language VARCHAR(16) NOT NULL DEFAULT 'unknown',
    visibility ENUM('private', 'team', 'public') NOT NULL DEFAULT 'private',
    status ENUM('active', 'processing', 'failed', 'archived', 'deleted') NOT NULL DEFAULT 'processing',
    latest_version VARCHAR(64) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (doc_id),
    KEY idx_rag_documents_tenant_kb (tenant_id, kb_id, status),
    KEY idx_rag_documents_owner (owner_user_id),
    KEY idx_rag_documents_source (tenant_id, source_type, source_uri(255)),
    CONSTRAINT fk_rag_documents_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id),
    CONSTRAINT fk_rag_documents_kb
        FOREIGN KEY (kb_id) REFERENCES rag_knowledge_bases (kb_id),
    CONSTRAINT fk_rag_documents_owner
        FOREIGN KEY (owner_user_id) REFERENCES rag_users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_document_versions (
    doc_version VARCHAR(64) NOT NULL,
    doc_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    parser_version VARCHAR(64) NOT NULL,
    chunker_version VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(128) NOT NULL,
    embedding_version VARCHAR(64) NOT NULL,
    chunk_count INT UNSIGNED NOT NULL DEFAULT 0,
    status ENUM('pending', 'indexed', 'failed', 'superseded') NOT NULL DEFAULT 'pending',
    error_message TEXT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    indexed_at TIMESTAMP(6) NULL,
    PRIMARY KEY (doc_version),
    UNIQUE KEY uq_rag_document_versions_hash (doc_id, content_hash, embedding_model, embedding_version),
    KEY idx_rag_document_versions_doc_status (doc_id, status),
    KEY idx_rag_document_versions_tenant_status (tenant_id, status),
    CONSTRAINT fk_rag_document_versions_doc
        FOREIGN KEY (doc_id) REFERENCES rag_documents (doc_id),
    CONSTRAINT fk_rag_document_versions_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id VARCHAR(128) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    kb_id VARCHAR(64) NOT NULL,
    doc_id VARCHAR(64) NOT NULL,
    doc_version VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NULL,
    chunk_index INT UNSIGNED NOT NULL,
    content_hash CHAR(64) NOT NULL,
    source_type VARCHAR(32) NOT NULL,
    source_uri VARCHAR(2048) NOT NULL,
    title VARCHAR(512) NOT NULL DEFAULT '',
    section_path VARCHAR(1024) NOT NULL DEFAULT '',
    page_start INT UNSIGNED NULL,
    page_end INT UNSIGNED NULL,
    char_start INT UNSIGNED NULL,
    char_end INT UNSIGNED NULL,
    language VARCHAR(16) NOT NULL DEFAULT 'unknown',
    tags JSON NULL,
    visibility ENUM('private', 'team', 'public') NOT NULL DEFAULT 'private',
    embedding_model VARCHAR(128) NOT NULL,
    embedding_version VARCHAR(64) NOT NULL,
    schema_version INT UNSIGNED NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (chunk_id),
    UNIQUE KEY uq_rag_chunks_doc_index (doc_version, chunk_index),
    KEY idx_rag_chunks_scope (tenant_id, kb_id, is_active, visibility),
    KEY idx_rag_chunks_doc (doc_id, doc_version),
    KEY idx_rag_chunks_user (user_id),
    KEY idx_rag_chunks_hash (content_hash),
    CONSTRAINT fk_rag_chunks_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id),
    CONSTRAINT fk_rag_chunks_kb
        FOREIGN KEY (kb_id) REFERENCES rag_knowledge_bases (kb_id),
    CONSTRAINT fk_rag_chunks_doc
        FOREIGN KEY (doc_id) REFERENCES rag_documents (doc_id),
    CONSTRAINT fk_rag_chunks_doc_version
        FOREIGN KEY (doc_version) REFERENCES rag_document_versions (doc_version),
    CONSTRAINT fk_rag_chunks_user
        FOREIGN KEY (user_id) REFERENCES rag_users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_access_rules (
    rule_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    tenant_id VARCHAR(64) NOT NULL,
    kb_id VARCHAR(64) NOT NULL,
    subject_type ENUM('user', 'group', 'tenant') NOT NULL,
    subject_id VARCHAR(128) NOT NULL,
    permission ENUM('read', 'write', 'admin') NOT NULL DEFAULT 'read',
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (rule_id),
    UNIQUE KEY uq_rag_access_rule (tenant_id, kb_id, subject_type, subject_id, permission),
    KEY idx_rag_access_subject (tenant_id, subject_type, subject_id),
    CONSTRAINT fk_rag_access_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id),
    CONSTRAINT fk_rag_access_kb
        FOREIGN KEY (kb_id) REFERENCES rag_knowledge_bases (kb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS rag_ingestion_jobs (
    job_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    kb_id VARCHAR(64) NOT NULL,
    doc_id VARCHAR(64) NULL,
    requested_by_user_id VARCHAR(64) NULL,
    source_uri VARCHAR(2048) NOT NULL,
    status ENUM('queued', 'running', 'succeeded', 'failed', 'cancelled') NOT NULL DEFAULT 'queued',
    error_message TEXT NULL,
    started_at TIMESTAMP(6) NULL,
    finished_at TIMESTAMP(6) NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (job_id),
    KEY idx_rag_ingestion_jobs_scope (tenant_id, kb_id, status),
    KEY idx_rag_ingestion_jobs_user (requested_by_user_id),
    CONSTRAINT fk_rag_ingestion_jobs_tenant
        FOREIGN KEY (tenant_id) REFERENCES rag_tenants (tenant_id),
    CONSTRAINT fk_rag_ingestion_jobs_kb
        FOREIGN KEY (kb_id) REFERENCES rag_knowledge_bases (kb_id),
    CONSTRAINT fk_rag_ingestion_jobs_doc
        FOREIGN KEY (doc_id) REFERENCES rag_documents (doc_id),
    CONSTRAINT fk_rag_ingestion_jobs_user
        FOREIGN KEY (requested_by_user_id) REFERENCES rag_users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
