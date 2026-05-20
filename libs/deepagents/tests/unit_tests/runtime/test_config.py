from __future__ import annotations

import pytest

from deepagents.runtime import AgentRuntimeConfig


def test_runtime_config_from_env_parses_services_and_modes() -> None:
    config = AgentRuntimeConfig.from_env(
        env={
            "DEEPAGENTS_TENANT_ID": "tenant-a",
            "DEEPAGENTS_USER_ID": "user-1",
            "DEEPAGENTS_THREAD_ID": "thread-1",
            "DASHSCOPE_API_KEY": "key",
            "DASHSCOPE_CHAT_MODEL": "qwen-plus",
            "DASHSCOPE_EMBEDDING_MODEL": "text-embedding-v3",
            "DASHSCOPE_EMBEDDING_DIMENSIONS": "1024",
            "DEEPAGENTS_POSTGRES_DSN": "postgresql://app",
            "DEEPAGENTS_RAG_MODE": "hybrid",
            "DEEPAGENTS_MEMORY_MODE": "auto",
            "DEEPAGENTS_CONTEXT_SUMMARY_MODEL": "qwen-turbo",
            "DEEPAGENTS_ENABLE_MCP": "true",
            "DEEPAGENTS_MCP_CONFIG_PATH": "mcp.json",
            "DEEPAGENTS_TOOL_ALLOWED_RISKS": "read_only,external_read,write",
            "DEEPAGENTS_TOOL_DENIED_NAMES": "execute",
            "RAG_KB_IDS": "kb-main,kb-extra",
            "MEMORY_ES_INDEX": "memory_es",
            "MEMORY_MILVUS_COLLECTION": "memory_vectors",
            "DEEPAGENTS_MEMORY_CHECKPOINT_INTERVAL": "12",
            "DEEPAGENTS_MEMORY_CHECKPOINT_MAX_CHARS": "2048",
            "DEEPAGENTS_ENABLE_CONTEXT_SUMMARIZATION": "true",
            "DEEPAGENTS_CONTEXT_SUMMARY_TRIGGER_TOKENS": "90000",
            "DEEPAGENTS_CONTEXT_SUMMARY_TRIGGER_MESSAGES": "50",
            "DEEPAGENTS_CONTEXT_SUMMARY_KEEP_MESSAGES": "16",
            "DEEPAGENTS_REDIS_URL": "redis://example.test:6379/2",
            "KYURI_MAX_USER_INPUT_TOKENS": "12800",
            "QWEN_TOKENIZER_MODEL": "Qwen/Qwen3-8B",
            "DEEPAGENTS_API_ADMIN_KEY": "admin-secret",
            "DEEPAGENTS_AUTH_TOKEN_TTL_DAYS": "14",
            "DEEPAGENTS_API_CORS_ORIGINS": "http://127.0.0.1:5173,http://localhost:5173",
            "DEEPAGENTS_ENABLE_INGESTION_REDIS_QUEUE": "true",
            "DEEPAGENTS_INGESTION_REDIS_QUEUE_NAME": "kyuri:test:ingestion",
            "DEEPAGENTS_INGESTION_REDIS_BLOCK_TIMEOUT_SECONDS": "5",
        }
    )

    assert config.tenant_id == "tenant-a"
    assert config.user_id == "user-1"
    assert config.thread_id == "thread-1"
    assert config.dashscope_api_key == "key"
    assert config.embedding_dimensions == 1024
    assert config.postgres_dsn == "postgresql://app"
    assert config.rag_mode == "hybrid"
    assert config.memory_mode == "auto"
    assert config.context_summary_model == "qwen-turbo"
    assert config.enable_mcp
    assert config.mcp_config_path == "mcp.json"
    assert config.tool_policy().denied_tools == frozenset({"execute"})
    assert config.retrieval_defaults().thread_id == "thread-1"
    assert config.retrieval_defaults().kb_ids == ("kb-main", "kb-extra")
    assert config.memory_es_index == "memory_es"
    assert config.memory_milvus_collection == "memory_vectors"
    assert config.memory_checkpoint_interval == 12
    assert config.memory_checkpoint_max_chars == 2048
    assert config.enable_context_summarization
    assert config.context_summary_trigger() == ("tokens", 90000)
    assert config.context_summary_keep() == ("messages", 16)
    assert config.redis_url == "redis://example.test:6379/2"
    assert config.max_user_input_tokens == 12800
    assert config.tokenizer_model == "Qwen/Qwen3-8B"
    assert config.api_admin_key == "admin-secret"
    assert config.auth_token_ttl_days == 14
    assert config.api_cors_origins == ("http://127.0.0.1:5173", "http://localhost:5173")
    assert config.enable_ingestion_redis_queue
    assert config.ingestion_redis_queue_name == "kyuri:test:ingestion"
    assert config.ingestion_redis_block_timeout_seconds == 5


def test_runtime_config_rejects_invalid_modes() -> None:
    with pytest.raises(ValueError, match="RetrievalMode"):
        AgentRuntimeConfig.from_env(env={"DEEPAGENTS_RAG_MODE": "sometimes"})


def test_runtime_config_rejects_invalid_context_summary_window() -> None:
    with pytest.raises(ValueError, match="greater than"):
        AgentRuntimeConfig(context_summary_trigger_tokens=0, context_summary_trigger_messages=8, context_summary_keep_messages=8)


def test_runtime_config_reports_missing_runtime_secrets() -> None:
    config = AgentRuntimeConfig.from_env(env={})

    assert config.missing_for_model() == ("DASHSCOPE_API_KEY",)
    assert config.missing_for_memory() == ("DEEPAGENTS_POSTGRES_DSN",)
