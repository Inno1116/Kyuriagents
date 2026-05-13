from __future__ import annotations

import pytest

from deepagents.runtime import AgentRuntimeConfig


def test_runtime_config_from_env_parses_services_and_modes() -> None:
    config = AgentRuntimeConfig.from_env(
        env={
            "DEEPAGENTS_TENANT_ID": "tenant-a",
            "DEEPAGENTS_USER_ID": "user-1",
            "DASHSCOPE_API_KEY": "key",
            "DASHSCOPE_CHAT_MODEL": "qwen-plus",
            "DASHSCOPE_EMBEDDING_MODEL": "text-embedding-v3",
            "DASHSCOPE_EMBEDDING_DIMENSIONS": "1024",
            "DEEPAGENTS_POSTGRES_DSN": "postgresql://app",
            "DEEPAGENTS_RAG_MODE": "hybrid",
            "DEEPAGENTS_MEMORY_MODE": "auto",
            "DEEPAGENTS_ENABLE_MCP": "true",
            "DEEPAGENTS_MCP_CONFIG_PATH": "mcp.json",
            "DEEPAGENTS_TOOL_ALLOWED_RISKS": "read_only,external_read,write",
            "DEEPAGENTS_TOOL_DENIED_NAMES": "execute",
            "RAG_KB_IDS": "kb-main,kb-extra",
        }
    )

    assert config.tenant_id == "tenant-a"
    assert config.user_id == "user-1"
    assert config.dashscope_api_key == "key"
    assert config.embedding_dimensions == 1024
    assert config.postgres_dsn == "postgresql://app"
    assert config.rag_mode == "hybrid"
    assert config.memory_mode == "auto"
    assert config.enable_mcp
    assert config.mcp_config_path == "mcp.json"
    assert config.tool_policy().denied_tools == frozenset({"execute"})
    assert config.retrieval_defaults().kb_ids == ("kb-main", "kb-extra")


def test_runtime_config_rejects_invalid_modes() -> None:
    with pytest.raises(ValueError, match="RetrievalMode"):
        AgentRuntimeConfig.from_env(env={"DEEPAGENTS_RAG_MODE": "sometimes"})


def test_runtime_config_reports_missing_runtime_secrets() -> None:
    config = AgentRuntimeConfig.from_env(env={})

    assert config.missing_for_model() == ("DASHSCOPE_API_KEY",)
    assert config.missing_for_memory() == ("DEEPAGENTS_POSTGRES_DSN",)
