from __future__ import annotations

import pytest

from deepagents.runtime import AgentRuntimeConfig, create_dashscope_embed_query, create_dashscope_model


def test_dashscope_model_requires_api_key() -> None:
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        create_dashscope_model(AgentRuntimeConfig(dashscope_api_key=None))


def test_dashscope_embeddings_require_api_key() -> None:
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        create_dashscope_embed_query(AgentRuntimeConfig(dashscope_api_key=None))
