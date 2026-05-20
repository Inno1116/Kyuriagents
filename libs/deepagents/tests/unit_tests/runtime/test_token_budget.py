from __future__ import annotations

import pytest

from deepagents.runtime import AgentRuntimeConfig
from deepagents.runtime.token_budget import TokenBudgetExceeded, enforce_user_input_budget


class WordCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


def test_enforce_user_input_budget_accepts_messages_within_limit() -> None:
    config = AgentRuntimeConfig(max_user_input_tokens=3)

    enforce_user_input_budget(config, "one two three", WordCounter())


def test_enforce_user_input_budget_rejects_messages_over_limit() -> None:
    config = AgentRuntimeConfig(max_user_input_tokens=3)

    with pytest.raises(TokenBudgetExceeded, match="limit 3"):
        enforce_user_input_budget(config, "one two three four", WordCounter())
