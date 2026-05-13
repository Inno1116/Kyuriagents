from __future__ import annotations

import pytest

from deepagents.tools import ToolDescriptor, ToolPolicy, parse_tool_names, parse_tool_risks


def test_tool_policy_allows_read_and_ordinary_write_by_default() -> None:
    policy = ToolPolicy()

    assert policy.evaluate(ToolDescriptor(name="search", risk="read_only")).allowed
    assert policy.evaluate(ToolDescriptor(name="save_memory", risk="write")).allowed


def test_tool_policy_blocks_confirmation_gated_tool() -> None:
    policy = ToolPolicy()

    decision = policy.evaluate(ToolDescriptor(name="write_file", risk="write", requires_confirmation=True))

    assert not decision.allowed
    assert "requires human confirmation" in decision.reason


def test_tool_policy_applies_allow_and_deny_lists() -> None:
    policy = ToolPolicy(allowed_tools=frozenset({"search"}), denied_tools=frozenset({"blocked"}))

    assert policy.evaluate(ToolDescriptor(name="search", risk="read_only")).allowed
    assert not policy.evaluate(ToolDescriptor(name="other", risk="read_only")).allowed
    assert not policy.evaluate(ToolDescriptor(name="blocked", risk="read_only")).allowed


def test_parse_tool_risks_rejects_unknown_risk() -> None:
    with pytest.raises(ValueError, match="ToolRisk"):
        parse_tool_risks("read_only,unknown", default=frozenset())


def test_parse_tool_names_trims_empty_values() -> None:
    assert parse_tool_names(" search, ,save_memory ") == frozenset({"search", "save_memory"})
