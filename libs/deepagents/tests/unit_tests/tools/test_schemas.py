from __future__ import annotations

from importlib import resources


def test_tools_postgres_schema_contains_expected_tables() -> None:
    schema = resources.files("deepagents.tools").joinpath("schemas/postgres_schema.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS agent_tool_calls" in schema
    assert "CREATE TABLE IF NOT EXISTS agent_tool_policies" in schema
    assert "CREATE TABLE IF NOT EXISTS agent_mcp_servers" in schema
    assert "secret_refs JSONB" in schema
