from __future__ import annotations

from importlib import resources


def test_api_postgres_schema_contains_user_center_tables() -> None:
    text = resources.files("deepagents.server").joinpath("schemas/postgres_schema.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS agent_tenants" in text
    assert "CREATE TABLE IF NOT EXISTS agent_users" in text
    assert "CREATE TABLE IF NOT EXISTS agent_api_keys" in text
    assert "CREATE TABLE IF NOT EXISTS agent_threads" in text
    assert "CREATE TABLE IF NOT EXISTS agent_messages" in text
    assert "password_hash TEXT" in text
    assert "ADD COLUMN IF NOT EXISTS password_hash" in text


def test_api_postgres_schema_allows_evidence_task_steps() -> None:
    """Task step SQL constraints should match task runtime step kinds."""
    text = resources.files("deepagents.server").joinpath("schemas/postgres_schema.sql").read_text(encoding="utf-8")

    assert "kind IN ('think', 'tool', 'rag', 'web', 'process', 'answer')" in text
    assert "DROP CONSTRAINT IF EXISTS agent_task_steps_kind_check" in text
    assert "ADD CONSTRAINT agent_task_steps_kind_check" in text
