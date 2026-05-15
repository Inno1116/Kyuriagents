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
