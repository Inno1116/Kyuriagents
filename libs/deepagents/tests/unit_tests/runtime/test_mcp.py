from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from deepagents.runtime.mcp import build_mcp_tool_descriptors, load_mcp_config


def _create_ticket(title: str) -> str:
    """Create a ticket."""
    return title


def test_load_mcp_config_expands_env_and_keeps_governance_out_of_connection(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "internal": {
                        "transport": "streamable_http",
                        "url": "http://localhost:8000/mcp",
                        "headers": {"Authorization": "Bearer ${TOKEN}"},
                        "risk": "external_read",
                        "tool_risks": {"create_ticket": "write"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_mcp_config(path, env={"TOKEN": "secret"})

    server = config.servers["internal"]
    assert server.connection["headers"] == {"Authorization": "Bearer secret"}
    assert "risk" not in server.connection
    assert server.tool_risks["create_ticket"] == "write"


def test_build_mcp_tool_descriptors_uses_server_and_tool_overrides(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "internal": {
                        "transport": "stdio",
                        "command": "run",
                        "risk": "external_read",
                        "requires_confirmation": False,
                        "tool_risks": {"internal_create_ticket": "write"},
                        "tool_confirmation": {"internal_create_ticket": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = load_mcp_config(
        path,
    )
    tool = StructuredTool.from_function(name="internal_create_ticket", func=_create_ticket)

    descriptors = build_mcp_tool_descriptors([tool], config)

    assert descriptors[0].name == "internal_create_ticket"
    assert descriptors[0].source == "mcp"
    assert descriptors[0].risk == "write"
    assert descriptors[0].requires_confirmation
    assert descriptors[0].metadata["mcp_server"] == "internal"
