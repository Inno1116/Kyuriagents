# Deep Agents Runtime

This package wires the SDK primitives into a runnable deployment shape:

- DashScope chat and embedding clients through the OpenAI-compatible API.
- PostgreSQL schema/bootstrap helpers.
- `PostgresMemoryStore` for durable long-term memory.
- `RetrievalMiddleware` with RAG and memory enabled from one config object.
- `ToolGovernanceMiddleware` with policy checks and optional PostgreSQL audit logs.
- Optional MCP tool loading via `langchain-mcp-adapters`.

Minimal usage:

```python
from deepagents.runtime import AgentRuntimeConfig, create_kyuri_agent

config = AgentRuntimeConfig.from_env(
    tenant_id="default",
    user_id="local-user",
)
agent = create_kyuri_agent(config)
```

Before first startup, create the database and apply schemas:

```python
from deepagents.runtime import (
    AgentRuntimeConfig,
    apply_deepagents_postgres_schemas,
    create_postgres_database,
)

config = AgentRuntimeConfig.from_env()
if config.postgres_admin_dsn:
    create_postgres_database(
        admin_dsn=config.postgres_admin_dsn,
        database=config.postgres_database,
        owner="deepagents",
    )
if config.postgres_dsn:
    apply_deepagents_postgres_schemas(dsn=config.postgres_dsn)
```

API keys should come from environment variables, never from checked-in files.

MCP usage is optional. Set `DEEPAGENTS_ENABLE_MCP=true` and point
`DEEPAGENTS_MCP_CONFIG_PATH` at a JSON file shaped like `mcp.json.example`.
Secrets in that file should be referenced as `${ENV_VAR}` placeholders.
