-- User center and API service tables for deployed Deep Agents runtimes.
-- RAG vectors live in Milvus; memory and tool tables live in their own schemas.

CREATE TABLE IF NOT EXISTS agent_tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_users (
    user_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    password_hash TEXT,
    password_updated_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

ALTER TABLE IF EXISTS agent_users
    ADD COLUMN IF NOT EXISTS password_hash TEXT;

ALTER TABLE IF EXISTS agent_users
    ADD COLUMN IF NOT EXISTS password_updated_at TIMESTAMPTZ;

ALTER TABLE IF EXISTS agent_users
    ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS agent_api_keys (
    key_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT '',
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_api_keys_tenant_user_idx ON agent_api_keys(tenant_id, user_id);
CREATE INDEX IF NOT EXISTS agent_api_keys_prefix_idx ON agent_api_keys(key_prefix);

CREATE TABLE IF NOT EXISTS agent_threads (
    thread_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_threads_tenant_user_updated_idx ON agent_threads(tenant_id, user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id TEXT PRIMARY KEY,
    message_seq BIGSERIAL UNIQUE,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL REFERENCES agent_threads(thread_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES agent_users(user_id) ON DELETE SET NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS agent_messages
    ADD COLUMN IF NOT EXISTS message_seq BIGSERIAL;

CREATE UNIQUE INDEX IF NOT EXISTS agent_messages_message_seq_uidx ON agent_messages(message_seq);
CREATE INDEX IF NOT EXISTS agent_messages_thread_seq_idx ON agent_messages(tenant_id, thread_id, message_seq ASC);
