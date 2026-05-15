from __future__ import annotations

import pytest

from deepagents.server.identity import DuplicateUserError, InMemoryUserCenter, hash_api_key, hash_password, verify_password


def test_in_memory_user_center_creates_user_api_key_and_authenticates() -> None:
    center = InMemoryUserCenter()
    tenant = center.create_tenant(name="Acme", tenant_id="tenant-a")
    user = center.create_user(tenant_id=tenant.tenant_id, email="user@example.com", display_name="User", user_id="user-1")

    created = center.create_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id, name="local")
    context = center.authenticate_api_key(created.raw_key)

    assert created.raw_key.startswith("kya_")
    assert created.record.key_hash == hash_api_key(created.raw_key)
    assert created.raw_key != created.record.key_hash
    assert context is not None
    assert context.tenant.tenant_id == "tenant-a"
    assert context.user.user_id == "user-1"
    assert context.api_key.last_used_at is not None

    assert center.revoke_api_key(tenant_id=tenant.tenant_id, user_id=user.user_id, key_id=created.record.key_id)
    assert center.authenticate_api_key(created.raw_key) is None


def test_in_memory_user_center_stores_threads_and_messages_by_user() -> None:
    center = InMemoryUserCenter()
    center.create_tenant(name="Acme", tenant_id="tenant-a")
    center.create_user(tenant_id="tenant-a", email="user@example.com", user_id="user-1")
    center.create_user(tenant_id="tenant-a", email="other@example.com", user_id="user-2")

    thread = center.create_thread(tenant_id="tenant-a", user_id="user-1", title="Hello", thread_id="thread-1")
    message = center.append_message(
        tenant_id="tenant-a",
        thread_id=thread.thread_id,
        user_id="user-1",
        role="user",
        content="Hello agent",
        message_id="msg-1",
    )

    assert center.get_thread(tenant_id="tenant-a", user_id="user-1", thread_id="thread-1") == thread
    assert center.get_thread(tenant_id="tenant-a", user_id="user-2", thread_id="thread-1") is None
    assert center.list_threads(tenant_id="tenant-a", user_id="user-1") == [thread]
    assert center.list_messages(tenant_id="tenant-a", thread_id="thread-1") == [message]


def test_password_registration_hashes_and_authenticates() -> None:
    center = InMemoryUserCenter()
    sample = "password123"
    wrong = "wrong-password"
    center.ensure_tenant(name="Acme", tenant_id="tenant-a")

    user = center.create_user_with_password(tenant_id="tenant-a", email="User@Example.com", password=sample, user_id="user-1")

    assert user.email == "user@example.com"
    assert center.authenticate_password(tenant_id="tenant-a", email="user@example.com", password=sample) == user
    assert center.authenticate_password(tenant_id="tenant-a", email="user@example.com", password=wrong) is None
    with pytest.raises(DuplicateUserError):
        center.create_user_with_password(tenant_id="tenant-a", email="user@example.com", password=sample)


def test_password_hash_does_not_store_raw_password() -> None:
    sample = "password123"
    wrong = "wrong-password"
    password_hash = hash_password(sample)

    assert sample not in password_hash
    assert verify_password(sample, password_hash)
    assert not verify_password(wrong, password_hash)
