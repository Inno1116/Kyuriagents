from __future__ import annotations

import sys
from types import ModuleType

from deepagents.memory import MemoryScope, build_langmem_namespace, create_langmem_memory_tools


def test_build_langmem_namespace_is_tenant_and_user_scoped() -> None:
    namespace = build_langmem_namespace(
        MemoryScope(tenant_id="tenant-a", user_id="user-1"),
        collection="memories",
    )

    assert namespace == ("tenant-a", "users", "user-1", "memories")


def test_create_langmem_memory_tools_uses_lazy_import(monkeypatch) -> None:
    module = ModuleType("langmem")

    def create_manage_memory_tool(*, namespace, instructions=None, store=None):
        return {
            "name": "manage",
            "namespace": namespace,
            "instructions": instructions,
            "store": store,
        }

    def create_search_memory_tool(*, namespace, store=None):
        return {
            "name": "search",
            "namespace": namespace,
            "store": store,
        }

    module.create_manage_memory_tool = create_manage_memory_tool
    module.create_search_memory_tool = create_search_memory_tool
    monkeypatch.setitem(sys.modules, "langmem", module)

    store = object()
    tools = create_langmem_memory_tools(
        scope=MemoryScope(tenant_id="tenant-a", user_id="user-1"),
        instructions="Remember durable preferences.",
        store=store,
    )

    assert tools[0]["name"] == "manage"
    assert tools[0]["namespace"] == ("tenant-a", "users", "user-1", "memories")
    assert tools[0]["instructions"] == "Remember durable preferences."
    assert tools[0]["store"] is store
    assert tools[1]["name"] == "search"
    assert tools[1]["namespace"] == ("tenant-a", "users", "user-1", "memories")
    assert tools[1]["store"] is store
