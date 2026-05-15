from __future__ import annotations

from deepagents.memory import (
    InMemoryMemoryStore,
    MemoryCompactionConfig,
    MemoryMaintenanceService,
    MemoryRecord,
    MemoryScope,
    MemoryService,
)


def _memory(memory_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        tenant_id="tenant-a",
        user_id="user-1",
        scope_type="user",
        scope_id="user-1",
        memory_type="preference",
        content=content,
        tags=("style",),
    )


def test_memory_maintenance_compacts_group_and_supersedes_sources() -> None:
    service = MemoryService(
        InMemoryMemoryStore(
            [
                _memory("mem-1", "User prefers Chinese answers."),
                _memory("mem-2", "User wants concise technical conclusions."),
                _memory("mem-3", "User prefers English terms preserved."),
            ]
        )
    )
    maintenance = MemoryMaintenanceService(service, config=MemoryCompactionConfig(min_group_size=3, max_summary_chars=200))

    result = maintenance.compact_scope(MemoryScope(tenant_id="tenant-a", user_id="user-1"))

    assert len(result.created) == 1
    assert set(result.superseded_ids) == {"mem-1", "mem-2", "mem-3"}
    assert result.created[0].status == "active"
    assert "compressed" in result.created[0].tags
    assert service.get("mem-1", scope=MemoryScope(tenant_id="tenant-a", user_id="user-1", active_only=False)).status == "superseded"
