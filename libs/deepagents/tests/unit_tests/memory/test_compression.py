from __future__ import annotations

from deepagents.memory import (
    MemoryContextBudget,
    MemoryContextCompressor,
    MemoryRecord,
    MemorySearchResult,
    format_memory_context,
)


def _result(memory_id: str, content: str, *, score: float, importance: float = 0.5) -> MemorySearchResult:
    return MemorySearchResult(
        memory=MemoryRecord(
            memory_id=memory_id,
            tenant_id="tenant-a",
            user_id="user-1",
            scope_type="user",
            scope_id="user-1",
            memory_type="fact",
            content=content,
            importance=importance,
        ),
        score=score,
    )


def test_memory_context_compressor_respects_budget_and_ranking() -> None:
    compressor = MemoryContextCompressor(MemoryContextBudget(max_memories=2, max_context_chars=80, max_memory_chars=40))

    compressed = compressor.compress(
        [
            _result("low", "low priority memory", score=0.1),
            _result("high", "high priority memory " * 10, score=0.9),
            _result("middle", "middle memory", score=0.5),
        ]
    )

    assert [result.memory_id for result in compressed.results] == ["high", "middle"]
    assert compressed.omitted_count == 1
    assert compressed.truncated_count == 1
    assert len(compressed.results[0].memory.content) <= 40


def test_format_memory_context_includes_budget_accounting() -> None:
    text = format_memory_context(
        [
            _result("one", "one " * 20, score=1.0),
            _result("two", "two", score=0.9),
        ],
        compressor=MemoryContextCompressor(MemoryContextBudget(max_memories=1, max_context_chars=30, max_memory_chars=30)),
    )

    assert "<agent_long_term_memory>" in text
    assert "[omitted] 1 additional memories" in text
    assert "[truncated] 1 memories" in text
