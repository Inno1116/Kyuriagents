"""Reranker implementations for hybrid retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.rag._text import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.rag.types import RetrievedChunk

_DEFAULT_FUSED_WEIGHT = 0.25


class FusedScoreReranker:
    """Reranker that trusts the hybrid fusion score."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Sort candidates by fused score.

        Args:
            query: Original user query. Accepted for protocol compatibility.
            candidates: Candidate chunks after hybrid fusion.
            limit: Maximum number of chunks to return.

        Returns:
            Final ranked chunks with `rerank_score` populated.
        """
        del query
        ranked = [candidate.with_scores(rerank_score=candidate.fused_score) for candidate in candidates]
        return sorted(ranked, key=lambda item: (-_score_or_zero(item.rerank_score), item.chunk_id))[:limit]


class LexicalReranker:
    """Dependency-free reranker for local validation and fallback deployments."""

    def __init__(self, *, fused_weight: float = _DEFAULT_FUSED_WEIGHT) -> None:
        """Initialize the reranker.

        Args:
            fused_weight: Weight added from the upstream fusion score.
        """
        self._fused_weight = fused_weight

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Rerank by query-token overlap plus a small fused-score prior.

        Args:
            query: Original user query.
            candidates: Candidate chunks after hybrid fusion.
            limit: Maximum number of chunks to return.

        Returns:
            Final ranked chunks with `rerank_score` populated.
        """
        query_terms = set(tokenize(query))
        ranked: list[RetrievedChunk] = []
        for candidate in candidates:
            candidate_terms = set(tokenize(f"{candidate.text} {candidate.metadata.title} {candidate.metadata.section_path}"))
            overlap = len(query_terms & candidate_terms)
            score = float(overlap) + (candidate.fused_score * self._fused_weight)
            ranked.append(candidate.with_scores(rerank_score=score))
        return sorted(ranked, key=lambda item: (-_score_or_zero(item.rerank_score), item.chunk_id))[:limit]


def _score_or_zero(score: float | None) -> float:
    return 0.0 if score is None else score
