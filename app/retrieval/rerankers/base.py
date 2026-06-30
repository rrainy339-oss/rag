from __future__ import annotations

from typing import Protocol

from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class RerankerDependencyError(RuntimeError):
    """Raised when an optional reranker dependency is unavailable."""


class RerankerAPIError(RuntimeError):
    """Raised when a remote reranker API request fails."""


class Reranker(Protocol):
    model_name: str

    def rerank(
        self,
        analyzed_query: AnalyzedQuery,
        candidates: list[RetrievalCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Rerank retrieval candidates."""


def candidate_document_text(candidate: RetrievalCandidate) -> str:
    return candidate.record.contextual_text or candidate.record.text


def apply_rerank_scores(
    candidates: list[RetrievalCandidate],
    scores: list[float],
    *,
    top_k: int,
    model_name: str,
) -> list[RetrievalCandidate]:
    if len(candidates) != len(scores):
        raise ValueError(
            f"Reranker returned {len(scores)} score(s) for {len(candidates)} candidate(s)."
        )

    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = float(score)
        candidate.final_score = float(score)
        candidate.metadata["reranker"] = model_name
        candidate.metadata["pre_rerank_score"] = candidate.fused_score

    ranked = sorted(candidates, key=lambda item: item.final_score, reverse=True)[:top_k]
    for index, candidate in enumerate(ranked, start=1):
        candidate.rank = index
    return ranked
