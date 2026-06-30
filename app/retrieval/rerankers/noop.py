from __future__ import annotations

from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class NoopReranker:
    model_name = "noop-reranker"

    def rerank(
        self,
        analyzed_query: AnalyzedQuery,
        candidates: list[RetrievalCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        del analyzed_query
        ranked = sorted(candidates, key=lambda item: item.fused_score, reverse=True)[:top_k]
        for index, candidate in enumerate(ranked, start=1):
            candidate.rerank_score = candidate.fused_score
            candidate.final_score = candidate.fused_score
            candidate.rank = index
        return ranked

