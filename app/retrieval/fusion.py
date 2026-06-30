from __future__ import annotations

from app.chunking.schemas import ChunkType
from app.retrieval.schemas import AnalyzedQuery, QueryType, RetrievalCandidate


def reciprocal_rank_fusion(
    candidate_lists: dict[str, list[RetrievalCandidate]],
    *,
    analyzed_query: AnalyzedQuery,
    rrf_k: int = 60,
    limit: int = 60,
    table_query_boost: float = 1.15,
    exact_match_boost: float = 1.2,
) -> list[RetrievalCandidate]:
    merged: dict[str, RetrievalCandidate] = {}

    for source_name, candidates in candidate_lists.items():
        for rank, candidate in enumerate(candidates, start=1):
            record_id = candidate.record.record_id
            contribution = 1.0 / (rrf_k + rank)
            if record_id not in merged:
                merged[record_id] = candidate.model_copy(deep=True)
                merged[record_id].source_scores = {}
                merged[record_id].fused_score = 0.0
            merged_candidate = merged[record_id]
            merged_candidate.source_scores[source_name] = candidate.source_scores.get(
                source_name, candidate.final_score
            )
            merged_candidate.fused_score += contribution

    fused = list(merged.values())
    for candidate in fused:
        boost = _candidate_boost(candidate, analyzed_query, table_query_boost, exact_match_boost)
        candidate.fused_score *= boost
        candidate.final_score = candidate.fused_score
        candidate.metadata["fusion_boost"] = boost

    fused.sort(key=lambda item: item.fused_score, reverse=True)
    for index, candidate in enumerate(fused, start=1):
        candidate.rank = index
    return fused[:limit]


def _candidate_boost(
    candidate: RetrievalCandidate,
    analyzed_query: AnalyzedQuery,
    table_query_boost: float,
    exact_match_boost: float,
) -> float:
    boost = 1.0
    record_text = candidate.record.contextual_text.lower()

    if (
        analyzed_query.query_type == QueryType.TABLE
        and candidate.record.chunk_type == ChunkType.TABLE
    ):
        boost *= table_query_boost

    for term in analyzed_query.exact_terms:
        if term.lower() in record_text:
            boost *= exact_match_boost
            break

    return boost

