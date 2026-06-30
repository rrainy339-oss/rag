from __future__ import annotations

from app.indexing.sparse.memory_bm25 import MemoryBM25Store
from app.retrieval.filters import is_record_visible
from app.retrieval.permissions import filters_for_query
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class SparseRetriever:
    source_name = "sparse"

    def __init__(self, *, sparse_store: MemoryBM25Store) -> None:
        self.sparse_store = sparse_store

    def retrieve(
        self,
        analyzed_query: AnalyzedQuery,
        *,
        top_k: int,
        oversample: int = 3,
    ) -> list[RetrievalCandidate]:
        results = self.sparse_store.search(
            analyzed_query.query.query,
            top_k=top_k * oversample,
            filters=filters_for_query(analyzed_query.query),
        )
        candidates: list[RetrievalCandidate] = []
        for result in results:
            if not is_record_visible(result.record, analyzed_query.query):
                continue
            candidates.append(
                RetrievalCandidate(
                    record=result.record,
                    source_scores={self.source_name: result.score},
                    fused_score=result.score,
                    final_score=result.score,
                )
            )
            if len(candidates) >= top_k:
                break
        return candidates
