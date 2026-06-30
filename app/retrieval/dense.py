from __future__ import annotations

from app.indexing.embeddings.base import EmbeddingProvider
from app.indexing.vectorstores.base import VectorStore
from app.retrieval.filters import is_record_visible
from app.retrieval.permissions import filters_for_query
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class DenseRetriever:
    source_name = "dense"

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        analyzed_query: AnalyzedQuery,
        *,
        top_k: int,
        oversample: int = 3,
    ) -> list[RetrievalCandidate]:
        vector = self.embedding_provider.embed_query(analyzed_query.query.query)
        results = self.vector_store.search(
            vector,
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

    def release_embedding_provider(self) -> None:
        close = getattr(self.embedding_provider, "close", None)
        if callable(close):
            close()
