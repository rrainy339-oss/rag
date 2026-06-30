from __future__ import annotations

from app.indexing.embeddings.base import HybridEmbedding
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore
from app.retrieval.config import RetrievalConfig
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.filters import is_record_visible
from app.retrieval.permissions import filters_for_query
from app.retrieval.query import QueryAnalyzer
from app.retrieval.rerankers.base import Reranker
from app.retrieval.rerankers.noop import NoopReranker
from app.retrieval.schemas import (
    AnalyzedQuery,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResponse,
)


class QdrantHybridRetriever:
    source_name = "qdrant_hybrid"

    def __init__(
        self,
        *,
        embedding_provider: object,
        vector_store: QdrantHybridVectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        analyzed_query: AnalyzedQuery,
        *,
        top_k: int,
        dense_top_k: int,
        sparse_top_k: int,
    ) -> list[RetrievalCandidate]:
        embed_query_hybrid = getattr(self.embedding_provider, "embed_query_hybrid", None)
        if not callable(embed_query_hybrid):
            raise ValueError(
                "qdrant_hybrid retrieval requires an embedding provider with "
                "embed_query_hybrid(). Use --embedding-provider bge-m3."
            )

        query_embedding: HybridEmbedding = embed_query_hybrid(analyzed_query.query.query)
        results = self.vector_store.hybrid_search(
            dense_vector=query_embedding.dense,
            sparse_indices=query_embedding.sparse.indices,
            sparse_values=query_embedding.sparse.values,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
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
                    metadata={"fusion_boost": 1.0},
                )
            )
        for index, candidate in enumerate(candidates, start=1):
            candidate.rank = index
        return candidates

    def release_embedding_provider(self) -> None:
        close = getattr(self.embedding_provider, "close", None)
        if callable(close):
            close()


class QdrantHybridRetrievalPipeline:
    def __init__(
        self,
        *,
        hybrid_retriever: QdrantHybridRetriever,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        context_builder: ContextBuilder | None = None,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or NoopReranker()
        self.context_builder = context_builder or ContextBuilder()
        self.query_analyzer = query_analyzer or QueryAnalyzer()

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        analyzed_query = self.query_analyzer.analyze(query)
        fused_candidates = self.hybrid_retriever.retrieve(
            analyzed_query,
            top_k=self.config.fusion_top_k,
            dense_top_k=self.config.dense_top_k * self.config.retriever_oversample,
            sparse_top_k=self.config.sparse_top_k * self.config.retriever_oversample,
        )
        rerank_input = fused_candidates[: self.config.rerank_top_k]
        if self.config.release_dense_model_before_rerank:
            self.hybrid_retriever.release_embedding_provider()
        reranked_candidates = self.reranker.rerank(
            analyzed_query,
            rerank_input,
            top_k=self.config.rerank_top_k,
        )
        contexts = self.context_builder.build(
            reranked_candidates,
            final_top_k=self.config.final_top_k,
            query=analyzed_query.query,
        )

        return RetrievalResponse(
            query=query.query,
            query_type=analyzed_query.query_type,
            contexts=contexts,
            candidates=reranked_candidates,
            stats={
                "dense_count": self.config.dense_top_k,
                "sparse_count": self.config.sparse_top_k,
                "fused_count": len(fused_candidates),
                "reranked_count": len(reranked_candidates),
                "context_count": len(contexts),
                "retriever": self.hybrid_retriever.source_name,
                "reranker": self.reranker.model_name,
            },
        )
