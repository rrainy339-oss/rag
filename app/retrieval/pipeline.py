from __future__ import annotations

from app.retrieval.config import RetrievalConfig
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query import QueryAnalyzer
from app.retrieval.rerankers.base import Reranker
from app.retrieval.rerankers.noop import NoopReranker
from app.retrieval.schemas import RetrievalQuery, RetrievalResponse
from app.retrieval.sparse import SparseRetriever


class RetrievalPipeline:
    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        context_builder: ContextBuilder | None = None,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker or NoopReranker()
        self.context_builder = context_builder or ContextBuilder()
        self.query_analyzer = query_analyzer or QueryAnalyzer()

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        analyzed_query = self.query_analyzer.analyze(query)

        dense_candidates = self.dense_retriever.retrieve(
            analyzed_query,
            top_k=self.config.dense_top_k,
            oversample=self.config.retriever_oversample,
        )
        sparse_candidates = self.sparse_retriever.retrieve(
            analyzed_query,
            top_k=self.config.sparse_top_k,
            oversample=self.config.retriever_oversample,
        )

        fused_candidates = reciprocal_rank_fusion(
            {
                self.dense_retriever.source_name: dense_candidates,
                self.sparse_retriever.source_name: sparse_candidates,
            },
            analyzed_query=analyzed_query,
            rrf_k=self.config.rrf_k,
            limit=self.config.fusion_top_k,
            table_query_boost=self.config.table_query_boost,
            exact_match_boost=self.config.exact_match_boost,
        )
        rerank_input = fused_candidates[: self.config.rerank_top_k]
        if self.config.release_dense_model_before_rerank:
            self.dense_retriever.release_embedding_provider()
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
                "dense_count": len(dense_candidates),
                "sparse_count": len(sparse_candidates),
                "fused_count": len(fused_candidates),
                "reranked_count": len(reranked_candidates),
                "context_count": len(contexts),
                "reranker": self.reranker.model_name,
            },
        )
