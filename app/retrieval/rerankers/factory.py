from __future__ import annotations

from app.retrieval.rerankers.base import Reranker
from app.retrieval.rerankers.bge import BGEReranker
from app.retrieval.rerankers.cohere import CohereReranker
from app.retrieval.rerankers.config import RerankerConfig, RerankerProvider
from app.retrieval.rerankers.noop import NoopReranker
from app.retrieval.rerankers.qwen import QwenReranker
from app.retrieval.rerankers.voyage import VoyageReranker
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


DEFAULT_MODELS = {
    RerankerProvider.NOOP: "noop-reranker",
    RerankerProvider.BGE: "BAAI/bge-reranker-v2-m3",
    RerankerProvider.QWEN: "Qwen/Qwen3-Reranker-0.6B",
    RerankerProvider.COHERE: "rerank-v4.0-pro",
    RerankerProvider.VOYAGE: "rerank-2.5",
}


class LazyReranker:
    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.model_name = config.model_name or DEFAULT_MODELS[config.provider]
        self._reranker: Reranker | None = None

    def rerank(
        self,
        analyzed_query: AnalyzedQuery,
        candidates: list[RetrievalCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if self._reranker is None:
            self._reranker = _build_eager_reranker(self.config)
        return self._reranker.rerank(analyzed_query, candidates, top_k=top_k)


def build_reranker(config: RerankerConfig | None = None, *, lazy: bool = True) -> Reranker:
    config = config or RerankerConfig()
    if config.provider == RerankerProvider.NOOP:
        return NoopReranker()
    if lazy:
        return LazyReranker(config)

    return _build_eager_reranker(config)


def _build_eager_reranker(config: RerankerConfig) -> Reranker:
    model_name = config.model_name or DEFAULT_MODELS[config.provider]

    if config.provider == RerankerProvider.NOOP:
        return NoopReranker()
    if config.provider == RerankerProvider.BGE:
        return BGEReranker(
            model_name=model_name,
            use_fp16=config.use_fp16,
            batch_size=config.batch_size,
            max_length=config.max_length or 512,
            normalize=config.normalize,
            cache_dir=config.cache_dir,
            devices=config.device,
        )
    if config.provider == RerankerProvider.QWEN:
        return QwenReranker(
            model_name=model_name,
            batch_size=config.batch_size,
            max_length=config.max_length or 8192,
            device=config.device,
            cache_dir=config.cache_dir,
            instruction=config.instruction,
            use_sigmoid=config.normalize,
        )
    if config.provider == RerankerProvider.COHERE:
        return CohereReranker(
            model_name=model_name,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_tokens_per_doc=config.max_tokens_per_doc,
        )
    if config.provider == RerankerProvider.VOYAGE:
        return VoyageReranker(
            model_name=model_name,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )

    raise ValueError(f"Unsupported reranker provider: {config.provider}")
