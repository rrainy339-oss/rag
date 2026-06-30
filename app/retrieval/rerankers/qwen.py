from __future__ import annotations

from typing import Any

from app.retrieval.rerankers.base import (
    RerankerDependencyError,
    apply_rerank_scores,
    candidate_document_text,
)
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class QwenReranker:
    """Local Qwen3 cross-encoder reranker via sentence-transformers."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        batch_size: int = 8,
        max_length: int | None = 8192,
        device: str | None = None,
        cache_dir: str | None = None,
        instruction: str | None = None,
        use_sigmoid: bool = False,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.instruction = instruction
        self.use_sigmoid = use_sigmoid
        self._model = model or self._load_model(
            model_name=model_name,
            max_length=max_length,
            device=device,
            cache_dir=cache_dir,
            instruction=instruction,
        )

    def rerank(
        self,
        analyzed_query: AnalyzedQuery,
        candidates: list[RetrievalCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        query = analyzed_query.query.query
        pairs = [(query, candidate_document_text(candidate)) for candidate in candidates]
        scores = _as_float_list(
            self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=_sigmoid_activation() if self.use_sigmoid else None,
            )
        )
        return apply_rerank_scores(
            candidates,
            scores,
            top_k=top_k,
            model_name=self.model_name,
        )

    def _load_model(
        self,
        *,
        model_name: str,
        max_length: int | None,
        device: str | None,
        cache_dir: str | None,
        instruction: str | None,
    ) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerDependencyError(
                "Qwen reranking requires sentence-transformers. Install it in the "
                "project virtual environment with `pip install -e \".[reranking]\"`."
            ) from exc

        prompts = None
        default_prompt_name = None
        if instruction:
            prompts = {"rag": instruction}
            default_prompt_name = "rag"

        return CrossEncoder(
            model_name,
            device=device,
            cache_folder=cache_dir,
            max_length=max_length,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
        )


def _sigmoid_activation() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RerankerDependencyError(
            "Qwen sigmoid scoring requires torch, which should be installed with "
            "sentence-transformers."
        ) from exc
    return torch.nn.Sigmoid()


def _as_float_list(values: Any) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]
