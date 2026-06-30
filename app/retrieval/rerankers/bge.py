from __future__ import annotations

from typing import Any

from app.retrieval.rerankers.base import (
    RerankerDependencyError,
    apply_rerank_scores,
    candidate_document_text,
)
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class BGEReranker:
    """Local BGE cross-encoder reranker powered by FlagEmbedding."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = False,
        cache_dir: str | None = None,
        devices: str | list[str] | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model or self._load_model(
            model_name=model_name,
            use_fp16=use_fp16,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            cache_dir=cache_dir,
            devices=devices,
        )
        _ensure_prepare_for_model(self._model)

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
            self._model.compute_score(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
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
        use_fp16: bool,
        batch_size: int,
        max_length: int,
        normalize: bool,
        cache_dir: str | None,
        devices: str | list[str] | None,
    ) -> Any:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RerankerDependencyError(
                "BGE reranking requires FlagEmbedding. Install it in the project "
                "virtual environment with `pip install -e \".[reranking]\"`."
            ) from exc

        return FlagReranker(
            model_name,
            use_fp16=use_fp16,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            cache_dir=cache_dir,
            devices=devices,
        )


def _ensure_prepare_for_model(model: Any) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or hasattr(tokenizer, "prepare_for_model"):
        return

    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if cls_token_id is None or sep_token_id is None:
        return

    def prepare_for_model(
        ids: list[int],
        pair_ids: list[int] | None = None,
        *,
        truncation: str | bool | None = None,
        max_length: int | None = None,
        padding: bool | str = False,
        **_: object,
    ) -> dict[str, list[int]]:
        first = list(ids)
        second = list(pair_ids or [])
        special_count = 4 if pair_ids is not None else 2

        if max_length is not None:
            available = max_length - len(first) - special_count
            if pair_ids is not None and truncation == "only_second":
                second = second[: max(available, 0)]
            elif len(first) + len(second) + special_count > max_length:
                if pair_ids is not None:
                    second = second[: max(available, 0)]
                else:
                    first = first[: max(max_length - special_count, 0)]

        input_ids = [cls_token_id, *first, sep_token_id]
        if pair_ids is not None:
            input_ids.extend([sep_token_id, *second, sep_token_id])

        output = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if padding:
            pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
            target_length = max_length if isinstance(padding, bool) else None
            if target_length is not None and len(input_ids) < target_length:
                pad_length = target_length - len(input_ids)
                output["input_ids"].extend([pad_token_id] * pad_length)
                output["attention_mask"].extend([0] * pad_length)
        return output

    tokenizer.prepare_for_model = prepare_for_model


def _as_float_list(values: Any) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]
