from __future__ import annotations

import gc
from typing import Any

from app.indexing.embeddings.base import (
    EmbeddingDependencyError,
    HybridEmbedding,
    SparseEmbeddingVector,
)


class BGEM3EmbeddingProvider:
    """BGE-M3 dense embedding provider.

    This adapter is intentionally lazy so the indexing module can be tested
    without installing model-serving dependencies.
    """

    model_name = "BAAI/bge-m3"
    dimension = 1024

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        normalize_embeddings: bool = True,
        max_length: int = 8192,
        batch_size: int = 12,
        devices: str | list[str] | None = None,
        cache_dir: str | None = None,
        truncate_dim: int | None = None,
        sparse_top_n: int | None = 512,
    ) -> None:
        self.model_name = model_name
        self.dimension = truncate_dim or 1024
        self.max_length = max_length
        self.batch_size = batch_size
        self.sparse_top_n = sparse_top_n
        self._model = self._load_model(
            model_name=model_name,
            use_fp16=use_fp16,
            normalize_embeddings=normalize_embeddings,
            devices=devices,
            cache_dir=cache_dir,
            truncate_dim=truncate_dim,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return _dense_vectors_to_list(output)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents_hybrid(self, texts: list[str]) -> list[HybridEmbedding]:
        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = _dense_vectors_to_list(output)
        sparse_vectors = _lexical_weights_to_sparse_vectors(
            output.get("lexical_weights") if isinstance(output, dict) else None,
            expected_count=len(texts),
            top_n=self.sparse_top_n,
        )
        if len(dense_vectors) != len(sparse_vectors):
            raise ValueError("BGE-M3 returned mismatched dense and sparse vectors.")
        return [
            HybridEmbedding(dense=dense, sparse=sparse)
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def embed_query_hybrid(self, text: str) -> HybridEmbedding:
        return self.embed_documents_hybrid([text])[0]

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(
        self,
        *,
        model_name: str,
        use_fp16: bool,
        normalize_embeddings: bool,
        devices: str | list[str] | None,
        cache_dir: str | None,
        truncate_dim: int | None,
    ) -> Any:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "BGE-M3 requires FlagEmbedding. Install the indexing extra or run "
                "`pip install FlagEmbedding` inside the project virtual environment."
            ) from exc

        return BGEM3FlagModel(
            model_name,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            devices=devices,
            cache_dir=cache_dir,
            truncate_dim=truncate_dim,
        )


def _dense_vectors_to_list(output: Any) -> list[list[float]]:
    vectors = output["dense_vecs"] if isinstance(output, dict) else output
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [list(vector) for vector in vectors]


def _lexical_weights_to_sparse_vectors(
    lexical_weights: Any,
    *,
    expected_count: int,
    top_n: int | None,
) -> list[SparseEmbeddingVector]:
    if lexical_weights is None:
        return [SparseEmbeddingVector(indices=[], values=[]) for _ in range(expected_count)]
    if isinstance(lexical_weights, dict):
        lexical_weights = [lexical_weights]
    sparse_vectors = [
        _lexical_weights_to_sparse_vector(item, top_n=top_n)
        for item in lexical_weights
    ]
    if len(sparse_vectors) != expected_count:
        raise ValueError(
            "BGE-M3 returned "
            f"{len(sparse_vectors)} sparse vector(s), expected {expected_count}."
        )
    return sparse_vectors


def _lexical_weights_to_sparse_vector(
    lexical_weights: dict[object, object],
    *,
    top_n: int | None,
) -> SparseEmbeddingVector:
    weighted_indices: list[tuple[int, float]] = []
    for raw_index, raw_weight in lexical_weights.items():
        try:
            index = int(raw_index)
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weighted_indices.append((index, weight))

    weighted_indices.sort(key=lambda item: item[1], reverse=True)
    if top_n is not None and top_n > 0:
        weighted_indices = weighted_indices[:top_n]
    weighted_indices.sort(key=lambda item: item[0])
    return SparseEmbeddingVector(
        indices=[index for index, _ in weighted_indices],
        values=[weight for _, weight in weighted_indices],
    )
