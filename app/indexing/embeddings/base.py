from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EmbeddingDependencyError(RuntimeError):
    """Raised when an optional embedding dependency is unavailable."""


@dataclass(frozen=True)
class SparseEmbeddingVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class HybridEmbedding:
    dense: list[float]
    sparse: SparseEmbeddingVector


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts for retrieval indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string for retrieval."""
