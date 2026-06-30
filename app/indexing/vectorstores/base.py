from __future__ import annotations

from typing import Protocol

from app.indexing.schemas import IndexRecord


class VectorStoreDependencyError(RuntimeError):
    """Raised when an optional vector store dependency is unavailable."""


class VectorSearchResult:
    def __init__(self, record: IndexRecord, score: float) -> None:
        self.record = record
        self.score = score


class VectorStore(Protocol):
    def upsert(self, records: list[IndexRecord]) -> None:
        """Insert or replace index records."""

    def delete_by_document(self, document_id: str) -> int:
        """Delete all records for one document and return deleted count."""

    def count(self) -> int:
        """Return indexed record count."""

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]:
        """Return the nearest records for a query vector."""
