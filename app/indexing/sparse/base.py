from __future__ import annotations

from typing import Protocol

from app.indexing.schemas import IndexRecord


class SparseSearchResult:
    def __init__(self, record: IndexRecord, score: float) -> None:
        self.record = record
        self.score = score


class SparseStore(Protocol):
    def upsert(self, records: list[IndexRecord]) -> None:
        """Insert or replace sparse records."""

    def delete_by_document(self, document_id: str) -> int:
        """Delete all records for a document."""

    def count(self) -> int:
        """Return indexed record count."""

