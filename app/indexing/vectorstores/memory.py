from __future__ import annotations

import math

from app.indexing.metadata_filters import metadata_matches
from app.indexing.schemas import IndexRecord
from app.indexing.vectorstores.base import VectorSearchResult


class MemoryVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, IndexRecord] = {}

    def upsert(self, records: list[IndexRecord]) -> None:
        for record in records:
            self.records[record.record_id] = record

    def delete_by_document(self, document_id: str) -> int:
        record_ids = [
            record_id
            for record_id, record in self.records.items()
            if record.document_id == document_id
        ]
        for record_id in record_ids:
            del self.records[record_id]
        return len(record_ids)

    def count(self) -> int:
        return len(self.records)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]:
        filters = filters or {}
        results: list[VectorSearchResult] = []
        for record in self.records.values():
            if not metadata_matches(record.metadata, filters):
                continue
            results.append(
                VectorSearchResult(record=record, score=_cosine(query_vector, record.vector))
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
