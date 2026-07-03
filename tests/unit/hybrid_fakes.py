from __future__ import annotations

from app.indexing.embeddings.base import HybridEmbedding, SparseEmbeddingVector
from app.indexing.metadata_filters import metadata_matches
from app.indexing.schemas import IndexRecord
from app.indexing.vectorstores.base import VectorSearchResult


class FakeHybridEmbeddingProvider:
    model_name = "BAAI/bge-m3-test"
    dimension = 8

    def __init__(self, **kwargs: object) -> None:
        self.model_name = str(kwargs.get("model_name") or self.model_name)
        self.closed = False

    def embed_documents_hybrid(self, texts: list[str]) -> list[HybridEmbedding]:
        return [self._embedding_for_text(text) for text in texts]

    def embed_query_hybrid(self, text: str) -> HybridEmbedding:
        return self._embedding_for_text(text)

    def close(self) -> None:
        self.closed = True

    def _embedding_for_text(self, text: str) -> HybridEmbedding:
        normalized = text.lower()
        if "remote" in normalized:
            sparse_index = 10
            dense = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif "health insurance" in normalized or "table" in normalized:
            sparse_index = 20
            dense = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif "benefits" in normalized:
            sparse_index = 30
            dense = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            sparse_index = 40
            dense = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        return HybridEmbedding(
            dense=dense,
            sparse=SparseEmbeddingVector(indices=[sparse_index], values=[1.0]),
        )


class CapturingHybridVectorStore:
    def __init__(self, records: list[IndexRecord] | None = None) -> None:
        self.records = list(records or [])
        self.closed = False
        self.upsert_calls = 0
        self.set_payload_calls = 0

    def delete_by_document(self, document_id: str) -> int:
        before = len(self.records)
        self.records = [
            record for record in self.records if record.document_id != document_id
        ]
        return before - len(self.records)

    def upsert(self, records: list[IndexRecord]) -> None:
        self.upsert_calls += 1
        existing = {record.record_id: record for record in self.records}
        for record in records:
            existing[record.record_id] = record
        self.records = list(existing.values())

    def set_payload(self, records: list[IndexRecord]) -> int:
        self.set_payload_calls += 1
        existing = {record.record_id: record for record in self.records}
        updated_count = 0
        for record in records:
            current = existing.get(record.record_id)
            if current is None:
                continue
            existing[record.record_id] = current.model_copy(
                update={
                    "metadata": record.metadata,
                    "access": record.access,
                    "document_id": record.document_id,
                }
            )
            updated_count += 1
        self.records = list(existing.values())
        return updated_count

    def count(self) -> int:
        return len(self.records)

    def hybrid_search(
        self,
        *,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        filters: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]:
        del dense_top_k, sparse_top_k
        filters = filters or {}
        scored: list[VectorSearchResult] = []
        for record in self.records:
            if not metadata_matches(record.metadata, filters):
                continue
            sparse_score = _sparse_overlap(
                sparse_indices,
                sparse_values,
                record.sparse_vector_indices,
                record.sparse_vector_values,
            )
            dense_score = _dot(dense_vector, record.vector)
            scored.append(VectorSearchResult(record, sparse_score + dense_score))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def close(self) -> None:
        self.closed = True


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sparse_overlap(
    left_indices: list[int],
    left_values: list[float],
    right_indices: list[int],
    right_values: list[float],
) -> float:
    right = dict(zip(right_indices, right_values, strict=True))
    return sum(value * right.get(index, 0.0) for index, value in zip(left_indices, left_values, strict=True))
