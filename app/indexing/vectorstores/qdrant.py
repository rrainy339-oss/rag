from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.domain.permissions import (
    CLASSIFICATION_LEVEL_LTE_FILTER,
    PRINCIPAL_IDS_ANY_FILTER,
)
from app.domain.schemas import AccessControl, CitationSpan
from app.indexing.schemas import IndexRecord
from app.indexing.vectorstores.base import VectorSearchResult, VectorStoreDependencyError


class QdrantHybridVectorStore:
    """Qdrant adapter for native dense+sparse hybrid retrieval."""

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | Path | None = None,
        api_key: str | None = None,
        collection_name: str = "rag_chunks_hybrid",
        vector_size: int = 1024,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        timeout: int | None = None,
    ) -> None:
        if url and path:
            raise ValueError("QdrantHybridVectorStore accepts either url or path, not both.")

        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise VectorStoreDependencyError(
                "QdrantHybridVectorStore requires qdrant-client. Install the indexing "
                "extra or run `pip install qdrant-client` inside .venv."
            ) from exc

        self._models = models
        if path is not None:
            storage_path = Path(path)
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path), timeout=timeout)
        else:
            self.client = QdrantClient(
                url=url or "http://localhost:6333",
                api_key=api_key,
                timeout=timeout,
            )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self._ensure_collection()

    def upsert(self, records: list[IndexRecord]) -> None:
        if not records:
            return

        points = [
            self._models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, record.record_id)),
                vector={
                    self.dense_vector_name: record.vector,
                    self.sparse_vector_name: self._models.SparseVector(
                        indices=record.sparse_vector_indices,
                        values=record.sparse_vector_values,
                    ),
                },
                payload=_record_payload(record),
            )
            for record in records
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def delete_by_document(self, document_id: str) -> int:
        query_filter = self._filter_for_document(document_id)
        count = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        ).count
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.FilterSelector(filter=query_filter),
            wait=True,
        )
        return int(count)

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return int(result.count)

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
        query_filter = _metadata_filter(self._models, filters or {})
        prefetch = [
            self._models.Prefetch(
                query=dense_vector,
                using=self.dense_vector_name,
                filter=query_filter,
                limit=dense_top_k,
            ),
            self._models.Prefetch(
                query=self._models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=self.sparse_vector_name,
                filter=query_filter,
                limit=sparse_top_k,
            ),
        ]
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=self._models.FusionQuery(fusion=self._models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        return [
            VectorSearchResult(
                record=_record_from_point(point, dense_vector_name=self.dense_vector_name),
                score=float(point.score),
            )
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            collection = self.client.get_collection(self.collection_name)
            actual_size = _collection_vector_size(collection, self.dense_vector_name)
            if actual_size is None:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} does not have dense "
                    f"vector {self.dense_vector_name!r}. Use a new hybrid collection."
                )
            if actual_size != self.vector_size:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} dense vector "
                    f"{self.dense_vector_name!r} has size {actual_size}, expected "
                    f"{self.vector_size}."
                )
            if not _collection_has_sparse_vector(collection, self.sparse_vector_name):
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} does not have sparse "
                    f"vector {self.sparse_vector_name!r}. Use a new hybrid collection."
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: self._models.VectorParams(
                    size=self.vector_size,
                    distance=self._models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.sparse_vector_name: self._models.SparseVectorParams()
            },
        )

    def _filter_for_document(self, document_id: str) -> Any:
        return self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="document_id",
                    match=self._models.MatchValue(value=document_id),
                )
            ]
        )


def _record_payload(record: IndexRecord) -> dict[str, Any]:
    return {
        **record.metadata,
        "metadata": record.metadata,
        "record_id": record.record_id,
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
        "parent_chunk_id": record.parent_chunk_id,
        "chunk_type": record.chunk_type.value,
        "text": record.text,
        "contextual_text": record.contextual_text,
        "sparse_terms": record.sparse_terms,
        "sparse_vector_indices": record.sparse_vector_indices,
        "sparse_vector_values": record.sparse_vector_values,
        "content_hash": record.content_hash,
        "embedding_model": record.embedding_model,
        "embedding_dimension": record.embedding_dimension,
        "index_version": record.index_version,
        "access": record.access.model_dump(mode="json"),
        "citations": [citation.model_dump(mode="json") for citation in record.citations],
    }


def _record_from_point(
    point: Any,
    *,
    dense_vector_name: str | None = None,
) -> IndexRecord:
    payload = dict(point.payload or {})
    vector = point.vector
    if isinstance(vector, dict):
        vector = next(iter(vector.values()), [])

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "record_id",
                "chunk_id",
                "document_id",
                "parent_chunk_id",
                "chunk_type",
                "text",
                "contextual_text",
                "sparse_terms",
                "sparse_vector_indices",
                "sparse_vector_values",
                "content_hash",
                "embedding_model",
                "embedding_dimension",
                "index_version",
                "access",
                "citations",
            }
        }

    return IndexRecord(
        record_id=str(payload["record_id"]),
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        parent_chunk_id=payload.get("parent_chunk_id"),
        chunk_type=payload["chunk_type"],
        text=str(payload.get("text") or ""),
        contextual_text=str(payload.get("contextual_text") or ""),
        vector=_dense_vector_from_point_vector(vector, dense_vector_name=dense_vector_name),
        sparse_terms=dict(payload.get("sparse_terms") or {}),
        sparse_vector_indices=[int(value) for value in payload.get("sparse_vector_indices") or []],
        sparse_vector_values=[float(value) for value in payload.get("sparse_vector_values") or []],
        metadata=metadata,
        access=AccessControl.model_validate(payload.get("access") or {}),
        citations=[
            CitationSpan.model_validate(citation)
            for citation in payload.get("citations") or []
        ],
        content_hash=str(payload["content_hash"]),
        embedding_model=str(payload["embedding_model"]),
        embedding_dimension=int(payload["embedding_dimension"]),
        index_version=str(payload["index_version"]),
    )


def _metadata_filter(models: Any, filters: dict[str, object]) -> Any | None:
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if key == PRINCIPAL_IDS_ANY_FILTER:
            values = _as_list(value)
            if not values:
                return _impossible_filter(models)
            conditions.append(
                models.FieldCondition(
                    key="principal_ids",
                    match=models.MatchAny(any=values),
                )
            )
            continue

        if key == CLASSIFICATION_LEVEL_LTE_FILTER:
            try:
                limit = float(value)
            except (TypeError, ValueError):
                return _impossible_filter(models)
            conditions.append(
                models.FieldCondition(
                    key="classification_level",
                    range=models.Range(lte=limit),
                )
            )
            continue

        if _is_many(value):
            match = models.MatchAny(any=_as_list(value))
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=conditions)


def _impossible_filter(models: Any) -> Any:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="__never_match__",
                match=models.MatchValue(value="__never_match__"),
            )
        ]
    )


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if _is_many(value):
        return list(value)
    return [value]


def _is_many(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def _dense_vector_from_point_vector(
    vector: Any,
    *,
    dense_vector_name: str | None = None,
) -> list[float]:
    if isinstance(vector, dict):
        if dense_vector_name and dense_vector_name in vector:
            value = vector[dense_vector_name]
            if isinstance(value, list):
                return [float(item) for item in value]
            if hasattr(value, "tolist"):
                return [float(item) for item in value.tolist()]
        for value in vector.values():
            if isinstance(value, list):
                return [float(item) for item in value]
            if hasattr(value, "tolist"):
                return [float(item) for item in value.tolist()]
        return []
    if isinstance(vector, list):
        return [float(item) for item in vector]
    if hasattr(vector, "tolist"):
        return [float(item) for item in vector.tolist()]
    return []


def _collection_vector_size(collection: Any, vector_name: str | None = None) -> int | None:
    vectors = collection.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict) and vectors:
        if vector_name is not None:
            vector = vectors.get(vector_name)
            return int(vector.size) if hasattr(vector, "size") else None
        for vector in vectors.values():
            if hasattr(vector, "size"):
                return int(vector.size)
    return None


def _collection_has_sparse_vector(collection: Any, sparse_vector_name: str) -> bool:
    sparse_vectors = getattr(collection.config.params, "sparse_vectors", None)
    if sparse_vectors is None:
        sparse_vectors = getattr(collection.config.params, "sparse_vectors_config", None)
    return isinstance(sparse_vectors, dict) and sparse_vector_name in sparse_vectors
