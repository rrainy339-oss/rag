from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.chunking.schemas import ChunkType
from app.domain.schemas import AccessControl, CitationSpan


class IndexBackend(StrEnum):
    QDRANT_HYBRID = "qdrant_hybrid"


class IndexingStatus(StrEnum):
    INDEXED = "indexed"
    SKIPPED = "skipped"


class IndexRecord(BaseModel):
    record_id: str
    chunk_id: str
    document_id: str
    parent_chunk_id: str | None = None
    chunk_type: ChunkType
    text: str
    contextual_text: str
    vector: list[float] = Field(default_factory=list)
    sparse_terms: dict[str, float] = Field(default_factory=dict)
    sparse_vector_indices: list[int] = Field(default_factory=list)
    sparse_vector_values: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    access: AccessControl = Field(default_factory=AccessControl)
    citations: list[CitationSpan] = Field(default_factory=list)
    content_hash: str
    embedding_model: str
    embedding_dimension: int
    index_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IndexManifest(BaseModel):
    document_id: str
    source_uri: str | None = None
    document_hash: str
    chunk_hashes: dict[str, str] = Field(default_factory=dict)
    embedding_model: str
    embedding_dimension: int
    index_version: str
    backend: IndexBackend
    status: IndexingStatus
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def matches(
        self,
        *,
        document_hash: str,
        chunk_hashes: dict[str, str],
        embedding_model: str,
        embedding_dimension: int,
        index_version: str,
        backend: IndexBackend,
    ) -> bool:
        return (
            self.document_hash == document_hash
            and self.chunk_hashes == chunk_hashes
            and self.embedding_model == embedding_model
            and self.embedding_dimension == embedding_dimension
            and self.index_version == index_version
            and self.backend == backend
        )


class IndexingResult(BaseModel):
    document_id: str
    status: IndexingStatus
    indexed_count: int
    skipped_count: int = 0
    deleted_count: int = 0
    manifest: IndexManifest
    records: list[IndexRecord] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
