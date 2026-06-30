from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    title: str
    tenant_id: str | None = None
    owner_id: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    principal_ids: list[str] = Field(default_factory=list)
    classification: str | None = None
    status: DocumentStatus = DocumentStatus.UPLOADED
    content_type: str | None = None
    size_bytes: int = 0
    content_hash: str
    source_path: str
    parsed_path: str | None = None
    chunks_path: str | None = None
    index_path: str | None = None
    error_message: str | None = None
    chunk_count: int = 0
    indexed_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
