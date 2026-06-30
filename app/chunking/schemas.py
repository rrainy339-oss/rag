from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.schemas import AccessControl, CitationSpan


class ChunkType(StrEnum):
    PARENT = "parent"
    CHILD = "child"
    TABLE = "table"


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_chunk_id: str | None = None
    chunk_type: ChunkType
    text: str
    markdown: str
    contextual_text: str
    chunk_index: int
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    citations: list[CitationSpan] = Field(default_factory=list)
    access: AccessControl = Field(default_factory=AccessControl)
    token_count: int = 0
    content_hash: str
    char_start: int | None = None
    char_end: int | None = None
    element_order_start: int | None = None
    element_order_end: int | None = None
    neighbor_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    document_id: str
    chunks: list[Chunk]
    stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

