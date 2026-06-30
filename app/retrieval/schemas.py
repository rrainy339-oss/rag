from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.chunking.schemas import ChunkType
from app.domain.schemas import CitationSpan
from app.indexing.schemas import IndexRecord


class QueryType(StrEnum):
    GENERAL = "general"
    EXACT = "exact"
    TABLE = "table"
    SUMMARY = "summary"


class RetrievalQuery(BaseModel):
    query: str
    tenant_id: str | None = None
    user_id: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    max_classification: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class AnalyzedQuery(BaseModel):
    query: RetrievalQuery
    query_type: QueryType = QueryType.GENERAL
    exact_terms: list[str] = Field(default_factory=list)


class RetrievalCandidate(BaseModel):
    record: IndexRecord
    source_scores: dict[str, float] = Field(default_factory=dict)
    fused_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextItem(BaseModel):
    context_id: str
    source_chunk_id: str
    parent_chunk_id: str | None = None
    chunk_type: ChunkType
    text: str
    contextual_text: str
    score: float
    source_scores: dict[str, float] = Field(default_factory=dict)
    section_path: list[str] = Field(default_factory=list)
    citations: list[CitationSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    query_type: QueryType
    contexts: list[ContextItem]
    candidates: list[RetrievalCandidate]
    stats: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

