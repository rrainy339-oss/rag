from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerCitation(BaseModel):
    context_id: str
    source_chunk_id: str
    parent_chunk_id: str | None = None
    label: str
    quote: str
    section_path: list[str] = Field(default_factory=list)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    query: str
    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    used_context_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provider: str
    model: str
    prompt: str | None = None
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
