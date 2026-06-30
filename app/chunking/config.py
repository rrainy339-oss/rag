from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    child_target_tokens: int = Field(default=450, ge=80)
    child_max_tokens: int = Field(default=650, ge=100)
    parent_target_tokens: int = Field(default=1400, ge=200)
    parent_max_tokens: int = Field(default=2000, ge=300)
    oversized_overlap_tokens: int = Field(default=80, ge=0)
    include_title_in_chunks: bool = True
    include_contextual_prefix: bool = True
    isolate_tables: bool = True

