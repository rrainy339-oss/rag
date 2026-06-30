from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    dense_top_k: int = Field(default=50, ge=1)
    sparse_top_k: int = Field(default=50, ge=1)
    fusion_top_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=40, ge=1)
    final_top_k: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    table_query_boost: float = Field(default=1.15, ge=1.0)
    exact_match_boost: float = Field(default=1.2, ge=1.0)
    retriever_oversample: int = Field(default=3, ge=1)
    release_dense_model_before_rerank: bool = False
