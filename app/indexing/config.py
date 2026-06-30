from __future__ import annotations

from pydantic import BaseModel, Field

from app.chunking.schemas import ChunkType
from app.indexing.schemas import IndexBackend


class IndexingConfig(BaseModel):
    index_version: str = "idx-v1"
    backend: IndexBackend = IndexBackend.MEMORY
    embedding_model: str = "hashing-embedding"
    embedding_dimension: int = Field(default=384, ge=8)
    embedding_batch_size: int = Field(default=32, ge=1)
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    sparse_top_n: int | None = Field(default=512, ge=1)
    index_chunk_types: set[ChunkType] = Field(
        default_factory=lambda: {ChunkType.CHILD, ChunkType.TABLE}
    )
    force_reindex: bool = False
