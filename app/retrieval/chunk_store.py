from __future__ import annotations

from app.chunking.schemas import Chunk, ChunkingResult


class ChunkStore:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks or []}

    @classmethod
    def from_chunking_result(cls, result: ChunkingResult) -> "ChunkStore":
        return cls(result.chunks)

    def get(self, chunk_id: str | None) -> Chunk | None:
        if chunk_id is None:
            return None
        return self._chunks.get(chunk_id)

