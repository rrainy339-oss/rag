from __future__ import annotations

from typing import Protocol

from app.chunking.schemas import ChunkingResult
from app.domain.schemas import ParsedDocument


class DocumentChunker(Protocol):
    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        """Convert a parsed document into retrieval-ready chunks."""

