from __future__ import annotations

from app.chunking.config import ChunkingConfig
from app.chunking.hierarchical import StructureAwareChunker
from app.chunking.schemas import ChunkingResult
from app.domain.schemas import ParsedDocument


class ChunkingPipeline:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.chunker = StructureAwareChunker(self.config)

    def chunk_document(self, document: ParsedDocument) -> ChunkingResult:
        return self.chunker.chunk(document)

