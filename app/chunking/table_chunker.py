from __future__ import annotations

from app.chunking.config import ChunkingConfig
from app.chunking.context import build_contextual_text
from app.chunking.metadata import (
    collect_citations,
    collect_element_ids,
    estimate_token_count,
    make_chunk_id,
    stable_hash,
)
from app.chunking.schemas import Chunk, ChunkType
from app.domain.schemas import ParsedDocument, ParsedElement


class TableChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def build_table_chunk(
        self,
        *,
        document: ParsedDocument,
        element: ParsedElement,
        chunk_index: int,
        section_path: list[str],
        parent_chunk_id: str | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Chunk:
        markdown = element.markdown or element.text
        text = element.text or markdown
        content_hash = stable_hash(
            document.metadata.content_hash,
            ChunkType.TABLE.value,
            element.order_index,
            markdown,
        )
        token_count = estimate_token_count(text)

        return Chunk(
            chunk_id=make_chunk_id(
                document_content_hash=document.metadata.content_hash,
                chunk_type=ChunkType.TABLE.value,
                chunk_index=chunk_index,
                element_order_start=element.order_index,
                element_order_end=element.order_index,
                content_hash=content_hash,
            ),
            document_id=document.document_id,
            parent_chunk_id=parent_chunk_id,
            chunk_type=ChunkType.TABLE,
            text=text,
            markdown=markdown,
            contextual_text=build_contextual_text(
                document=document,
                chunk_type=ChunkType.TABLE,
                section_path=section_path,
                text=markdown,
                include_prefix=self.config.include_contextual_prefix,
            ),
            chunk_index=chunk_index,
            section_title=section_path[-1] if section_path else None,
            section_path=section_path,
            element_ids=collect_element_ids([element]),
            citations=collect_citations([element]),
            access=document.metadata.access,
            token_count=token_count,
            content_hash=content_hash,
            char_start=char_start,
            char_end=char_end,
            element_order_start=element.order_index,
            element_order_end=element.order_index,
            metadata={
                "source_filename": document.metadata.filename,
                "table_row_count": len(element.table.rows) if element.table else 0,
                "table_has_html": bool(element.table and element.table.html),
            },
        )

