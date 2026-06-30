from __future__ import annotations

from dataclasses import dataclass, field

from app.chunking.config import ChunkingConfig
from app.chunking.context import build_contextual_text
from app.chunking.metadata import (
    collect_citations,
    collect_element_ids,
    estimate_token_count,
    make_chunk_id,
    merge_markdown,
    merge_text,
    stable_hash,
)
from app.chunking.schemas import Chunk, ChunkingResult, ChunkType
from app.chunking.table_chunker import TableChunker
from app.domain.schemas import ParsedDocument, ParsedElement, ParsedElementType


@dataclass
class ElementSpan:
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class SectionSegment:
    section_path: list[str]
    text_elements: list[ParsedElement] = field(default_factory=list)
    table_elements: list[ParsedElement] = field(default_factory=list)


class StructureAwareChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.table_chunker = TableChunker(self.config)

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        spans = _locate_element_spans(document)
        segments = _build_section_segments(document)

        chunks: list[Chunk] = []
        chunk_index = 0

        for segment in segments:
            parent_chunks = self._build_parent_chunks(
                document=document,
                segment=segment,
                spans=spans,
                starting_index=chunk_index,
            )
            chunks.extend(parent_chunks)
            chunk_index += len(parent_chunks)

            for parent in parent_chunks:
                parent_elements = [
                    element
                    for element in segment.text_elements
                    if element.element_id in set(parent.element_ids)
                ]
                child_chunks = self._build_child_chunks(
                    document=document,
                    parent_chunk=parent,
                    elements=parent_elements,
                    section_path=segment.section_path,
                    spans=spans,
                    starting_index=chunk_index,
                )
                chunks.extend(child_chunks)
                chunk_index += len(child_chunks)

            nearest_parent_id = parent_chunks[0].chunk_id if parent_chunks else None
            for table_element in segment.table_elements:
                span = spans.get(table_element.element_id, ElementSpan())
                table_chunk = self.table_chunker.build_table_chunk(
                    document=document,
                    element=table_element,
                    chunk_index=chunk_index,
                    section_path=segment.section_path,
                    parent_chunk_id=nearest_parent_id,
                    char_start=span.char_start,
                    char_end=span.char_end,
                )
                chunks.append(table_chunk)
                chunk_index += 1

        _attach_neighbor_links(chunks)
        return ChunkingResult(
            document_id=document.document_id,
            chunks=chunks,
            stats={
                "chunk_count": len(chunks),
                "parent_count": sum(1 for chunk in chunks if chunk.chunk_type == ChunkType.PARENT),
                "child_count": sum(1 for chunk in chunks if chunk.chunk_type == ChunkType.CHILD),
                "table_count": sum(1 for chunk in chunks if chunk.chunk_type == ChunkType.TABLE),
            },
        )

    def _build_parent_chunks(
        self,
        *,
        document: ParsedDocument,
        segment: SectionSegment,
        spans: dict[str, ElementSpan],
        starting_index: int,
    ) -> list[Chunk]:
        return self._build_chunks_from_elements(
            document=document,
            elements=segment.text_elements,
            chunk_type=ChunkType.PARENT,
            target_tokens=self.config.parent_target_tokens,
            max_tokens=self.config.parent_max_tokens,
            section_path=segment.section_path,
            spans=spans,
            starting_index=starting_index,
            parent_chunk_id=None,
        )

    def _build_child_chunks(
        self,
        *,
        document: ParsedDocument,
        parent_chunk: Chunk,
        elements: list[ParsedElement],
        section_path: list[str],
        spans: dict[str, ElementSpan],
        starting_index: int,
    ) -> list[Chunk]:
        return self._build_chunks_from_elements(
            document=document,
            elements=elements,
            chunk_type=ChunkType.CHILD,
            target_tokens=self.config.child_target_tokens,
            max_tokens=self.config.child_max_tokens,
            section_path=section_path,
            spans=spans,
            starting_index=starting_index,
            parent_chunk_id=parent_chunk.chunk_id,
        )

    def _build_chunks_from_elements(
        self,
        *,
        document: ParsedDocument,
        elements: list[ParsedElement],
        chunk_type: ChunkType,
        target_tokens: int,
        max_tokens: int,
        section_path: list[str],
        spans: dict[str, ElementSpan],
        starting_index: int,
        parent_chunk_id: str | None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[ParsedElement] = []
        current_tokens = 0

        for element in elements:
            element_tokens = estimate_token_count(element.text or element.markdown or "")
            if element_tokens > max_tokens:
                if current:
                    chunks.append(
                        self._make_chunk(
                            document=document,
                            elements=current,
                            chunk_type=chunk_type,
                            chunk_index=starting_index + len(chunks),
                            section_path=section_path,
                            spans=spans,
                            parent_chunk_id=parent_chunk_id,
                        )
                    )
                    current = []
                    current_tokens = 0
                chunks.extend(
                    self._split_oversized_element(
                        document=document,
                        element=element,
                        chunk_type=chunk_type,
                        starting_index=starting_index + len(chunks),
                        section_path=section_path,
                        parent_chunk_id=parent_chunk_id,
                    )
                )
                continue

            would_exceed = current and current_tokens + element_tokens > max_tokens
            reached_target = current and current_tokens >= target_tokens
            if would_exceed or reached_target:
                chunks.append(
                    self._make_chunk(
                        document=document,
                        elements=current,
                        chunk_type=chunk_type,
                        chunk_index=starting_index + len(chunks),
                        section_path=section_path,
                        spans=spans,
                        parent_chunk_id=parent_chunk_id,
                    )
                )
                current = []
                current_tokens = 0

            current.append(element)
            current_tokens += element_tokens

        if current:
            chunks.append(
                self._make_chunk(
                    document=document,
                    elements=current,
                    chunk_type=chunk_type,
                    chunk_index=starting_index + len(chunks),
                    section_path=section_path,
                    spans=spans,
                    parent_chunk_id=parent_chunk_id,
                )
            )

        return chunks

    def _make_chunk(
        self,
        *,
        document: ParsedDocument,
        elements: list[ParsedElement],
        chunk_type: ChunkType,
        chunk_index: int,
        section_path: list[str],
        spans: dict[str, ElementSpan],
        parent_chunk_id: str | None,
    ) -> Chunk:
        text = merge_text(elements)
        markdown = merge_markdown(elements)
        token_count = estimate_token_count(text)
        element_ids = collect_element_ids(elements)
        element_orders = [element.order_index for element in elements]
        span_values = [spans.get(element.element_id) for element in elements]
        char_starts = [span.char_start for span in span_values if span and span.char_start is not None]
        char_ends = [span.char_end for span in span_values if span and span.char_end is not None]
        content_hash = stable_hash(
            document.metadata.content_hash,
            chunk_type.value,
            "|".join(str(order) for order in element_orders),
            markdown,
        )

        return Chunk(
            chunk_id=make_chunk_id(
                document_content_hash=document.metadata.content_hash,
                chunk_type=chunk_type.value,
                chunk_index=chunk_index,
                element_order_start=min(element_orders) if element_orders else None,
                element_order_end=max(element_orders) if element_orders else None,
                content_hash=content_hash,
            ),
            document_id=document.document_id,
            parent_chunk_id=parent_chunk_id,
            chunk_type=chunk_type,
            text=text,
            markdown=markdown,
            contextual_text=build_contextual_text(
                document=document,
                chunk_type=chunk_type,
                section_path=section_path,
                text=text,
                include_prefix=self.config.include_contextual_prefix,
            ),
            chunk_index=chunk_index,
            section_title=section_path[-1] if section_path else None,
            section_path=section_path,
            element_ids=element_ids,
            citations=collect_citations(elements),
            access=document.metadata.access,
            token_count=token_count,
            content_hash=content_hash,
            char_start=min(char_starts) if char_starts else None,
            char_end=max(char_ends) if char_ends else None,
            element_order_start=min(element_orders) if element_orders else None,
            element_order_end=max(element_orders) if element_orders else None,
            metadata={
                "source_filename": document.metadata.filename,
                "source_file_type": document.metadata.file_type,
                "element_count": len(elements),
            },
        )

    def _split_oversized_element(
        self,
        *,
        document: ParsedDocument,
        element: ParsedElement,
        chunk_type: ChunkType,
        starting_index: int,
        section_path: list[str],
        parent_chunk_id: str | None,
    ) -> list[Chunk]:
        text = element.text or element.markdown or ""
        parts = _split_text_by_tokens(
            text=text,
            max_tokens=(
                self.config.parent_max_tokens
                if chunk_type == ChunkType.PARENT
                else self.config.child_max_tokens
            ),
            overlap_tokens=self.config.oversized_overlap_tokens,
        )
        chunks: list[Chunk] = []
        for offset, part in enumerate(parts):
            part_element = element.model_copy(
                update={
                    "text": part,
                    "markdown": part,
                    "metadata": {
                        **element.metadata,
                        "oversized_split_index": offset,
                        "oversized_split_count": len(parts),
                    },
                }
            )
            chunks.append(
                self._make_chunk(
                    document=document,
                    elements=[part_element],
                    chunk_type=chunk_type,
                    chunk_index=starting_index + offset,
                    section_path=section_path,
                    spans={},
                    parent_chunk_id=parent_chunk_id,
                )
            )
        return chunks


def _build_section_segments(document: ParsedDocument) -> list[SectionSegment]:
    segments: list[SectionSegment] = []
    heading_stack: list[tuple[int, str]] = []
    current: SectionSegment | None = None

    def close_current() -> None:
        nonlocal current
        if current and (current.text_elements or current.table_elements):
            segments.append(current)
        current = None

    for element in sorted(document.elements, key=lambda item: item.order_index):
        if element.type == ParsedElementType.TITLE:
            close_current()
            level = element.level or 1
            heading_stack[:] = [
                (heading_level, title)
                for heading_level, title in heading_stack
                if heading_level < level
            ]
            heading_stack.append((level, element.text))
            current = SectionSegment(section_path=[title for _, title in heading_stack])
            if current and element.text:
                current.text_elements.append(element)
            continue

        if current is None:
            current = SectionSegment(section_path=[document.metadata.filename])

        if element.type == ParsedElementType.TABLE:
            current.table_elements.append(element)
        else:
            current.text_elements.append(element)

    close_current()
    return segments


def _locate_element_spans(document: ParsedDocument) -> dict[str, ElementSpan]:
    spans: dict[str, ElementSpan] = {}
    cursor = 0
    markdown = document.markdown

    for element in sorted(document.elements, key=lambda item: item.order_index):
        needle = (element.markdown or element.text).strip()
        if not needle:
            spans[element.element_id] = ElementSpan()
            continue

        position = markdown.find(needle, cursor)
        if position == -1:
            position = markdown.find(needle)
        if position == -1:
            spans[element.element_id] = ElementSpan()
            continue

        end = position + len(needle)
        spans[element.element_id] = ElementSpan(char_start=position, char_end=end)
        cursor = end

    return spans


def _split_text_by_tokens(
    *, text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    parts: list[str] = []
    start = 0
    overlap = min(overlap_tokens, max_tokens // 3)
    while start < len(words):
        end = min(start + max_tokens, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return parts


def _attach_neighbor_links(chunks: list[Chunk]) -> None:
    retrieval_chunks = [
        chunk
        for chunk in sorted(chunks, key=lambda item: item.chunk_index)
        if chunk.chunk_type in {ChunkType.CHILD, ChunkType.TABLE}
    ]
    for index, chunk in enumerate(retrieval_chunks):
        neighbors: list[str] = []
        if index > 0:
            neighbors.append(retrieval_chunks[index - 1].chunk_id)
        if index + 1 < len(retrieval_chunks):
            neighbors.append(retrieval_chunks[index + 1].chunk_id)
        chunk.neighbor_chunk_ids = neighbors

