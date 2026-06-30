from __future__ import annotations

from app.chunking.schemas import ChunkType
from app.domain.schemas import ParsedDocument


def build_contextual_text(
    *,
    document: ParsedDocument,
    chunk_type: ChunkType,
    section_path: list[str],
    text: str,
    include_prefix: bool = True,
) -> str:
    if not include_prefix:
        return text

    section = " > ".join(section_path) if section_path else "Root"
    prefix = [
        f"Document: {document.metadata.filename}",
        f"Source: {document.metadata.source_uri}",
        f"Section: {section}",
        f"Chunk type: {chunk_type.value}",
        "Content:",
    ]
    return "\n".join(prefix) + "\n" + text

