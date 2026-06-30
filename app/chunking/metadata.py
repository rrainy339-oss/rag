from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from app.domain.schemas import CitationSpan, ParsedElement


def stable_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def make_chunk_id(
    *,
    document_content_hash: str,
    chunk_type: str,
    chunk_index: int,
    element_order_start: int | None,
    element_order_end: int | None,
    content_hash: str,
) -> str:
    raw = stable_hash(
        document_content_hash,
        chunk_type,
        chunk_index,
        element_order_start,
        element_order_end,
        content_hash,
    )
    return f"chk_{raw[:24]}"


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
    return len(tokens)


def collect_element_ids(elements: Iterable[ParsedElement]) -> list[str]:
    return [element.element_id for element in elements]


def collect_citations(elements: Iterable[ParsedElement]) -> list[CitationSpan]:
    citations: list[CitationSpan] = []
    seen: set[tuple[str, int | None, str | None, str | None]] = set()

    for element in elements:
        if element.citations:
            element_citations = element.citations
        else:
            element_citations = [CitationSpan(source_uri="", text=element.text[:500])]

        for citation in element_citations:
            normalized = citation.model_copy(
                update={
                    "element_id": citation.element_id or element.element_id,
                    "text": citation.text or element.text[:500],
                }
            )
            key = (
                normalized.source_uri,
                normalized.page_number,
                normalized.element_id,
                normalized.text,
            )
            if key not in seen:
                citations.append(normalized)
                seen.add(key)

    return citations


def merge_markdown(elements: Iterable[ParsedElement]) -> str:
    blocks = [element.markdown or element.text for element in elements]
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def merge_text(elements: Iterable[ParsedElement]) -> str:
    blocks = [element.text for element in elements]
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())

