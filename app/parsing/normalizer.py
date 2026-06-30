from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from app.domain.schemas import (
    CitationSpan,
    DocumentMetadata,
    ImageAsset,
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    RawDocument,
    Section,
    TableData,
)


class ParsedDocumentNormalizer:
    def from_docling(
        self,
        *,
        raw_document: RawDocument,
        markdown: str,
        text: str | None,
        docling_data: Mapping[str, Any] | None,
        parser_name: str,
        parser_version: str | None = None,
    ) -> ParsedDocument:
        markdown = markdown.strip()
        text = (text or _markdown_to_plain_text(markdown)).strip()

        elements = _elements_from_markdown(markdown, raw_document.source_uri)
        tables = [element.table for element in elements if element.table is not None]
        images = _extract_images(docling_data)
        sections = _sections_from_elements(elements)

        metadata = DocumentMetadata(
            source_uri=raw_document.source_uri,
            filename=raw_document.filename,
            file_type=raw_document.file_type,
            mime_type=raw_document.mime_type,
            content_hash=raw_document.content_hash,
            size_bytes=raw_document.size_bytes,
            modified_at=raw_document.modified_at,
            parser_name=parser_name,
            parser_version=parser_version,
            access=raw_document.access,
            extra={
                **raw_document.metadata,
                "docling_keys": sorted(docling_data.keys()) if docling_data else [],
            },
        )

        return ParsedDocument(
            document_id=raw_document.document_id,
            metadata=metadata,
            text=text,
            markdown=markdown,
            elements=elements,
            sections=sections,
            tables=tables,
            images=images,
            stats={
                "element_count": len(elements),
                "section_count": len(sections),
                "table_count": len(tables),
                "image_count": len(images),
                "character_count": len(text),
            },
        )


def _elements_from_markdown(markdown: str, source_uri: str) -> list[ParsedElement]:
    blocks = _split_markdown_blocks(markdown)
    elements: list[ParsedElement] = []

    for index, block in enumerate(blocks):
        stripped = block.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        table_rows = _parse_markdown_table(stripped)

        if heading_match:
            element_type = ParsedElementType.TITLE
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            table = None
        elif table_rows:
            element_type = ParsedElementType.TABLE
            level = None
            text = _markdown_to_plain_text(stripped)
            table = TableData(markdown=stripped, rows=table_rows)
        elif stripped.startswith(("- ", "* ", "1. ")):
            element_type = ParsedElementType.LIST
            level = None
            text = _markdown_to_plain_text(stripped)
            table = None
        elif stripped.startswith("```"):
            element_type = ParsedElementType.CODE
            level = None
            text = stripped.strip("`").strip()
            table = None
        else:
            element_type = ParsedElementType.PARAGRAPH
            level = None
            text = _markdown_to_plain_text(stripped)
            table = None

        elements.append(
            ParsedElement(
                type=element_type,
                text=text,
                markdown=stripped,
                order_index=index,
                level=level,
                table=table,
                citations=[CitationSpan(source_uri=source_uri, text=text[:500])],
            )
        )

    return elements


def _split_markdown_blocks(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_table = False

    for line in lines:
        if _looks_like_table_line(line):
            if current and not in_table:
                blocks.append("\n".join(current).strip())
                current = []
            in_table = True
            current.append(line)
            continue

        if in_table:
            blocks.append("\n".join(current).strip())
            current = []
            in_table = False

        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current).strip())
            current = []

    if current:
        blocks.append("\n".join(current).strip())

    return [block for block in blocks if block]


def _parse_markdown_table(block: str) -> list[list[str]]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2 or not all(_looks_like_table_line(line) for line in lines[:2]):
        return []
    if not re.fullmatch(r"\|?[\s:\-|]+\|?", lines[1]):
        return []

    rows: list[list[str]] = []
    for line in lines:
        if re.fullmatch(r"\|?[\s:\-|]+\|?", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _looks_like_table_line(line: str) -> bool:
    return line.count("|") >= 2


def _sections_from_elements(elements: list[ParsedElement]) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None

    for element in elements:
        if element.type == ParsedElementType.TITLE:
            current = Section(
                title=element.text,
                level=element.level or 1,
                element_ids=[element.element_id],
            )
            sections.append(current)
        elif current is not None:
            current.element_ids.append(element.element_id)
            current.text = (current.text + "\n\n" + element.text).strip()

    return sections


def _extract_images(docling_data: Mapping[str, Any] | None) -> list[ImageAsset]:
    if not docling_data:
        return []

    images: list[ImageAsset] = []
    for item in _walk_values(docling_data):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("type") or "").lower()
        if label not in {"picture", "image", "figure"}:
            continue
        images.append(
            ImageAsset(
                caption=str(item.get("caption") or item.get("text") or "") or None,
                metadata={key: value for key, value in item.items() if key not in {"caption", "text"}},
            )
        )
    return images


def _walk_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_values(child))
    return values


def _markdown_to_plain_text(markdown: str) -> str:
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"[*_`>]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_docling_export(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, Mapping) else None
    return None

