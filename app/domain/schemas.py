from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ParsedElementType(StrEnum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    FORMULA = "formula"
    UNKNOWN = "unknown"


class AccessControl(BaseModel):
    tenant_id: str | None = None
    allowed_user_ids: list[str] = Field(default_factory=list)
    allowed_group_ids: list[str] = Field(default_factory=list)
    classification: str | None = None


class RawDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    path: Path
    source_uri: str
    filename: str
    file_type: str
    mime_type: str | None = None
    size_bytes: int
    content_hash: str
    modified_at: datetime | None = None
    access: AccessControl = Field(default_factory=AccessControl)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class CitationSpan(BaseModel):
    source_uri: str
    page_number: int | None = None
    element_id: str | None = None
    bbox: BoundingBox | None = None
    text: str | None = None


class TableData(BaseModel):
    caption: str | None = None
    markdown: str | None = None
    html: str | None = None
    rows: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageAsset(BaseModel):
    uri: str | None = None
    caption: str | None = None
    page_number: int | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedElement(BaseModel):
    element_id: str = Field(default_factory=lambda: str(uuid4()))
    type: ParsedElementType
    text: str = ""
    markdown: str | None = None
    order_index: int
    page_number: int | None = None
    level: int | None = None
    table: TableData | None = None
    image: ImageAsset | None = None
    citations: list[CitationSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    section_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    level: int = 1
    text: str = ""
    element_ids: list[str] = Field(default_factory=list)
    children: list["Section"] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class DocumentMetadata(BaseModel):
    source_uri: str
    filename: str
    file_type: str
    mime_type: str | None = None
    content_hash: str
    size_bytes: int
    modified_at: datetime | None = None
    parser_name: str
    parser_version: str | None = None
    access: AccessControl = Field(default_factory=AccessControl)
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    text: str
    markdown: str
    elements: list[ParsedElement] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    images: list[ImageAsset] = Field(default_factory=list)
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stats: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

