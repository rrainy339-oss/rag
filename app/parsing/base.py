from __future__ import annotations

from typing import Protocol

from app.domain.schemas import ParsedDocument, RawDocument


class ParserDependencyError(RuntimeError):
    """Raised when an optional parser dependency is not installed."""


class DocumentParser(Protocol):
    parser_name: str

    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        """Parse a raw document into the normalized document shape."""

