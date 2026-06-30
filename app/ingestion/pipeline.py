from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.connectors.local_file import LocalFileConnector
from app.domain.schemas import AccessControl, ParsedDocument
from app.parsing.base import DocumentParser
from app.parsing.docling_parser import DoclingParser


class LocalParsingPipeline:
    def __init__(
        self,
        *,
        parser: DocumentParser | None = None,
        recursive: bool = True,
        pdf_backend: str | None = None,
        do_ocr: bool | None = None,
        artifacts_path: str | Path | None = None,
        access: AccessControl | None = None,
        access_by_filename: dict[str, AccessControl] | None = None,
    ) -> None:
        self.parser = parser or DoclingParser(
            pdf_backend=pdf_backend,
            do_ocr=do_ocr,
            artifacts_path=artifacts_path,
        )
        self.recursive = recursive
        self.access = access or AccessControl()
        self.access_by_filename = access_by_filename or {}

    def parse_path(self, path: str | Path) -> Iterable[ParsedDocument]:
        connector = LocalFileConnector(
            path,
            recursive=self.recursive,
            access=self.access,
            access_by_filename=self.access_by_filename,
        )
        for raw_document in connector.iter_documents():
            yield self.parser.parse(raw_document)
