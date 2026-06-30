from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from app.domain.schemas import ParsedDocument, RawDocument
from app.parsing.base import ParserDependencyError
from app.parsing.normalizer import ParsedDocumentNormalizer, parse_docling_export


ConverterFactory = Callable[[], Any]
SUPPORTED_PDF_BACKENDS = {
    "docling_parse",
    "threaded_docling_parse",
    "pypdfium2",
}


class DoclingParser:
    parser_name = "docling"

    def __init__(
        self,
        *,
        converter_factory: ConverterFactory | None = None,
        normalizer: ParsedDocumentNormalizer | None = None,
        pdf_backend: str | None = None,
        do_ocr: bool | None = None,
        artifacts_path: str | Path | None = None,
    ) -> None:
        if pdf_backend is not None and pdf_backend not in SUPPORTED_PDF_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_PDF_BACKENDS))
            raise ValueError(
                f"Unsupported Docling PDF backend {pdf_backend!r}. "
                f"Use one of: {supported}."
            )
        self._converter_factory = converter_factory
        self._normalizer = normalizer or ParsedDocumentNormalizer()
        self._pdf_backend = pdf_backend
        self._do_ocr = do_ocr
        self._artifacts_path = artifacts_path

    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        converter = self._create_converter()
        result = converter.convert(str(raw_document.path))
        docling_document = getattr(result, "document", result)

        markdown = _call_export(docling_document, "export_to_markdown")
        text = _call_export(docling_document, "export_to_text")
        docling_data = _export_docling_data(docling_document)

        if not markdown:
            markdown = text or ""
        if not markdown:
            raise ValueError(f"Docling returned no text for {raw_document.path}")

        return self._normalizer.from_docling(
            raw_document=raw_document,
            markdown=markdown,
            text=text,
            docling_data=docling_data,
            parser_name=self.parser_name,
            parser_version=_docling_version(),
        )

    def _create_converter(self) -> Any:
        if self._converter_factory is not None:
            return self._converter_factory()

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ParserDependencyError(
                "Docling is not installed. Install it with `pip install docling` "
                "or `pip install .[parsing]` before running real document parsing."
            ) from exc

        if (
            self._pdf_backend is None
            and self._do_ocr is None
            and self._artifacts_path is None
        ):
            return DocumentConverter()

        try:
            from docling.backend.docling_parse_backend import (
                DoclingParseDocumentBackend,
                ThreadedDoclingParseDocumentBackend,
            )
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.backend_options import (
                ThreadedDoclingParseBackendOptions,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
        except ImportError as exc:
            raise ParserDependencyError(
                "This Docling installation does not expose configurable PDF backend "
                "options. Upgrade Docling or install the parsing extra again."
            ) from exc

        pdf_backend = self._pdf_backend or "docling_parse"
        backend_options = None
        backend: type[Any]
        if pdf_backend == "docling_parse":
            backend = DoclingParseDocumentBackend
        elif pdf_backend == "threaded_docling_parse":
            backend = ThreadedDoclingParseDocumentBackend
            backend_options = ThreadedDoclingParseBackendOptions()
        elif pdf_backend == "pypdfium2":
            backend = PyPdfiumDocumentBackend
        else:
            supported = ", ".join(sorted(SUPPORTED_PDF_BACKENDS))
            raise ValueError(
                f"Unsupported Docling PDF backend {pdf_backend!r}. "
                f"Use one of: {supported}."
            )

        pipeline_options = PdfPipelineOptions()
        if self._do_ocr is not None:
            pipeline_options.do_ocr = self._do_ocr
        if self._artifacts_path is not None:
            pipeline_options.artifacts_path = self._artifacts_path

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=backend,
                    backend_options=backend_options,
                )
            }
        )


def _call_export(docling_document: Any, method_name: str) -> str | None:
    method = getattr(docling_document, method_name, None)
    if not callable(method):
        return None
    value = method()
    return value if isinstance(value, str) else str(value)


def _export_docling_data(docling_document: Any) -> Mapping[str, Any] | None:
    for method_name in ("export_to_dict", "export_to_json", "model_dump"):
        method = getattr(docling_document, method_name, None)
        if not callable(method):
            continue
        exported = parse_docling_export(method())
        if exported is not None:
            return exported
    return None


def _docling_version() -> str | None:
    try:
        return importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        return None
