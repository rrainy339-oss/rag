from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.connectors.local_file import LocalFileConnector
from app.domain.schemas import ParsedElementType
from app.parsing.base import ParserDependencyError
from app.parsing.docling_parser import DoclingParser


class FakeDoclingDocument:
    def export_to_markdown(self) -> str:
        return Path("tests/fixtures/sample.md").read_text(encoding="utf-8")

    def export_to_text(self) -> str:
        return "Employee Handbook\nRemote work is available.\nBenefits"

    def export_to_dict(self) -> dict:
        return {
            "body": {
                "children": [
                    {"label": "text", "text": "Employee Handbook"},
                    {"label": "picture", "caption": "Sample figure"},
                ]
            }
        }


class FakeConverter:
    def convert(self, path: str) -> SimpleNamespace:
        return SimpleNamespace(document=FakeDoclingDocument(), source=path)


class DoclingParserTest(unittest.TestCase):
    def test_normalizes_docling_output(self) -> None:
        raw = next(LocalFileConnector("tests/fixtures/sample.md").iter_documents())
        parser = DoclingParser(converter_factory=lambda: FakeConverter())

        parsed = parser.parse(raw)

        self.assertEqual(parsed.metadata.parser_name, "docling")
        self.assertEqual(parsed.metadata.filename, "sample.md")
        self.assertIn("Employee Handbook", parsed.markdown)
        self.assertEqual(parsed.elements[0].type, ParsedElementType.TITLE)
        self.assertEqual(parsed.elements[0].level, 1)
        self.assertEqual(len(parsed.tables), 1)
        self.assertEqual(parsed.tables[0].rows[0], ["Benefit", "Eligibility"])
        self.assertEqual(len(parsed.images), 1)
        self.assertGreaterEqual(parsed.stats["element_count"], 4)

    def test_missing_docling_dependency_has_actionable_error(self) -> None:
        if importlib.util.find_spec("docling") is not None:
            self.skipTest("Docling is installed in this environment.")

        raw = next(LocalFileConnector("tests/fixtures/sample.md").iter_documents())
        parser = DoclingParser()

        with self.assertRaises(ParserDependencyError) as context:
            parser.parse(raw)

        self.assertIn("pip install docling", str(context.exception))

    def test_rejects_unsupported_pdf_backend(self) -> None:
        with self.assertRaises(ValueError) as context:
            DoclingParser(pdf_backend="unsupported")

        self.assertIn("Unsupported Docling PDF backend", str(context.exception))


if __name__ == "__main__":
    unittest.main()
