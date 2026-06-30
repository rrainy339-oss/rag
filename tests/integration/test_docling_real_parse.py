from __future__ import annotations

import importlib.util
import unittest

from app.connectors.local_file import LocalFileConnector
from app.parsing.docling_parser import DoclingParser


@unittest.skipUnless(
    importlib.util.find_spec("docling") is not None,
    "Docling is not installed; install `.[parsing]` to run this integration test.",
)
class RealDoclingParseTest(unittest.TestCase):
    def test_docling_parses_sample_markdown_fixture(self) -> None:
        raw = next(LocalFileConnector("tests/fixtures/sample.md").iter_documents())
        parsed = DoclingParser().parse(raw)

        self.assertIn("Employee Handbook", parsed.text)
        self.assertEqual(parsed.metadata.parser_name, "docling")
        self.assertGreater(parsed.stats["element_count"], 0)


if __name__ == "__main__":
    unittest.main()

