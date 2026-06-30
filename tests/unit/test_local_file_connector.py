from __future__ import annotations

import unittest
from pathlib import Path

from app.connectors.local_file import LocalFileConnector
from app.domain.schemas import AccessControl


class LocalFileConnectorTest(unittest.TestCase):
    def test_discovers_supported_files_with_metadata(self) -> None:
        fixture = Path("tests/fixtures/sample.md").resolve()
        documents = list(LocalFileConnector(fixture).iter_documents())

        self.assertEqual(len(documents), 1)
        raw = documents[0]
        self.assertEqual(raw.filename, "sample.md")
        self.assertEqual(raw.file_type, "md")
        self.assertGreater(raw.size_bytes, 0)
        self.assertEqual(len(raw.content_hash), 64)
        self.assertEqual(raw.metadata["connector"], "local_file")

    def test_applies_file_specific_access_rules(self) -> None:
        fixture = Path("tests/fixtures/sample.md").resolve()
        access = AccessControl(
            tenant_id="tenant-a",
            allowed_group_ids=["hr"],
            classification="internal",
        )

        documents = list(
            LocalFileConnector(
                fixture,
                access_by_filename={"sample.md": access},
            ).iter_documents()
        )

        self.assertEqual(documents[0].access.tenant_id, "tenant-a")
        self.assertEqual(documents[0].access.allowed_group_ids, ["hr"])
        self.assertEqual(documents[0].access.classification, "internal")


if __name__ == "__main__":
    unittest.main()
