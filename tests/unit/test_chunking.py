from __future__ import annotations

import unittest

from app.chunking.config import ChunkingConfig
from app.chunking.pipeline import ChunkingPipeline
from app.chunking.schemas import ChunkType
from app.domain.schemas import (
    AccessControl,
    CitationSpan,
    DocumentMetadata,
    ParsedDocument,
    ParsedElement,
    ParsedElementType,
    TableData,
)


class ChunkingPipelineTest(unittest.TestCase):
    def test_structure_aware_parent_child_and_table_chunks(self) -> None:
        document = _sample_document()
        result = ChunkingPipeline(_test_config()).chunk_document(document)

        parents = [chunk for chunk in result.chunks if chunk.chunk_type == ChunkType.PARENT]
        children = [chunk for chunk in result.chunks if chunk.chunk_type == ChunkType.CHILD]
        tables = [chunk for chunk in result.chunks if chunk.chunk_type == ChunkType.TABLE]

        self.assertGreaterEqual(len(parents), 2)
        self.assertEqual(len(tables), 1)
        self.assertTrue(children)
        self.assertEqual(tables[0].section_path, ["Employee Handbook", "Benefits"])
        self.assertIn("| Benefit | Eligibility |", tables[0].markdown)
        self.assertNotIn("Contact HR", tables[0].markdown)

        parent_ids = {chunk.chunk_id for chunk in parents}
        self.assertTrue(all(chunk.parent_chunk_id in parent_ids for chunk in children))
        self.assertEqual(result.stats["table_count"], 1)

    def test_chunks_inherit_metadata_and_have_stable_hashes(self) -> None:
        document = _sample_document()
        pipeline = ChunkingPipeline(_test_config())

        first = pipeline.chunk_document(document)
        second = pipeline.chunk_document(document)

        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunks],
            [chunk.chunk_id for chunk in second.chunks],
        )
        self.assertEqual(
            [chunk.content_hash for chunk in first.chunks],
            [chunk.content_hash for chunk in second.chunks],
        )

        for chunk in first.chunks:
            self.assertEqual(chunk.access.tenant_id, "tenant-a")
            self.assertTrue(chunk.element_ids)
            self.assertTrue(chunk.citations)
            self.assertIn("Document: sample.md", chunk.contextual_text)

        retrieval_chunks = [
            chunk
            for chunk in first.chunks
            if chunk.chunk_type in {ChunkType.CHILD, ChunkType.TABLE}
        ]
        self.assertTrue(any(chunk.neighbor_chunk_ids for chunk in retrieval_chunks))


def _test_config() -> ChunkingConfig:
    return ChunkingConfig(
        child_target_tokens=80,
        child_max_tokens=100,
        parent_target_tokens=200,
        parent_max_tokens=300,
    )


def _sample_document() -> ParsedDocument:
    source_uri = "file:///sample.md"
    access = AccessControl(tenant_id="tenant-a", allowed_group_ids=["hr"])

    elements = [
        ParsedElement(
            element_id="e-title-1",
            type=ParsedElementType.TITLE,
            text="Employee Handbook",
            markdown="# Employee Handbook",
            order_index=0,
            level=1,
            citations=[CitationSpan(source_uri=source_uri, text="Employee Handbook")],
        ),
        ParsedElement(
            element_id="e-policy",
            type=ParsedElementType.PARAGRAPH,
            text="Remote work is available for eligible employees after manager approval.",
            markdown="Remote work is available for eligible employees after manager approval.",
            order_index=1,
            citations=[CitationSpan(source_uri=source_uri, text="Remote work is available")],
        ),
        ParsedElement(
            element_id="e-title-2",
            type=ParsedElementType.TITLE,
            text="Benefits",
            markdown="## Benefits",
            order_index=2,
            level=2,
            citations=[CitationSpan(source_uri=source_uri, text="Benefits")],
        ),
        ParsedElement(
            element_id="e-table",
            type=ParsedElementType.TABLE,
            text=(
                "| Benefit | Eligibility |\n"
                "| --- | --- |\n"
                "| Health insurance | Full-time employees |"
            ),
            markdown=(
                "| Benefit | Eligibility |\n"
                "| --- | --- |\n"
                "| Health insurance | Full-time employees |"
            ),
            order_index=3,
            table=TableData(
                markdown=(
                    "| Benefit | Eligibility |\n"
                    "| --- | --- |\n"
                    "| Health insurance | Full-time employees |"
                ),
                rows=[
                    ["Benefit", "Eligibility"],
                    ["Health insurance", "Full-time employees"],
                ],
            ),
            citations=[CitationSpan(source_uri=source_uri, text="Health insurance")],
        ),
        ParsedElement(
            element_id="e-contact",
            type=ParsedElementType.PARAGRAPH,
            text="Contact HR for policy exceptions.",
            markdown="Contact HR for policy exceptions.",
            order_index=4,
            citations=[CitationSpan(source_uri=source_uri, text="Contact HR")],
        ),
    ]

    markdown = "\n\n".join(element.markdown or element.text for element in elements)
    return ParsedDocument(
        document_id="doc-sample",
        metadata=DocumentMetadata(
            source_uri=source_uri,
            filename="sample.md",
            file_type="md",
            mime_type="text/markdown",
            content_hash="doc-content-hash",
            size_bytes=123,
            parser_name="test",
            access=access,
        ),
        text="\n\n".join(element.text for element in elements),
        markdown=markdown,
        elements=elements,
    )


if __name__ == "__main__":
    unittest.main()

