from __future__ import annotations

import unittest

from app.chunking.schemas import Chunk, ChunkType, ChunkingResult
from app.domain.schemas import AccessControl, CitationSpan
from app.indexing.config import IndexingConfig
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.manifest import InMemoryManifestStore
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.config import RetrievalConfig
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.parent_expander import ParentExpander
from app.retrieval.qdrant_hybrid import (
    QdrantHybridRetrievalPipeline,
    QdrantHybridRetriever,
)
from app.retrieval.schemas import QueryType, RetrievalQuery
from tests.unit.hybrid_fakes import CapturingHybridVectorStore, FakeHybridEmbeddingProvider


class RetrievalPipelineTest(unittest.TestCase):
    def test_hybrid_retrieval_prioritizes_table_context_for_table_queries(self) -> None:
        pipeline = _build_pipeline(_load_sample_chunks())

        response = pipeline.retrieve(
            RetrievalQuery(
                query="table health insurance",
                tenant_id="tenant-a",
                group_ids=["hr"],
            )
        )

        self.assertEqual(response.query_type, QueryType.TABLE)
        self.assertTrue(response.contexts)
        self.assertEqual(response.contexts[0].chunk_type, ChunkType.TABLE)
        self.assertIn("Health insurance", response.contexts[0].text)
        self.assertEqual(response.stats["retriever"], "qdrant_hybrid")
        self.assertEqual(response.stats["reranker"], "noop-reranker")

    def test_child_hit_expands_to_parent_context(self) -> None:
        pipeline = _build_pipeline(_load_sample_chunks())

        response = pipeline.retrieve(
            RetrievalQuery(
                query="remote work",
                tenant_id="tenant-a",
                group_ids=["hr"],
            )
        )

        self.assertTrue(response.contexts)
        self.assertEqual(response.contexts[0].context_id, "parent-remote")
        self.assertEqual(response.contexts[0].source_chunk_id, "child-remote")
        self.assertIn("Remote work is available", response.contexts[0].text)
        self.assertTrue(response.contexts[0].metadata["expanded_from_parent"])

    def test_acl_filters_candidates_before_context_building(self) -> None:
        pipeline = _build_pipeline(_load_sample_chunks())

        denied = pipeline.retrieve(
            RetrievalQuery(
                query="health insurance",
                tenant_id="tenant-a",
                group_ids=["engineering"],
            )
        )
        allowed = pipeline.retrieve(
            RetrievalQuery(
                query="health insurance",
                tenant_id="tenant-a",
                group_ids=["hr"],
            )
        )

        self.assertEqual(denied.contexts, [])
        self.assertEqual(denied.candidates, [])
        self.assertTrue(allowed.contexts)


def _build_pipeline(chunking_result: ChunkingResult) -> QdrantHybridRetrievalPipeline:
    embedding_provider = FakeHybridEmbeddingProvider()
    vector_store = CapturingHybridVectorStore()
    indexer = EnterpriseIndexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        config=IndexingConfig(
            embedding_model=embedding_provider.model_name,
            embedding_dimension=embedding_provider.dimension,
        ),
        manifest_store=InMemoryManifestStore(),
    )
    indexer.index(chunking_result)

    chunk_store = ChunkStore.from_chunking_result(chunking_result)
    return QdrantHybridRetrievalPipeline(
        hybrid_retriever=QdrantHybridRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        ),
        config=RetrievalConfig(final_top_k=3),
        context_builder=ContextBuilder(ParentExpander(chunk_store)),
    )


def _load_sample_chunks() -> ChunkingResult:
    source_uri = "file:///sample.md"
    access = AccessControl(tenant_id="tenant-a", allowed_group_ids=["hr"])
    chunks = [
        Chunk(
            chunk_id="parent-remote",
            document_id="doc-sample",
            chunk_type=ChunkType.PARENT,
            text="Employee Handbook\n\nRemote work is available.",
            markdown="# Employee Handbook\n\nRemote work is available.",
            contextual_text=(
                "Document: sample.md\n"
                "Section: Employee Handbook\n"
                "Content:\nRemote work is available."
            ),
            chunk_index=0,
            section_path=["Employee Handbook"],
            element_ids=["e-title", "e-remote"],
            citations=[CitationSpan(source_uri=source_uri, text="Remote work")],
            access=access,
            token_count=8,
            content_hash="parent-remote-hash",
        ),
        Chunk(
            chunk_id="child-remote",
            document_id="doc-sample",
            parent_chunk_id="parent-remote",
            chunk_type=ChunkType.CHILD,
            text="Employee Handbook\n\nRemote work is available.",
            markdown="# Employee Handbook\n\nRemote work is available.",
            contextual_text=(
                "Document: sample.md\n"
                "Section: Employee Handbook\n"
                "Content:\nRemote work is available."
            ),
            chunk_index=1,
            section_path=["Employee Handbook"],
            element_ids=["e-title", "e-remote"],
            citations=[CitationSpan(source_uri=source_uri, text="Remote work")],
            access=access,
            token_count=8,
            content_hash="child-remote-hash",
        ),
        Chunk(
            chunk_id="parent-benefits",
            document_id="doc-sample",
            chunk_type=ChunkType.PARENT,
            text=(
                "Benefits\n\n"
                "Health insurance is offered to full-time employees. "
                "Contact HR for policy exceptions."
            ),
            markdown=(
                "## Benefits\n\n"
                "Health insurance is offered to full-time employees.\n\n"
                "Contact HR for policy exceptions."
            ),
            contextual_text=(
                "Document: sample.md\n"
                "Section: Employee Handbook > Benefits\n"
                "Content:\nHealth insurance is offered to full-time employees. "
                "Contact HR for policy exceptions."
            ),
            chunk_index=2,
            section_path=["Employee Handbook", "Benefits"],
            element_ids=["e-benefits", "e-contact"],
            citations=[CitationSpan(source_uri=source_uri, text="Benefits")],
            access=access,
            token_count=14,
            content_hash="parent-benefits-hash",
        ),
        Chunk(
            chunk_id="child-benefits",
            document_id="doc-sample",
            parent_chunk_id="parent-benefits",
            chunk_type=ChunkType.CHILD,
            text="Benefits\n\nContact HR for policy exceptions.",
            markdown="## Benefits\n\nContact HR for policy exceptions.",
            contextual_text=(
                "Document: sample.md\n"
                "Section: Employee Handbook > Benefits\n"
                "Content:\nContact HR for policy exceptions."
            ),
            chunk_index=3,
            section_path=["Employee Handbook", "Benefits"],
            element_ids=["e-benefits", "e-contact"],
            citations=[CitationSpan(source_uri=source_uri, text="Contact HR")],
            access=access,
            token_count=7,
            content_hash="child-benefits-hash",
        ),
        Chunk(
            chunk_id="table-benefits",
            document_id="doc-sample",
            parent_chunk_id="parent-benefits",
            chunk_type=ChunkType.TABLE,
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
            contextual_text=(
                "Document: sample.md\n"
                "Section: Employee Handbook > Benefits\n"
                "Content:\n"
                "| Benefit | Eligibility |\n"
                "| Health insurance | Full-time employees |"
            ),
            chunk_index=4,
            section_path=["Employee Handbook", "Benefits"],
            element_ids=["e-table"],
            citations=[CitationSpan(source_uri=source_uri, text="Health insurance")],
            access=access,
            token_count=8,
            content_hash="table-benefits-hash",
        ),
    ]
    return ChunkingResult(document_id="doc-sample", chunks=chunks)


if __name__ == "__main__":
    unittest.main()
