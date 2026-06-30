from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from app.chunking.schemas import Chunk, ChunkType, ChunkingResult
from app.domain.schemas import AccessControl, CitationSpan
from app.indexing.config import IndexingConfig
from app.indexing.embeddings.bge import _lexical_weights_to_sparse_vector
from app.indexing.embeddings.base import HybridEmbedding, SparseEmbeddingVector
from app.indexing.embeddings.hashing import HashingEmbeddingProvider
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.manifest import InMemoryManifestStore
from app.indexing.schemas import IndexBackend, IndexRecord, IndexingStatus
from app.indexing.sparse.memory_bm25 import MemoryBM25Store
from app.indexing.vectorstores.memory import MemoryVectorStore
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore, QdrantVectorStore


class IndexingPipelineTest(unittest.TestCase):
    def test_bge_lexical_weights_convert_to_sorted_sparse_vector(self) -> None:
        sparse = _lexical_weights_to_sparse_vector(
            {"42": 0.1, "7": 0.7, "9": 0.2},
            top_n=2,
        )

        self.assertEqual(sparse.indices, [7, 9])
        self.assertEqual(sparse.values, [0.7, 0.2])

    def test_indexes_retrieval_chunks_with_metadata_and_acl(self) -> None:
        result = _load_sample_chunks()
        vector_store = MemoryVectorStore()
        sparse_store = MemoryBM25Store()
        indexer = EnterpriseIndexer(
            config=IndexingConfig(embedding_dimension=64),
            embedding_provider=HashingEmbeddingProvider(64),
            vector_store=vector_store,
            sparse_store=sparse_store,
            manifest_store=InMemoryManifestStore(),
        )

        indexing_result = indexer.index(result)

        self.assertEqual(indexing_result.status, IndexingStatus.INDEXED)
        self.assertEqual(indexing_result.indexed_count, 3)
        self.assertEqual(vector_store.count(), 3)
        self.assertEqual(sparse_store.count(), 3)

        record = indexing_result.records[0]
        self.assertEqual(record.embedding_dimension, 64)
        self.assertEqual(len(record.vector), 64)
        self.assertIn(record.chunk_type.value, {"child", "table"})
        self.assertIn("document_id", record.metadata)
        self.assertIn("section_path", record.metadata)
        self.assertIn("allowed_group_ids", record.metadata)
        self.assertIn("principal_ids", record.metadata)
        self.assertIn("group:tenant-a:hr", record.metadata["principal_ids"])
        self.assertEqual(record.metadata["classification_level"], 0)
        self.assertTrue(record.sparse_terms)

    def test_hybrid_indexing_stores_bge_sparse_vectors(self) -> None:
        result = _load_sample_chunks()
        indexer = EnterpriseIndexer(
            config=IndexingConfig(
                backend=IndexBackend.QDRANT_HYBRID,
                embedding_dimension=8,
            ),
            embedding_provider=_FakeHybridEmbeddingProvider(),
            vector_store=MemoryVectorStore(),
            sparse_store=None,
            manifest_store=InMemoryManifestStore(),
        )

        indexing_result = indexer.index(result)

        self.assertEqual(indexing_result.status, IndexingStatus.INDEXED)
        self.assertEqual(indexing_result.indexed_count, 3)
        self.assertTrue(indexing_result.records[0].sparse_vector_indices)
        self.assertTrue(indexing_result.records[0].sparse_vector_values)
        self.assertEqual(
            indexing_result.manifest.metadata["sparse_vector_name"],
            "sparse",
        )

    def test_reindex_is_idempotent_with_matching_manifest(self) -> None:
        result = _load_sample_chunks()
        vector_store = MemoryVectorStore()
        sparse_store = MemoryBM25Store()
        manifest_store = InMemoryManifestStore()
        indexer = EnterpriseIndexer(
            config=IndexingConfig(embedding_dimension=32),
            embedding_provider=HashingEmbeddingProvider(32),
            vector_store=vector_store,
            sparse_store=sparse_store,
            manifest_store=manifest_store,
        )

        first = indexer.index(result)
        second = indexer.index(result)

        self.assertEqual(first.status, IndexingStatus.INDEXED)
        self.assertEqual(second.status, IndexingStatus.SKIPPED)
        self.assertEqual(second.skipped_count, 3)
        self.assertEqual(vector_store.count(), 3)
        self.assertEqual(sparse_store.count(), 3)

    def test_memory_dense_and_bm25_search_work(self) -> None:
        result = _load_sample_chunks()
        vector_store = MemoryVectorStore()
        sparse_store = MemoryBM25Store()
        embedding_provider = HashingEmbeddingProvider(64)
        indexer = EnterpriseIndexer(
            config=IndexingConfig(embedding_dimension=64),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_store=sparse_store,
            manifest_store=InMemoryManifestStore(),
        )
        indexer.index(result)

        vector_results = vector_store.search(
            embedding_provider.embed_query("health insurance benefit"), top_k=2
        )
        sparse_results = sparse_store.search("health insurance", top_k=2)

        self.assertTrue(vector_results)
        self.assertTrue(sparse_results)
        self.assertIn("Health insurance", sparse_results[0].record.text)

    @unittest.skipIf(
        importlib.util.find_spec("qdrant_client") is None,
        "qdrant-client is not installed",
    )
    def test_qdrant_vector_store_persists_and_searches_records(self) -> None:
        result = _load_sample_chunks()
        embedding_provider = HashingEmbeddingProvider(32)

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "qdrant"
            vector_store = QdrantVectorStore(
                path=storage_path,
                collection_name="test_chunks",
                vector_size=32,
            )
            try:
                indexer = EnterpriseIndexer(
                    config=IndexingConfig(embedding_dimension=32),
                    embedding_provider=embedding_provider,
                    vector_store=vector_store,
                    sparse_store=MemoryBM25Store(),
                    manifest_store=InMemoryManifestStore(),
                )

                indexing_result = indexer.index(result)
                self.assertEqual(indexing_result.indexed_count, 3)
                self.assertEqual(vector_store.count(), 3)
            finally:
                vector_store.close()

            reopened_store = QdrantVectorStore(
                path=storage_path,
                collection_name="test_chunks",
                vector_size=32,
            )
            try:
                self.assertEqual(reopened_store.count(), 3)

                search_results = reopened_store.search(
                    embedding_provider.embed_query("health insurance"),
                    top_k=3,
                )
                self.assertEqual(len(search_results), 3)
                self.assertTrue(
                    any("Health insurance" in item.record.contextual_text for item in search_results)
                )
                self.assertEqual(reopened_store.delete_by_document("doc-sample"), 3)
                self.assertEqual(reopened_store.count(), 0)
            finally:
                reopened_store.close()

    @unittest.skipIf(
        importlib.util.find_spec("qdrant_client") is None,
        "qdrant-client is not installed",
    )
    def test_qdrant_hybrid_vector_store_uses_dense_and_sparse_vectors(self) -> None:
        records = [
            _index_record(
                record_id="idx-hybrid-alpha",
                chunk_id="hybrid-alpha",
                text="Alpha finance policy",
                vector=[1.0, 0.0, 0.0],
                sparse_indices=[11, 13],
                sparse_values=[1.0, 0.25],
            ),
            _index_record(
                record_id="idx-hybrid-beta",
                chunk_id="hybrid-beta",
                text="Beta security policy",
                vector=[0.0, 1.0, 0.0],
                sparse_indices=[21],
                sparse_values=[1.0],
            ),
        ]
        records[0].metadata.update(
            {"principal_ids": ["group:tenant-a:hr"], "classification_level": 1}
        )
        records[1].metadata.update(
            {"principal_ids": ["group:tenant-a:engineering"], "classification_level": 2}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = QdrantHybridVectorStore(
                path=Path(tmpdir) / "qdrant",
                collection_name="test_hybrid_chunks",
                vector_size=3,
            )
            try:
                vector_store.upsert(records)
                self.assertEqual(vector_store.count(), 2)

                results = vector_store.hybrid_search(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_indices=[11],
                    sparse_values=[1.0],
                    top_k=2,
                    dense_top_k=2,
                    sparse_top_k=2,
                )

                self.assertEqual(results[0].record.record_id, "idx-hybrid-alpha")
                self.assertEqual(results[0].record.sparse_vector_indices, [11, 13])
                self.assertTrue(results[0].score > results[1].score)

                filtered = vector_store.hybrid_search(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_indices=[11],
                    sparse_values=[1.0],
                    top_k=2,
                    dense_top_k=2,
                    sparse_top_k=2,
                    filters={
                        "principal_ids_any": ["group:tenant-a:hr"],
                        "classification_level_lte": 1,
                    },
                )
                self.assertEqual(
                    [item.record.record_id for item in filtered],
                    ["idx-hybrid-alpha"],
                )
            finally:
                vector_store.close()


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
            contextual_text="Document: sample.md\nSection: Employee Handbook\nContent:\nRemote work is available.",
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
            contextual_text="Document: sample.md\nSection: Employee Handbook\nContent:\nRemote work is available.",
            chunk_index=1,
            section_path=["Employee Handbook"],
            element_ids=["e-title", "e-remote"],
            citations=[CitationSpan(source_uri=source_uri, text="Remote work")],
            access=access,
            token_count=8,
            content_hash="child-remote-hash",
        ),
        Chunk(
            chunk_id="child-benefits",
            document_id="doc-sample",
            parent_chunk_id="parent-benefits",
            chunk_type=ChunkType.CHILD,
            text="Benefits\n\nContact HR for policy exceptions.",
            markdown="## Benefits\n\nContact HR for policy exceptions.",
            contextual_text="Document: sample.md\nSection: Employee Handbook > Benefits\nContent:\nContact HR for policy exceptions.",
            chunk_index=2,
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
            text="| Benefit | Eligibility |\n| --- | --- |\n| Health insurance | Full-time employees |",
            markdown="| Benefit | Eligibility |\n| --- | --- |\n| Health insurance | Full-time employees |",
            contextual_text=(
                "Document: sample.md\nSection: Employee Handbook > Benefits\nContent:\n"
                "| Benefit | Eligibility |\n| Health insurance | Full-time employees |"
            ),
            chunk_index=3,
            section_path=["Employee Handbook", "Benefits"],
            element_ids=["e-table"],
            citations=[CitationSpan(source_uri=source_uri, text="Health insurance")],
            access=access,
            token_count=8,
            content_hash="table-benefits-hash",
        ),
    ]
    return ChunkingResult(document_id="doc-sample", chunks=chunks)


def _index_record(
    *,
    record_id: str,
    chunk_id: str,
    text: str,
    vector: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
) -> IndexRecord:
    return IndexRecord(
        record_id=record_id,
        chunk_id=chunk_id,
        document_id="doc-hybrid",
        chunk_type=ChunkType.CHILD,
        text=text,
        contextual_text=text,
        vector=vector,
        sparse_vector_indices=sparse_indices,
        sparse_vector_values=sparse_values,
        content_hash=f"{chunk_id}-hash",
        embedding_model="fake-hybrid",
        embedding_dimension=len(vector),
        index_version="test",
    )


class _FakeHybridEmbeddingProvider:
    model_name = "fake-hybrid"
    dimension = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents_hybrid(self, texts: list[str]) -> list[HybridEmbedding]:
        embeddings: list[HybridEmbedding] = []
        for index, _ in enumerate(texts, start=1):
            embeddings.append(
                HybridEmbedding(
                    dense=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    sparse=SparseEmbeddingVector(
                        indices=[index],
                        values=[1.0],
                    ),
                )
            )
        return embeddings


if __name__ == "__main__":
    unittest.main()
