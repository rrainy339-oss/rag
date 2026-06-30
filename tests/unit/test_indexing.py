from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from app.indexing.config import IndexingConfig
from app.indexing.embeddings.bge import _lexical_weights_to_sparse_vector
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.manifest import InMemoryManifestStore
from app.indexing.schemas import IndexBackend, IndexRecord, IndexingStatus
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore
from tests.unit.hybrid_fakes import CapturingHybridVectorStore, FakeHybridEmbeddingProvider
from tests.unit.test_retrieval import _load_sample_chunks


class IndexingPipelineTest(unittest.TestCase):
    def test_bge_lexical_weights_convert_to_sorted_sparse_vector(self) -> None:
        sparse = _lexical_weights_to_sparse_vector(
            {"42": 0.1, "7": 0.7, "9": 0.2},
            top_n=2,
        )

        self.assertEqual(sparse.indices, [7, 9])
        self.assertEqual(sparse.values, [0.7, 0.2])

    def test_indexes_chunks_with_bge_hybrid_vectors_and_acl_payload(self) -> None:
        chunking_result = _load_sample_chunks()
        vector_store = CapturingHybridVectorStore()
        indexer = EnterpriseIndexer(
            embedding_provider=FakeHybridEmbeddingProvider(),
            vector_store=vector_store,
            config=IndexingConfig(
                embedding_model="BAAI/bge-m3-test",
                embedding_dimension=8,
            ),
            manifest_store=InMemoryManifestStore(),
        )

        indexing_result = indexer.index(chunking_result)

        self.assertEqual(indexing_result.status, IndexingStatus.INDEXED)
        self.assertEqual(indexing_result.indexed_count, 3)
        self.assertEqual(vector_store.count(), 3)
        self.assertEqual(indexing_result.manifest.backend, IndexBackend.QDRANT_HYBRID)

        record = indexing_result.records[0]
        self.assertEqual(record.embedding_dimension, 8)
        self.assertEqual(len(record.vector), 8)
        self.assertTrue(record.sparse_vector_indices)
        self.assertTrue(record.sparse_vector_values)
        self.assertEqual(record.sparse_terms, {})
        self.assertIn(record.chunk_type.value, {"child", "table"})
        self.assertIn("document_id", record.metadata)
        self.assertIn("section_path", record.metadata)
        self.assertIn("allowed_group_ids", record.metadata)
        self.assertIn("principal_ids", record.metadata)
        self.assertIn("group:tenant-a:hr", record.metadata["principal_ids"])
        self.assertEqual(record.metadata["classification_level"], 0)

    def test_reindex_is_idempotent_with_matching_hybrid_manifest(self) -> None:
        chunking_result = _load_sample_chunks()
        vector_store = CapturingHybridVectorStore()
        manifest_store = InMemoryManifestStore()
        indexer = EnterpriseIndexer(
            embedding_provider=FakeHybridEmbeddingProvider(),
            vector_store=vector_store,
            config=IndexingConfig(
                embedding_model="BAAI/bge-m3-test",
                embedding_dimension=8,
            ),
            manifest_store=manifest_store,
        )

        first = indexer.index(chunking_result)
        second = indexer.index(chunking_result)

        self.assertEqual(first.status, IndexingStatus.INDEXED)
        self.assertEqual(second.status, IndexingStatus.SKIPPED)
        self.assertEqual(second.skipped_count, 3)
        self.assertEqual(vector_store.count(), 3)

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
        chunk_type="child",
        text=text,
        contextual_text=text,
        vector=vector,
        sparse_vector_indices=sparse_indices,
        sparse_vector_values=sparse_values,
        content_hash=f"{chunk_id}-hash",
        embedding_model="BAAI/bge-m3-test",
        embedding_dimension=len(vector),
        index_version="test",
    )


if __name__ == "__main__":
    unittest.main()
