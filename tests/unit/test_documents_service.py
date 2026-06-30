from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.api.settings import APISettings
from app.documents.schemas import DocumentRecord, DocumentStatus
from app.documents.service import DocumentService
from app.indexing.embeddings.base import HybridEmbedding, SparseEmbeddingVector
from app.indexing.schemas import IndexBackend, IndexingResult
from tests.unit.test_retrieval import _load_sample_chunks


class DocumentServiceTest(unittest.TestCase):
    def test_qdrant_hybrid_document_indexing_writes_dense_and_sparse_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = _FakeHybridProvider()
            vector_store = _CapturingVectorStore()
            captured: dict[str, object] = {}

            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]

            def build_vector_store(*, backend: IndexBackend, embedding_dimension: int) -> object:
                captured["backend"] = backend
                captured["embedding_dimension"] = embedding_dimension
                return vector_store

            service._build_vector_store = build_vector_store  # type: ignore[method-assign]

            record = _record(root)
            chunking_result = _sample_chunks(record.document_id)
            indexing_result, index_path = service._index_document(record, chunking_result)
            index_path_exists = index_path.exists()

        self.assertEqual(captured["backend"], IndexBackend.QDRANT_HYBRID)
        self.assertEqual(captured["embedding_dimension"], provider.dimension)
        self.assertTrue(provider.closed)
        self.assertTrue(vector_store.closed)
        self.assertTrue(index_path_exists)
        self.assertEqual(indexing_result.manifest.backend, IndexBackend.QDRANT_HYBRID)
        self.assertEqual(indexing_result.manifest.metadata["dense_vector_name"], "dense")
        self.assertEqual(indexing_result.manifest.metadata["sparse_vector_name"], "sparse")
        self.assertTrue(vector_store.records)
        self.assertTrue(vector_store.records[0].vector)
        self.assertTrue(vector_store.records[0].sparse_vector_indices)
        self.assertTrue(vector_store.records[0].sparse_vector_values)

    def test_rebuild_collection_preserves_hybrid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            service._build_embedding_provider = lambda: _FakeHybridProvider()  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, backend, embedding_dimension: _CapturingVectorStore()
            )

            record = _record(root)
            service.registry.create(record)
            chunking_result = _sample_chunks(record.document_id)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
            indexing_result, index_path = service._index_document(record, chunking_result)
            service.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                index_path=index_path,
                chunk_count=len(chunking_result.chunks),
                indexed_count=indexing_result.indexed_count,
                status=DocumentStatus.READY,
            )

            service.rebuild_collection()
            collection = IndexingResult.model_validate_json(
                settings.documents_collection_index_path.read_text(encoding="utf-8")
            )

        self.assertEqual(collection.manifest.backend, IndexBackend.QDRANT_HYBRID)
        self.assertEqual(collection.manifest.embedding_model, "BAAI/bge-m3-test")
        self.assertEqual(collection.manifest.metadata["dense_vector_name"], "dense")
        self.assertEqual(collection.manifest.metadata["sparse_vector_name"], "sparse")
        self.assertEqual(collection.indexed_count, indexing_result.indexed_count)


class _FakeHybridProvider:
    model_name = "BAAI/bge-m3-test"
    dimension = 8

    def __init__(self) -> None:
        self.closed = False

    def embed_documents_hybrid(self, texts: list[str]) -> list[HybridEmbedding]:
        return [
            HybridEmbedding(
                dense=[1.0, 0.0, 0.0, float(index + 1), 0.0, 0.0, 0.0, 0.0],
                sparse=SparseEmbeddingVector(indices=[index + 10], values=[0.5]),
            )
            for index, _ in enumerate(texts)
        ]

    def close(self) -> None:
        self.closed = True


class _CapturingVectorStore:
    def __init__(self) -> None:
        self.records = []
        self.closed = False

    def delete_by_document(self, document_id: str) -> int:
        self.records = [
            record for record in self.records if record.document_id != document_id
        ]
        return 0

    def upsert(self, records: list[object]) -> None:
        self.records.extend(records)

    def count(self) -> int:
        return len(self.records)

    def close(self) -> None:
        self.closed = True


def _document_settings(root: Path) -> APISettings:
    return APISettings(
        documents_backend="qdrant_hybrid",
        documents_embedding_provider="bge-m3",
        documents_db_path=root / "documents.sqlite3",
        documents_storage_dir=root / "documents",
        documents_collection_index_path=root / "collection" / "index.json",
        documents_collection_chunks_path=root / "collection" / "chunks.json",
        qdrant_collection="rag_chunks_test",
    )


def _record(root: Path) -> DocumentRecord:
    source_path = root / "source.txt"
    source_path.write_text("source", encoding="utf-8")
    return DocumentRecord(
        document_id="doc-1",
        filename="source.txt",
        title="Source",
        tenant_id="tenant-a",
        owner_id="owner-1",
        group_ids=["legal"],
        principal_ids=["user-1"],
        classification="confidential",
        status=DocumentStatus.UPLOADED,
        content_type="text/plain",
        size_bytes=6,
        content_hash="hash",
        source_path=str(source_path),
    )


def _sample_chunks(document_id: str):
    chunking_result = _load_sample_chunks()
    chunking_result.document_id = document_id
    for chunk in chunking_result.chunks:
        chunk.document_id = document_id
    return chunking_result


if __name__ == "__main__":
    unittest.main()
