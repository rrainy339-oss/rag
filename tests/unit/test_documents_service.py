from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from app.api.settings import APISettings
from app.documents.schemas import DocumentJobStatus, DocumentRecord, DocumentStatus
from app.documents.service import DocumentService, IndexAction
from app.indexing.schemas import IndexBackend, IndexingResult
from tests.unit.hybrid_fakes import CapturingHybridVectorStore, FakeHybridEmbeddingProvider
from tests.unit.test_retrieval import _load_sample_chunks


class DocumentServiceTest(unittest.TestCase):
    def test_qdrant_hybrid_document_indexing_writes_dense_and_sparse_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            captured: dict[str, int] = {}

            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]

            def build_vector_store(*, embedding_dimension: int) -> object:
                captured["embedding_dimension"] = embedding_dimension
                return vector_store

            service._build_vector_store = build_vector_store  # type: ignore[method-assign]

            record = _record(root)
            chunking_result = _sample_chunks(record.document_id)
            indexing_result, index_path = service._index_document(record, chunking_result)
            index_path_exists = index_path.exists()

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
            service._build_embedding_provider = lambda: FakeHybridEmbeddingProvider()  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: CapturingHybridVectorStore()
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

    def test_delete_document_removes_qdrant_points_before_marking_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
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
            service.chunk_repository.replace_document_chunks(
                record.document_id,
                chunking_result.chunks,
            )

            indexed_count_before_delete = vector_store.count()
            chunk_count_before_delete = service.chunk_repository.count()
            deleted = service.delete_document(record.document_id)
            indexed_count_after_delete = vector_store.count()
            chunk_count_after_delete = service.chunk_repository.count()

        self.assertGreater(indexed_count_before_delete, 0)
        self.assertGreater(chunk_count_before_delete, 0)
        self.assertEqual(indexed_count_after_delete, 0)
        self.assertEqual(deleted.status, DocumentStatus.DELETED)
        self.assertEqual(chunk_count_after_delete, 0)

    def test_delete_document_keeps_record_ready_when_qdrant_delete_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
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
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: FailingDeleteVectorStore()
            )

            with self.assertRaises(RuntimeError):
                service.delete_document(record.document_id)

            current = service.registry.get(record.document_id)

        self.assertIsNotNone(current)
        self.assertEqual(current.status, DocumentStatus.READY)

    def test_create_document_result_deduplicates_same_content_and_access(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)

            first = service.create_document_result(
                filename="handbook.txt",
                content_type="text/plain",
                file=BytesIO(b"same handbook"),
                title="First",
                tenant_id="tenant-a",
                owner_id="owner-1",
                group_ids=["legal"],
                principal_ids=["user-1"],
                classification="confidential",
            )
            duplicate = service.create_document_result(
                filename="handbook-copy.txt",
                content_type="text/plain",
                file=BytesIO(b"same handbook"),
                title="Duplicate title",
                tenant_id="tenant-a",
                owner_id="owner-1",
                group_ids=["legal"],
                principal_ids=["user-1"],
                classification="confidential",
            )
            different_access = service.create_document_result(
                filename="handbook-finance.txt",
                content_type="text/plain",
                file=BytesIO(b"same handbook"),
                title="Finance copy",
                tenant_id="tenant-a",
                owner_id="owner-1",
                group_ids=["finance"],
                principal_ids=["user-1"],
                classification="confidential",
            )
            documents = service.list_documents()

        self.assertTrue(first.created)
        self.assertFalse(duplicate.created)
        self.assertEqual(duplicate.record.document_id, first.record.document_id)
        self.assertTrue(different_access.created)
        self.assertEqual(len(documents), 2)

    def test_create_document_result_deduplicates_legacy_record_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)

            first = service.create_document_result(
                filename="handbook.txt",
                content_type="text/plain",
                file=BytesIO(b"legacy handbook"),
                title="First",
                tenant_id="tenant-a",
                owner_id="owner-1",
                group_ids=["legal"],
                principal_ids=["user-1"],
                classification="confidential",
            )
            service.registry.update(
                first.record.document_id,
                access_signature=None,
                dedupe_key=None,
            )
            duplicate = service.create_document_result(
                filename="handbook-copy.txt",
                content_type="text/plain",
                file=BytesIO(b"legacy handbook"),
                title="Duplicate title",
                tenant_id="tenant-a",
                owner_id="owner-1",
                group_ids=["legal"],
                principal_ids=["user-1"],
                classification="confidential",
            )
            refreshed = service.get_document(first.record.document_id)

        self.assertFalse(duplicate.created)
        self.assertEqual(duplicate.record.document_id, first.record.document_id)
        self.assertIsNotNone(refreshed)
        self.assertIsNotNone(refreshed.access_signature)
        self.assertIsNotNone(refreshed.dedupe_key)

    def test_update_permissions_rewrites_qdrant_payloads_for_ready_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
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
            vector_before = list(vector_store.records[0].vector)
            sparse_indices_before = list(vector_store.records[0].sparse_vector_indices)
            sparse_values_before = list(vector_store.records[0].sparse_vector_values)
            upsert_calls_before = vector_store.upsert_calls

            updated = service.update_permissions(
                record.document_id,
                tenant_id="tenant-b",
                owner_id="owner-2",
                group_ids=["finance"],
                principal_ids=["user-2"],
                classification="secret",
            )
            synced_record = vector_store.records[0]
            local_index = IndexingResult.model_validate_json(
                index_path.read_text(encoding="utf-8")
            )
            synced_chunk = service.chunk_repository.get("child-remote")

        self.assertEqual(updated.tenant_id, "tenant-b")
        self.assertEqual(vector_store.upsert_calls, upsert_calls_before)
        self.assertEqual(vector_store.set_payload_calls, 1)
        self.assertEqual(synced_record.vector, vector_before)
        self.assertEqual(synced_record.sparse_vector_indices, sparse_indices_before)
        self.assertEqual(synced_record.sparse_vector_values, sparse_values_before)
        self.assertEqual(synced_record.access.tenant_id, "tenant-b")
        self.assertEqual(synced_record.access.allowed_group_ids, ["finance"])
        self.assertEqual(synced_record.access.allowed_user_ids, ["user-2"])
        self.assertEqual(synced_record.metadata["tenant_id"], "tenant-b")
        self.assertEqual(synced_record.metadata["allowed_group_ids"], ["finance"])
        self.assertEqual(synced_record.metadata["allowed_user_ids"], ["user-2"])
        self.assertEqual(synced_record.metadata["classification"], "secret")
        self.assertEqual(local_index.records[0].access.tenant_id, "tenant-b")
        self.assertEqual(local_index.records[0].metadata["tenant_id"], "tenant-b")
        self.assertIsNotNone(synced_chunk)
        self.assertEqual(synced_chunk.access.tenant_id, "tenant-b")
        self.assertEqual(synced_chunk.access.allowed_group_ids, ["finance"])

    def test_document_job_runs_successfully_from_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            record = service.registry.create(_record(root))
            called: dict[str, object] = {}

            def process_document(
                document_id: str,
                *,
                job_id: str | None = None,
                raise_errors: bool = False,
            ) -> DocumentRecord:
                called["document_id"] = document_id
                called["job_id"] = job_id
                called["raise_errors"] = raise_errors
                return record

            service.process_document = process_document  # type: ignore[method-assign]
            queued = service.enqueue_document(record.document_id)
            completed = service.run_next_job(worker_id="test-worker")

        self.assertIsNotNone(completed)
        self.assertEqual(completed.status, DocumentJobStatus.SUCCEEDED)
        self.assertEqual(completed.progress, 100)
        self.assertEqual(completed.attempt, 1)
        self.assertEqual(called["document_id"], record.document_id)
        self.assertEqual(called["job_id"], queued.job_id)
        self.assertTrue(called["raise_errors"])

    def test_process_document_resumes_from_existing_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
            )

            record = _record(root)
            service.registry.create(record)
            chunking_result = _sample_chunks(record.document_id)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
            service.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                chunk_count=len(chunking_result.chunks),
                status=DocumentStatus.FAILED,
                error_message="previous indexing failure",
            )

            def fail_parse(record: DocumentRecord) -> object:
                del record
                raise AssertionError("parse should not run when chunks exist")

            def fail_chunk(record: DocumentRecord, parsed_document: object) -> object:
                del record, parsed_document
                raise AssertionError("chunking should not run when chunks exist")

            service._parse_document = fail_parse  # type: ignore[method-assign]
            service._chunk_document = fail_chunk  # type: ignore[method-assign]

            completed = service.process_document(record.document_id, raise_errors=True)
            synced_parent = service.chunk_repository.get("parent-remote")
            manifest = service.manifest_repository.get(settings.qdrant_collection)

        self.assertEqual(completed.status, DocumentStatus.READY)
        self.assertGreater(vector_store.count(), 0)
        self.assertIsNotNone(completed.index_path)
        self.assertIsNotNone(synced_parent)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.embedding_model, provider.model_name)
        self.assertFalse(settings.documents_collection_index_path.exists())
        self.assertFalse(settings.documents_collection_chunks_path.exists())

    def test_process_document_skips_ready_document_with_matching_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
            )

            record = _record(root)
            service.registry.create(record)
            chunking_result = _sample_chunks(record.document_id)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
            service.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                chunk_count=len(chunking_result.chunks),
                status=DocumentStatus.FAILED,
                error_message="previous indexing failure",
            )
            first = service.process_document(record.document_id, raise_errors=True)
            upsert_calls_after_first_run = vector_store.upsert_calls

            def fail_parse(record: DocumentRecord) -> object:
                del record
                raise AssertionError("parse should be skipped")

            def fail_chunk(record: DocumentRecord, parsed_document: object) -> object:
                del record, parsed_document
                raise AssertionError("chunking should be skipped")

            def fail_index(record: DocumentRecord, chunking_result: object) -> object:
                del record, chunking_result
                raise AssertionError("indexing should be skipped")

            service._parse_document = fail_parse  # type: ignore[method-assign]
            service._chunk_document = fail_chunk  # type: ignore[method-assign]
            service._index_document = fail_index  # type: ignore[method-assign]
            second = service.process_document(record.document_id, raise_errors=True)

        self.assertEqual(first.status, DocumentStatus.READY)
        self.assertEqual(second.status, DocumentStatus.READY)
        self.assertEqual(vector_store.upsert_calls, upsert_calls_after_first_run)

    def test_index_version_change_reindexes_without_rechunking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
            )

            record = _record(root)
            service.registry.create(record)
            chunking_result = _sample_chunks(record.document_id)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
            service.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                chunk_count=len(chunking_result.chunks),
                status=DocumentStatus.FAILED,
                error_message="previous indexing failure",
            )
            service.process_document(record.document_id, raise_errors=True)
            settings.documents_index_version = "idx-v2"

            def fail_parse(record: DocumentRecord) -> object:
                del record
                raise AssertionError("parse should not run when chunks are reusable")

            def fail_chunk(record: DocumentRecord, parsed_document: object) -> object:
                del record, parsed_document
                raise AssertionError("chunking should not run when chunks are reusable")

            service._parse_document = fail_parse  # type: ignore[method-assign]
            service._chunk_document = fail_chunk  # type: ignore[method-assign]
            reindexed = service.process_document(record.document_id, raise_errors=True)
            manifest = service.document_index_manifest_repository.get(record.document_id)

        self.assertEqual(reindexed.status, DocumentStatus.READY)
        self.assertEqual(vector_store.upsert_calls, 2)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.index_version, "idx-v2")

    def test_chunk_config_change_requires_rechunk_and_reindex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            provider = FakeHybridEmbeddingProvider()
            vector_store = CapturingHybridVectorStore()
            service._build_embedding_provider = lambda: provider  # type: ignore[method-assign]
            service._build_vector_store = (  # type: ignore[method-assign]
                lambda *, embedding_dimension: vector_store
            )

            record = _record(root)
            service.registry.create(record)
            chunking_result = _sample_chunks(record.document_id)
            chunks_path = root / "chunks.json"
            chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
            service.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                chunk_count=len(chunking_result.chunks),
                status=DocumentStatus.FAILED,
            )
            service.process_document(record.document_id, raise_errors=True)
            settings.documents_child_target_tokens = 512
            current = service.registry.get(record.document_id)
            self.assertIsNotNone(current)
            plan = service._plan_document_index(current)

        self.assertEqual(plan.action, IndexAction.REINDEX)
        self.assertTrue(plan.reuse_parsed)
        self.assertFalse(plan.reuse_chunks)

    def test_document_job_failure_respects_max_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            record = service.registry.create(_record(root))

            def process_document(
                document_id: str,
                *,
                job_id: str | None = None,
                raise_errors: bool = False,
            ) -> DocumentRecord:
                del document_id, job_id, raise_errors
                raise RuntimeError("boom")

            service.process_document = process_document  # type: ignore[method-assign]
            queued = service.enqueue_document(record.document_id)
            service.job_registry.update(queued.job_id, max_attempts=1)
            failed = service.run_next_job(worker_id="test-worker")

        self.assertIsNotNone(failed)
        self.assertEqual(failed.status, DocumentJobStatus.FAILED)
        self.assertEqual(failed.attempt, 1)
        self.assertIn("Traceback", failed.error_message or "")
        self.assertIn("RuntimeError: boom", failed.error_message or "")

    def test_cancelled_queued_document_job_is_not_claimed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _document_settings(root)
            service = DocumentService(settings)
            record = service.registry.create(_record(root))
            queued = service.enqueue_document(record.document_id)

            cancelled = service.cancel_job(queued.job_id)
            next_job = service.run_next_job(worker_id="test-worker")

        self.assertEqual(cancelled.status, DocumentJobStatus.CANCELLED)
        self.assertIsNone(next_job)


def _document_settings(root: Path) -> APISettings:
    return APISettings(
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


class FailingDeleteVectorStore(CapturingHybridVectorStore):
    def delete_by_document(self, document_id: str) -> int:
        raise RuntimeError(f"delete failed for {document_id}")


if __name__ == "__main__":
    unittest.main()
