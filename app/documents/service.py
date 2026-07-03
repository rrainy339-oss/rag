from __future__ import annotations

import hashlib
import re
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import BinaryIO
from uuid import uuid4

from app.api.settings import APISettings
from app.chunking.config import ChunkingConfig
from app.chunking.pipeline import ChunkingPipeline
from app.chunking.schemas import ChunkingResult
from app.documents.jobs import DocumentJobRegistry, TERMINAL_JOB_STATUSES
from app.documents.registry import DocumentRegistry
from app.documents.schemas import (
    DocumentJob,
    DocumentJobStage,
    DocumentJobStatus,
    DocumentJobType,
    DocumentRecord,
    DocumentStatus,
)
from app.domain.schemas import AccessControl, ParsedDocument
from app.indexing.config import IndexingConfig
from app.indexing.embeddings.bge import BGEM3EmbeddingProvider
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.permissions import access_to_payload
from app.indexing.schemas import (
    IndexBackend,
    IndexManifest,
    IndexingResult,
    IndexingStatus,
)
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore
from app.ingestion.pipeline import LocalParsingPipeline


CollectionUpdated = Callable[[], None]


class DocumentService:
    def __init__(
        self,
        settings: APISettings,
        *,
        registry: DocumentRegistry | None = None,
        job_registry: DocumentJobRegistry | None = None,
        on_collection_updated: CollectionUpdated | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry or DocumentRegistry(settings.documents_db_path)
        self.job_registry = job_registry or DocumentJobRegistry(settings.documents_db_path)
        self.on_collection_updated = on_collection_updated
        self._collection_lock = Lock()

    def ensure_collection_files(self) -> None:
        if (
            self.settings.index_path == self.settings.documents_collection_index_path
            or self.settings.chunks_path == self.settings.documents_collection_chunks_path
        ):
            if (
                not self.settings.documents_collection_index_path.exists()
                or not self.settings.documents_collection_chunks_path.exists()
            ):
                self.rebuild_collection()

    def create_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        file: BinaryIO,
        title: str | None,
        tenant_id: str | None,
        owner_id: str | None,
        group_ids: list[str],
        principal_ids: list[str],
        classification: str | None,
    ) -> DocumentRecord:
        document_id = str(uuid4())
        clean_filename = _safe_filename(filename)
        document_dir = self._document_dir(document_id)
        raw_dir = document_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        source_path = raw_dir / clean_filename
        size_bytes, content_hash = _write_and_hash(file, source_path)
        now = datetime.now(timezone.utc)
        record = DocumentRecord(
            document_id=document_id,
            filename=clean_filename,
            title=_clean(title) or Path(clean_filename).stem,
            tenant_id=_clean(tenant_id),
            owner_id=_clean(owner_id),
            group_ids=_clean_many(group_ids),
            principal_ids=_clean_many(principal_ids),
            classification=_clean(classification),
            status=DocumentStatus.UPLOADED,
            content_type=content_type,
            size_bytes=size_bytes,
            content_hash=content_hash,
            source_path=str(source_path),
            created_at=now,
            updated_at=now,
        )
        return self.registry.create(record)

    def list_documents(self, *, include_deleted: bool = False) -> list[DocumentRecord]:
        return self.registry.list(include_deleted=include_deleted)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.registry.get(document_id)

    def enqueue_document(
        self,
        document_id: str,
        *,
        job_type: DocumentJobType = DocumentJobType.INGEST,
    ) -> DocumentJob:
        self._require_document(document_id)
        return self.job_registry.create(document_id=document_id, job_type=job_type)

    def list_jobs(
        self,
        *,
        document_id: str | None = None,
        include_terminal: bool = True,
        limit: int = 100,
    ) -> list[DocumentJob]:
        return self.job_registry.list(
            document_id=document_id,
            include_terminal=include_terminal,
            limit=limit,
        )

    def latest_job(self, document_id: str) -> DocumentJob | None:
        return self.job_registry.latest_for_document(document_id)

    def get_job(self, job_id: str) -> DocumentJob | None:
        return self.job_registry.get(job_id)

    def cancel_job(self, job_id: str) -> DocumentJob:
        return self.job_registry.request_cancel(job_id)

    def retry_job(self, job_id: str) -> DocumentJob:
        return self.job_registry.retry(job_id)

    def run_next_job(self, *, worker_id: str = "local-worker") -> DocumentJob | None:
        job = self.job_registry.claim_next(worker_id=worker_id)
        if job is None:
            return None
        return self.run_job(job.job_id, worker_id=worker_id)

    def run_job(self, job_id: str, *, worker_id: str = "local-worker") -> DocumentJob:
        job = self.job_registry.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        if job.status != DocumentJobStatus.RUNNING:
            claimed = self.job_registry.claim(job_id, worker_id=worker_id)
            if claimed is None:
                raise ValueError(f"Job {job_id} is not ready to run.")
            job = claimed

        try:
            self._raise_if_cancelled(job.job_id)
            if job.job_type in {DocumentJobType.INGEST, DocumentJobType.REINDEX}:
                self.process_document(
                    job.document_id,
                    job_id=job.job_id,
                    raise_errors=True,
                )
            else:
                raise ValueError(f"Unsupported document job type: {job.job_type}")
            return self.job_registry.mark_succeeded(job.job_id)
        except _JobCancelled:
            return self.job_registry.mark_cancelled(job.job_id)
        except Exception as exc:
            return self.job_registry.mark_failed(
                job.job_id,
                error_message=_exception_traceback(exc),
                retry_delay_seconds=_retry_delay_seconds(job.attempt),
            )

    def process_document(
        self,
        document_id: str,
        *,
        job_id: str | None = None,
        raise_errors: bool = False,
    ) -> DocumentRecord:
        try:
            record = self._require_document(document_id)

            chunking_result: ChunkingResult | None = None
            chunks_path = _existing_path(record.chunks_path)
            if chunks_path is not None:
                chunking_result, chunks_path = self._load_chunking_result(
                    record,
                    chunks_path,
                )
            else:
                parsed_document: ParsedDocument | None = None
                parsed_json_path = _existing_path(record.parsed_path)
                if parsed_json_path is not None:
                    parsed_document, parsed_json_path = self._load_parsed_document(
                        record,
                        parsed_json_path,
                    )
                else:
                    self._set_job_stage(job_id, DocumentJobStage.PARSING, 10)
                    record = self.registry.update_status(
                        record.document_id,
                        DocumentStatus.PARSING,
                    )
                    parsed_document, parsed_json_path = self._parse_document(record)
                    self._raise_if_cancelled(job_id)

                self._set_job_stage(job_id, DocumentJobStage.CHUNKING, 35)
                record = self.registry.update_artifacts(
                    record.document_id,
                    parsed_path=parsed_json_path,
                    status=DocumentStatus.CHUNKING,
                )
                chunking_result, chunks_path = self._chunk_document(
                    record,
                    parsed_document,
                )
                self._raise_if_cancelled(job_id)

            self._set_job_stage(job_id, DocumentJobStage.EMBEDDING, 60)
            record = self.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                chunk_count=len(chunking_result.chunks),
                status=DocumentStatus.INDEXING,
            )

            index_path = _existing_path(record.index_path)
            if index_path is not None:
                indexing_result, index_path = self._load_indexing_result(
                    record,
                    index_path,
                )
            else:
                self._set_job_stage(job_id, DocumentJobStage.INDEXING, 75)
                indexing_result, index_path = self._index_document(record, chunking_result)
            self._raise_if_cancelled(job_id)

            self._set_job_stage(job_id, DocumentJobStage.REBUILDING_COLLECTION, 90)
            record = self.registry.update_artifacts(
                record.document_id,
                index_path=index_path,
                indexed_count=indexing_result.indexed_count,
                status=DocumentStatus.READY,
                error_message=None,
            )
            self.rebuild_collection()
            self._notify_collection_updated()
            return record
        except _JobCancelled:
            record = self._require_document(document_id)
            if record.status != DocumentStatus.READY:
                record = self.registry.update_status(
                    document_id,
                    DocumentStatus.UPLOADED,
                    error_message="Job cancelled.",
                )
            self._notify_collection_updated()
            if raise_errors:
                raise
            return record
        except Exception as exc:
            record = self.registry.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )
            self._notify_collection_updated()
            if raise_errors:
                raise
            return record

    def reindex_document(self, document_id: str) -> DocumentRecord:
        self._require_document(document_id)
        return self.process_document(document_id)

    def delete_document(self, document_id: str) -> DocumentRecord:
        existing = self._require_document(document_id)
        self._cancel_document_jobs(document_id)
        self._delete_index_records(existing)
        record = self.registry.update_status(document_id, DocumentStatus.DELETED)
        self.rebuild_collection()
        self._notify_collection_updated()
        return record

    def update_permissions(
        self,
        document_id: str,
        *,
        tenant_id: str | None,
        owner_id: str | None,
        group_ids: list[str],
        principal_ids: list[str],
        classification: str | None,
    ) -> DocumentRecord:
        existing = self._require_document(document_id)
        record = self.registry.update_permissions(
            document_id,
            tenant_id=_clean(tenant_id),
            owner_id=_clean(owner_id),
            group_ids=_clean_many(group_ids),
            principal_ids=_clean_many(principal_ids),
            classification=_clean(classification),
        )
        if existing.status == DocumentStatus.READY:
            indexing_result = self._rewrite_access(record)
            if indexing_result is not None:
                self._set_index_record_payloads(indexing_result)
            self.rebuild_collection()
            self._notify_collection_updated()
        return record

    def rebuild_collection(self) -> None:
        with self._collection_lock:
            records = [
                record
                for record in self.registry.list()
                if record.status == DocumentStatus.READY
                and record.chunks_path
                and record.index_path
                and Path(record.chunks_path).exists()
                and Path(record.index_path).exists()
            ]
            collection_chunks = _combine_chunks(records)
            collection_index = _combine_indexes(
                records,
                fallback_embedding_model=self._configured_embedding_model(),
                fallback_embedding_dimension=self._configured_embedding_dimension(),
                dense_vector_name=self.settings.dense_vector_name,
                sparse_vector_name=self.settings.sparse_vector_name,
                sparse_top_n=self.settings.sparse_top_n,
            )
            chunks_path = self.settings.documents_collection_chunks_path
            index_path = self.settings.documents_collection_index_path
            chunks_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            chunks_path.write_text(collection_chunks.to_json(), encoding="utf-8")
            index_path.write_text(collection_index.to_json(), encoding="utf-8")

    def _parse_document(self, record: DocumentRecord) -> tuple[ParsedDocument, Path]:
        pipeline = LocalParsingPipeline(
            recursive=False,
            pdf_backend=self.settings.documents_pdf_backend,
            do_ocr=self.settings.documents_do_ocr,
            artifacts_path=self.settings.documents_docling_artifacts_path,
            access=_access_from_record(record),
        )
        parsed_documents = list(pipeline.parse_path(record.source_path))
        if not parsed_documents:
            raise ValueError(f"No parsed document was produced for {record.filename}")
        parsed_document = _normalize_parsed_document(parsed_documents[0], record)
        parsed_dir = self._document_dir(record.document_id) / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        parsed_json_path = parsed_dir / f"{Path(record.filename).stem}.parsed.json"
        parsed_markdown_path = parsed_dir / f"{Path(record.filename).stem}.parsed.md"
        parsed_json_path.write_text(parsed_document.to_json(), encoding="utf-8")
        parsed_markdown_path.write_text(parsed_document.markdown, encoding="utf-8")
        return parsed_document, parsed_json_path

    def _chunk_document(
        self,
        record: DocumentRecord,
        parsed_document: ParsedDocument,
    ) -> tuple[ChunkingResult, Path]:
        pipeline = ChunkingPipeline(
            ChunkingConfig(
                child_target_tokens=self.settings.documents_child_target_tokens,
                child_max_tokens=self.settings.documents_child_max_tokens,
                parent_target_tokens=self.settings.documents_parent_target_tokens,
                parent_max_tokens=self.settings.documents_parent_max_tokens,
            )
        )
        chunking_result = pipeline.chunk_document(parsed_document)
        _enrich_chunks(chunking_result, record)
        chunks_dir = self._document_dir(record.document_id) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = chunks_dir / f"{Path(record.filename).stem}.chunks.json"
        chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
        return chunking_result, chunks_path

    def _load_parsed_document(
        self,
        record: DocumentRecord,
        parsed_json_path: Path,
    ) -> tuple[ParsedDocument, Path]:
        parsed_document = ParsedDocument.model_validate_json(
            parsed_json_path.read_text(encoding="utf-8")
        )
        parsed_document = _normalize_parsed_document(parsed_document, record)
        parsed_json_path.write_text(parsed_document.to_json(), encoding="utf-8")
        return parsed_document, parsed_json_path

    def _load_chunking_result(
        self,
        record: DocumentRecord,
        chunks_path: Path,
    ) -> tuple[ChunkingResult, Path]:
        chunking_result = ChunkingResult.model_validate_json(
            chunks_path.read_text(encoding="utf-8")
        )
        _enrich_chunks(chunking_result, record)
        chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
        return chunking_result, chunks_path

    def _load_indexing_result(
        self,
        record: DocumentRecord,
        index_path: Path,
    ) -> tuple[IndexingResult, Path]:
        indexing_result = IndexingResult.model_validate_json(
            index_path.read_text(encoding="utf-8")
        )
        _enrich_index_records(indexing_result, record)
        index_path.write_text(indexing_result.to_json(), encoding="utf-8")
        return indexing_result, index_path

    def _index_document(
        self,
        record: DocumentRecord,
        chunking_result: ChunkingResult,
    ) -> tuple[IndexingResult, Path]:
        embedding_provider = self._build_embedding_provider()
        vector_store = None
        try:
            vector_store = self._build_vector_store(
                embedding_dimension=embedding_provider.dimension,
            )
            indexer = EnterpriseIndexer(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                config=IndexingConfig(
                    backend=IndexBackend.QDRANT_HYBRID,
                    embedding_model=embedding_provider.model_name,
                    embedding_dimension=embedding_provider.dimension,
                    embedding_batch_size=self.settings.documents_embedding_batch_size,
                    dense_vector_name=self.settings.dense_vector_name,
                    sparse_vector_name=self.settings.sparse_vector_name,
                    sparse_top_n=self.settings.sparse_top_n,
                    force_reindex=True,
                ),
            )
            indexing_result = indexer.index(chunking_result)
            _enrich_index_records(indexing_result, record)
            index_dir = self._document_dir(record.document_id) / "index"
            index_dir.mkdir(parents=True, exist_ok=True)
            index_path = index_dir / f"{Path(record.filename).stem}.index.json"
            index_path.write_text(indexing_result.to_json(), encoding="utf-8")
            return indexing_result, index_path
        finally:
            for resource in (vector_store, embedding_provider):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()

    def _build_embedding_provider(self) -> object:
        return BGEM3EmbeddingProvider(
            model_name=self.settings.bge_model or "BAAI/bge-m3",
            use_fp16=self.settings.bge_use_fp16,
            max_length=self.settings.bge_max_length,
            batch_size=self.settings.bge_batch_size,
            cache_dir=(
                str(self.settings.bge_cache_dir)
                if self.settings.bge_cache_dir
                else None
            ),
            sparse_top_n=self.settings.sparse_top_n,
        )

    def _build_vector_store(
        self,
        *,
        embedding_dimension: int,
    ) -> object:
        return QdrantHybridVectorStore(
            url=self.settings.qdrant_url,
            path=self._qdrant_path(),
            api_key=self.settings.qdrant_api_key,
            collection_name=self.settings.qdrant_collection,
            vector_size=embedding_dimension,
            dense_vector_name=self.settings.dense_vector_name,
            sparse_vector_name=self.settings.sparse_vector_name,
            timeout=self.settings.qdrant_timeout,
        )

    def _qdrant_path(self) -> Path | None:
        if self.settings.qdrant_url is not None:
            return None
        if self.settings.qdrant_path is not None:
            return self.settings.qdrant_path
        return self.settings.documents_collection_index_path.parent / "qdrant_hybrid"

    def _configured_embedding_model(self) -> str:
        return self.settings.bge_model or "BAAI/bge-m3"

    def _configured_embedding_dimension(self) -> int:
        return BGEM3EmbeddingProvider.dimension

    def _delete_index_records(self, record: DocumentRecord) -> int:
        if not _may_have_index_records(record):
            return 0

        vector_store = None
        try:
            vector_store = self._build_vector_store(
                embedding_dimension=self._record_embedding_dimension(record),
            )
            return int(vector_store.delete_by_document(record.document_id) or 0)
        finally:
            close = getattr(vector_store, "close", None)
            if callable(close):
                close()

    def _record_embedding_dimension(self, record: DocumentRecord) -> int:
        if record.index_path:
            index_path = Path(record.index_path)
            if index_path.exists():
                try:
                    indexing_result = IndexingResult.model_validate_json(
                        index_path.read_text(encoding="utf-8")
                    )
                    return indexing_result.manifest.embedding_dimension
                except Exception:
                    pass
        return self._configured_embedding_dimension()

    def _set_index_record_payloads(self, indexing_result: IndexingResult) -> int:
        if not indexing_result.records:
            return 0

        vector_store = None
        try:
            vector_store = self._build_vector_store(
                embedding_dimension=indexing_result.manifest.embedding_dimension,
            )
            set_payload = getattr(vector_store, "set_payload", None)
            if callable(set_payload):
                return int(set_payload(indexing_result.records) or 0)
            raise RuntimeError("Vector store does not support payload-only updates.")
        finally:
            close = getattr(vector_store, "close", None)
            if callable(close):
                close()

    def _rewrite_access(self, record: DocumentRecord) -> IndexingResult | None:
        if not record.chunks_path or not record.index_path:
            return None
        chunks_path = Path(record.chunks_path)
        index_path = Path(record.index_path)
        if not chunks_path.exists() or not index_path.exists():
            return None

        chunking_result = ChunkingResult.model_validate_json(
            chunks_path.read_text(encoding="utf-8")
        )
        _enrich_chunks(chunking_result, record)
        chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")

        indexing_result = IndexingResult.model_validate_json(
            index_path.read_text(encoding="utf-8")
        )
        _enrich_index_records(indexing_result, record)
        indexing_result.manifest.metadata["permissions_updated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        index_path.write_text(indexing_result.to_json(), encoding="utf-8")
        return indexing_result

    def _document_dir(self, document_id: str) -> Path:
        return self.settings.documents_storage_dir / document_id

    def _require_document(self, document_id: str) -> DocumentRecord:
        record = self.registry.get(document_id)
        if record is None:
            raise KeyError(document_id)
        return record

    def _notify_collection_updated(self) -> None:
        if self.on_collection_updated is not None:
            self.on_collection_updated()

    def _set_job_stage(
        self,
        job_id: str | None,
        stage: DocumentJobStage,
        progress: int,
    ) -> None:
        if job_id is not None:
            self.job_registry.set_stage(job_id, stage=stage, progress=progress)

    def _raise_if_cancelled(self, job_id: str | None) -> None:
        if job_id is None:
            return
        job = self.job_registry.get(job_id)
        if job is not None and job.cancel_requested:
            raise _JobCancelled()

    def _cancel_document_jobs(self, document_id: str) -> None:
        for job in self.job_registry.list(
            document_id=document_id,
            include_terminal=False,
        ):
            self.job_registry.request_cancel(job.job_id)


class _JobCancelled(RuntimeError):
    pass


def _combine_chunks(records: list[DocumentRecord]) -> ChunkingResult:
    results = [
        ChunkingResult.model_validate_json(Path(record.chunks_path).read_text(encoding="utf-8"))
        for record in records
        if record.chunks_path
    ]
    chunks = [chunk for result in results for chunk in result.chunks]
    return ChunkingResult(
        document_id="default",
        chunks=chunks,
        stats={
            "combined": True,
            "document_count": len(results),
            "chunk_count": len(chunks),
            "source_document_ids": [result.document_id for result in results],
        },
    )


def _combine_indexes(
    records: list[DocumentRecord],
    *,
    fallback_embedding_model: str,
    fallback_embedding_dimension: int,
    dense_vector_name: str,
    sparse_vector_name: str,
    sparse_top_n: int | None,
) -> IndexingResult:
    results = [
        IndexingResult.model_validate_json(Path(record.index_path).read_text(encoding="utf-8"))
        for record in records
        if record.index_path
    ]
    if not results:
        return _empty_indexing_result(
            embedding_model=fallback_embedding_model,
            embedding_dimension=fallback_embedding_dimension,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            sparse_top_n=sparse_top_n,
        )

    first = results[0]
    for result in results[1:]:
        if result.manifest.embedding_model != first.manifest.embedding_model:
            raise ValueError("Cannot combine indexes with different embedding models.")
        if result.manifest.embedding_dimension != first.manifest.embedding_dimension:
            raise ValueError("Cannot combine indexes with different embedding dimensions.")
        if result.manifest.index_version != first.manifest.index_version:
            raise ValueError("Cannot combine indexes with different index versions.")
        if result.manifest.backend != first.manifest.backend:
            raise ValueError("Cannot combine indexes with different backends.")
        if not _manifest_vector_config_matches(first.manifest, result.manifest):
            raise ValueError("Cannot combine indexes with different vector configurations.")

    records_flat = [record for result in results for record in result.records]
    manifest = IndexManifest(
        document_id="default",
        source_uri=None,
        document_hash=_hash_values(
            [result.document_id for result in results]
            + [record.content_hash for record in records_flat]
        ),
        chunk_hashes={record.chunk_id: record.content_hash for record in records_flat},
        embedding_model=first.manifest.embedding_model,
        embedding_dimension=first.manifest.embedding_dimension,
        index_version=first.manifest.index_version,
        backend=IndexBackend.QDRANT_HYBRID,
        status=IndexingStatus.INDEXED,
        metadata={
            "combined": True,
            "document_count": len(results),
            "record_count": len(records_flat),
            "source_document_ids": [result.document_id for result in results],
            "dense_vector_name": first.manifest.metadata.get("dense_vector_name"),
            "sparse_vector_name": first.manifest.metadata.get("sparse_vector_name"),
            "sparse_top_n": first.manifest.metadata.get("sparse_top_n"),
        },
    )
    return IndexingResult(
        document_id="default",
        status=IndexingStatus.INDEXED,
        indexed_count=len(records_flat),
        skipped_count=sum(result.skipped_count for result in results),
        deleted_count=sum(result.deleted_count for result in results),
        manifest=manifest,
        records=records_flat,
    )


def _empty_indexing_result(
    *,
    embedding_model: str,
    embedding_dimension: int,
    dense_vector_name: str,
    sparse_vector_name: str,
    sparse_top_n: int | None,
) -> IndexingResult:
    manifest = IndexManifest(
        document_id="default",
        source_uri=None,
        document_hash=_hash_values(["empty"]),
        chunk_hashes={},
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        index_version="idx-v1",
        backend=IndexBackend.QDRANT_HYBRID,
        status=IndexingStatus.INDEXED,
        metadata={
            "combined": True,
            "document_count": 0,
            "record_count": 0,
            "dense_vector_name": dense_vector_name,
            "sparse_vector_name": sparse_vector_name,
            "sparse_top_n": sparse_top_n,
        },
    )
    return IndexingResult(
        document_id="default",
        status=IndexingStatus.INDEXED,
        indexed_count=0,
        manifest=manifest,
        records=[],
    )


def _manifest_vector_config_matches(
    left: IndexManifest,
    right: IndexManifest,
) -> bool:
    return (
        left.metadata.get("dense_vector_name") == right.metadata.get("dense_vector_name")
        and left.metadata.get("sparse_vector_name")
        == right.metadata.get("sparse_vector_name")
        and left.metadata.get("sparse_top_n") == right.metadata.get("sparse_top_n")
    )


def _normalize_parsed_document(
    parsed_document: ParsedDocument,
    record: DocumentRecord,
) -> ParsedDocument:
    metadata = parsed_document.metadata.model_copy(
        update={
            "access": _access_from_record(record),
            "extra": {
                **parsed_document.metadata.extra,
                "document_registry_id": record.document_id,
                "document_title": record.title,
                "owner_id": record.owner_id,
            },
        }
    )
    return parsed_document.model_copy(
        update={
            "document_id": record.document_id,
            "metadata": metadata,
        }
    )


def _enrich_chunks(chunking_result: ChunkingResult, record: DocumentRecord) -> None:
    access = _access_from_record(record)
    metadata = _document_metadata(record)
    for chunk in chunking_result.chunks:
        chunk.document_id = record.document_id
        chunk.access = access
        chunk.metadata.update(metadata)


def _enrich_index_records(indexing_result: IndexingResult, record: DocumentRecord) -> None:
    access = _access_from_record(record)
    metadata = {
        **_document_metadata(record),
        **access_to_payload(access),
    }
    for index_record in indexing_result.records:
        index_record.document_id = record.document_id
        index_record.access = access
        index_record.metadata.update(metadata)
        index_record.metadata["document_id"] = record.document_id


def _may_have_index_records(record: DocumentRecord) -> bool:
    return (
        record.status in {DocumentStatus.READY, DocumentStatus.INDEXING}
        or bool(record.index_path)
        or record.indexed_count > 0
    )


def _document_metadata(record: DocumentRecord) -> dict[str, object | None]:
    return {
        "document_id": record.document_id,
        "document_title": record.title,
        "source_filename": record.filename,
        "tenant_id": record.tenant_id,
        "owner_id": record.owner_id,
        "group_ids": record.group_ids,
        "principal_ids": record.principal_ids,
        "classification": record.classification,
    }


def _access_from_record(record: DocumentRecord) -> AccessControl:
    return AccessControl(
        tenant_id=record.tenant_id,
        allowed_user_ids=record.principal_ids,
        allowed_group_ids=record.group_ids,
        classification=record.classification,
    )


def _write_and_hash(file: BinaryIO, target_path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        file.seek(0)
    except (AttributeError, OSError):
        pass
    with target_path.open("wb") as output:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _hash_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _retry_delay_seconds(attempt: int) -> int:
    delays = [30, 120, 600]
    index = max(0, min(len(delays) - 1, attempt - 1))
    return delays[index]


def _exception_traceback(exc: Exception) -> str:
    formatted = traceback.format_exc()
    if formatted and formatted.strip() != "NoneType: None":
        return formatted
    return f"{type(exc).__name__}: {exc}"


def _existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.exists() else None


def _safe_filename(value: str) -> str:
    filename = Path(value or "document").name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip(" .")
    return cleaned or "document"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_many(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        for part in str(value).replace(";", ",").split(","):
            item = part.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            cleaned.append(item)
    return cleaned
