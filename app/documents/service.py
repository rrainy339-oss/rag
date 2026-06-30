from __future__ import annotations

import hashlib
import re
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
from app.documents.registry import DocumentRegistry
from app.documents.schemas import DocumentRecord, DocumentStatus
from app.domain.schemas import AccessControl, ParsedDocument
from app.indexing.config import IndexingConfig
from app.indexing.embeddings.bge import BGEM3EmbeddingProvider
from app.indexing.embeddings.hashing import HashingEmbeddingProvider
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.permissions import access_to_payload
from app.indexing.schemas import (
    IndexBackend,
    IndexManifest,
    IndexingResult,
    IndexingStatus,
)
from app.indexing.vectorstores.memory import MemoryVectorStore
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore, QdrantVectorStore
from app.ingestion.pipeline import LocalParsingPipeline


CollectionUpdated = Callable[[], None]


class DocumentService:
    def __init__(
        self,
        settings: APISettings,
        *,
        registry: DocumentRegistry | None = None,
        on_collection_updated: CollectionUpdated | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry or DocumentRegistry(settings.documents_db_path)
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

    def process_document(self, document_id: str) -> DocumentRecord:
        try:
            record = self._require_document(document_id)
            record = self.registry.update_status(record.document_id, DocumentStatus.PARSING)
            parsed_document, parsed_json_path = self._parse_document(record)

            self.registry.update_artifacts(
                record.document_id,
                parsed_path=parsed_json_path,
                status=DocumentStatus.CHUNKING,
            )
            chunking_result, chunks_path = self._chunk_document(record, parsed_document)

            self.registry.update_artifacts(
                record.document_id,
                chunks_path=chunks_path,
                chunk_count=len(chunking_result.chunks),
                status=DocumentStatus.INDEXING,
            )
            indexing_result, index_path = self._index_document(record, chunking_result)

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
        except Exception as exc:
            record = self.registry.update_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )
            self._notify_collection_updated()
            return record

    def reindex_document(self, document_id: str) -> DocumentRecord:
        self._require_document(document_id)
        return self.process_document(document_id)

    def delete_document(self, document_id: str) -> DocumentRecord:
        self._require_document(document_id)
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
            self._rewrite_access(record)
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
                fallback_backend=IndexBackend(self.settings.documents_backend),
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

    def _index_document(
        self,
        record: DocumentRecord,
        chunking_result: ChunkingResult,
    ) -> tuple[IndexingResult, Path]:
        embedding_provider = self._build_embedding_provider()
        vector_store = None
        try:
            backend = IndexBackend(self.settings.documents_backend)
            if (
                backend == IndexBackend.QDRANT_HYBRID
                and self.settings.documents_embedding_provider != "bge-m3"
            ):
                raise ValueError(
                    "Document qdrant_hybrid indexing requires "
                    "RAG_DOCUMENTS_EMBEDDING_PROVIDER=bge-m3."
                )
            vector_store = self._build_vector_store(
                backend=backend,
                embedding_dimension=embedding_provider.dimension,
            )
            indexer = EnterpriseIndexer(
                config=IndexingConfig(
                    backend=backend,
                    embedding_model=embedding_provider.model_name,
                    embedding_dimension=embedding_provider.dimension,
                    embedding_batch_size=self.settings.documents_embedding_batch_size,
                    dense_vector_name=self.settings.dense_vector_name,
                    sparse_vector_name=self.settings.sparse_vector_name,
                    sparse_top_n=self.settings.sparse_top_n,
                    force_reindex=True,
                ),
                embedding_provider=embedding_provider,
                vector_store=vector_store,
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
        if self.settings.documents_embedding_provider == "bge-m3":
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
        return HashingEmbeddingProvider(self.settings.ingestion_embedding_dimension)

    def _build_vector_store(
        self,
        *,
        backend: IndexBackend,
        embedding_dimension: int,
    ) -> object:
        if backend == IndexBackend.MEMORY:
            return MemoryVectorStore()
        if backend == IndexBackend.QDRANT:
            return QdrantVectorStore(
                url=self.settings.qdrant_url,
                path=self._qdrant_path("qdrant"),
                api_key=self.settings.qdrant_api_key,
                collection_name=self.settings.qdrant_collection,
                vector_size=embedding_dimension,
                timeout=self.settings.qdrant_timeout,
            )
        if backend == IndexBackend.QDRANT_HYBRID:
            return QdrantHybridVectorStore(
                url=self.settings.qdrant_url,
                path=self._qdrant_path("qdrant_hybrid"),
                api_key=self.settings.qdrant_api_key,
                collection_name=self.settings.qdrant_collection,
                vector_size=embedding_dimension,
                dense_vector_name=self.settings.dense_vector_name,
                sparse_vector_name=self.settings.sparse_vector_name,
                timeout=self.settings.qdrant_timeout,
            )
        raise ValueError(f"Unsupported document indexing backend: {backend}")

    def _qdrant_path(self, default_folder: str) -> Path | None:
        if self.settings.qdrant_url is not None:
            return None
        if self.settings.qdrant_path is not None:
            return self.settings.qdrant_path
        return self.settings.documents_collection_index_path.parent / default_folder

    def _configured_embedding_model(self) -> str:
        if self.settings.documents_embedding_provider == "bge-m3":
            return self.settings.bge_model or "BAAI/bge-m3"
        return "hashing-embedding"

    def _configured_embedding_dimension(self) -> int:
        if self.settings.documents_embedding_provider == "bge-m3":
            return BGEM3EmbeddingProvider.dimension
        return self.settings.ingestion_embedding_dimension

    def _rewrite_access(self, record: DocumentRecord) -> None:
        if not record.chunks_path or not record.index_path:
            return
        chunks_path = Path(record.chunks_path)
        index_path = Path(record.index_path)
        if not chunks_path.exists() or not index_path.exists():
            return

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
    fallback_backend: IndexBackend,
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
            backend=fallback_backend,
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
        backend=first.manifest.backend,
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
    backend: IndexBackend,
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
        backend=backend,
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
    if left.backend != IndexBackend.QDRANT_HYBRID:
        return True
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
