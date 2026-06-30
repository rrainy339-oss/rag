from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Protocol

from app.chunking.schemas import Chunk, ChunkingResult
from app.indexing.batching import batched
from app.indexing.config import IndexingConfig
from app.indexing.embeddings.base import EmbeddingProvider
from app.indexing.manifest import InMemoryManifestStore
from app.indexing.permissions import access_to_payload
from app.indexing.schemas import (
    IndexBackend,
    IndexManifest,
    IndexRecord,
    IndexingResult,
    IndexingStatus,
)


class ManifestStore(Protocol):
    def get(self, document_id: str) -> IndexManifest | None:
        """Return the current manifest for a document, if any."""

    def save(self, manifest: IndexManifest) -> None:
        """Persist a manifest."""


class EnterpriseIndexer:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: object,
        config: IndexingConfig | None = None,
        manifest_store: ManifestStore | None = None,
    ) -> None:
        self.config = config or IndexingConfig()
        if self.config.backend != IndexBackend.QDRANT_HYBRID:
            raise ValueError("EnterpriseIndexer only supports qdrant_hybrid indexing.")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.manifest_store = manifest_store or InMemoryManifestStore()

    def index(self, chunking_result: ChunkingResult) -> IndexingResult:
        selected_chunks = self._select_chunks(chunking_result.chunks)
        chunk_hashes = {chunk.chunk_id: chunk.content_hash for chunk in selected_chunks}
        document_hash = _document_hash(chunking_result)
        source_uri = _source_uri(selected_chunks)

        existing_manifest = self.manifest_store.get(chunking_result.document_id)
        if (
            existing_manifest
            and not self.config.force_reindex
            and existing_manifest.matches(
                document_hash=document_hash,
                chunk_hashes=chunk_hashes,
                embedding_model=self.embedding_provider.model_name,
                embedding_dimension=self.embedding_provider.dimension,
                index_version=self.config.index_version,
                backend=self.config.backend,
            )
            and _manifest_config_matches(existing_manifest, self.config)
        ):
            return IndexingResult(
                document_id=chunking_result.document_id,
                status=IndexingStatus.SKIPPED,
                indexed_count=0,
                skipped_count=len(selected_chunks),
                manifest=existing_manifest,
                records=[],
            )

        deleted_count = self._delete_existing(chunking_result.document_id)
        records = self._build_records(selected_chunks)
        self._upsert(records)

        manifest = IndexManifest(
            document_id=chunking_result.document_id,
            source_uri=source_uri,
            document_hash=document_hash,
            chunk_hashes=chunk_hashes,
            embedding_model=self.embedding_provider.model_name,
            embedding_dimension=self.embedding_provider.dimension,
            index_version=self.config.index_version,
            backend=self.config.backend,
            status=IndexingStatus.INDEXED,
            indexed_at=datetime.now(timezone.utc),
            metadata={
                "selected_chunk_count": len(selected_chunks),
                "all_chunk_count": len(chunking_result.chunks),
                "indexed_chunk_types": sorted(
                    chunk_type.value for chunk_type in self.config.index_chunk_types
                ),
                "dense_vector_name": self.config.dense_vector_name,
                "sparse_vector_name": self.config.sparse_vector_name,
                "sparse_top_n": self.config.sparse_top_n,
            },
        )
        self.manifest_store.save(manifest)

        return IndexingResult(
            document_id=chunking_result.document_id,
            status=IndexingStatus.INDEXED,
            indexed_count=len(records),
            skipped_count=0,
            deleted_count=deleted_count,
            manifest=manifest,
            records=records,
        )

    def _select_chunks(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        return [
            chunk
            for chunk in sorted(chunks, key=lambda item: item.chunk_index)
            if chunk.chunk_type in self.config.index_chunk_types
        ]

    def _delete_existing(self, document_id: str) -> int:
        return int(self.vector_store.delete_by_document(document_id) or 0)

    def _build_records(self, chunks: list[Chunk]) -> list[IndexRecord]:
        records: list[IndexRecord] = []
        for batch in batched(chunks, self.config.embedding_batch_size):
            texts = [chunk.contextual_text for chunk in batch]
            embed_hybrid = getattr(self.embedding_provider, "embed_documents_hybrid", None)
            if not callable(embed_hybrid):
                raise ValueError(
                    "qdrant_hybrid indexing requires an embedding provider with "
                    "embed_documents_hybrid(). Use BGE-M3."
                )
            embeddings = embed_hybrid(texts)
            if len(embeddings) != len(batch):
                raise ValueError(
                    "Embedding provider returned a mismatched hybrid vector count."
                )

            for chunk, embedding in zip(batch, embeddings, strict=True):
                self._validate_dense_vector(embedding.dense)
                records.append(
                    self._record_from_chunk(
                        chunk=chunk,
                        vector=embedding.dense,
                        sparse_vector_indices=embedding.sparse.indices,
                        sparse_vector_values=embedding.sparse.values,
                    )
                )
        return records

    def _validate_dense_vector(self, vector: list[float]) -> None:
        if len(vector) != self.embedding_provider.dimension:
            raise ValueError(
                "Embedding provider returned vector dimension "
                f"{len(vector)}, expected {self.embedding_provider.dimension}."
            )

    def _record_from_chunk(
        self,
        *,
        chunk: Chunk,
        vector: list[float],
        sparse_vector_indices: list[int] | None = None,
        sparse_vector_values: list[float] | None = None,
    ) -> IndexRecord:
        metadata = {
            **chunk.metadata,
            **access_to_payload(chunk.access),
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "chunk_type": chunk.chunk_type.value,
            "chunk_index": chunk.chunk_index,
            "section_title": chunk.section_title,
            "section_path": chunk.section_path,
            "source_uri": chunk.citations[0].source_uri if chunk.citations else None,
            "content_hash": chunk.content_hash,
            "neighbor_chunk_ids": chunk.neighbor_chunk_ids,
            "sparse_vector_size": len(sparse_vector_indices or []),
        }
        return IndexRecord(
            record_id=f"idx_{chunk.chunk_id}",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            parent_chunk_id=chunk.parent_chunk_id,
            chunk_type=chunk.chunk_type,
            text=chunk.text,
            contextual_text=chunk.contextual_text,
            vector=vector,
            sparse_terms={},
            sparse_vector_indices=list(sparse_vector_indices or []),
            sparse_vector_values=list(sparse_vector_values or []),
            metadata=metadata,
            access=chunk.access,
            citations=chunk.citations,
            content_hash=chunk.content_hash,
            embedding_model=self.embedding_provider.model_name,
            embedding_dimension=self.embedding_provider.dimension,
            index_version=self.config.index_version,
        )

    def _upsert(self, records: list[IndexRecord]) -> None:
        self.vector_store.upsert(records)


def _document_hash(chunking_result: ChunkingResult) -> str:
    values = [
        chunking_result.document_id,
        *(chunk.content_hash for chunk in sorted(chunking_result.chunks, key=lambda item: item.chunk_id)),
    ]
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _source_uri(chunks: list[Chunk]) -> str | None:
    for chunk in chunks:
        for citation in chunk.citations:
            if citation.source_uri:
                return citation.source_uri
    return None


def _manifest_config_matches(manifest: IndexManifest, config: IndexingConfig) -> bool:
    return (
        manifest.metadata.get("dense_vector_name") == config.dense_vector_name
        and manifest.metadata.get("sparse_vector_name") == config.sparse_vector_name
        and manifest.metadata.get("sparse_top_n") == config.sparse_top_n
    )
