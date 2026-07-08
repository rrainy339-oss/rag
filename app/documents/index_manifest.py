from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.indexing.schemas import IndexManifest


@dataclass(frozen=True)
class DocumentIndexManifestRecord:
    document_id: str
    content_hash: str
    document_hash: str
    chunk_hashes: dict[str, str]
    parser_config_signature: str
    chunk_config_signature: str
    embedding_config_signature: str
    access_signature: str
    embedding_model: str
    embedding_dimension: int
    index_version: str
    backend: str
    dense_vector_name: str
    sparse_vector_name: str
    sparse_top_n: int | None
    chunk_count: int
    indexed_count: int
    active: bool
    indexed_at: datetime | None
    updated_at: datetime
    deleted_at: datetime | None
    manifest: IndexManifest


class DocumentIndexManifestRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_index_manifests (
                    document_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    chunk_hashes_json TEXT NOT NULL,
                    parser_config_signature TEXT NOT NULL,
                    chunk_config_signature TEXT NOT NULL,
                    embedding_config_signature TEXT NOT NULL,
                    access_signature TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    index_version TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    dense_vector_name TEXT NOT NULL,
                    sparse_vector_name TEXT NOT NULL,
                    sparse_top_n INTEGER,
                    chunk_count INTEGER NOT NULL,
                    indexed_count INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    indexed_at TEXT,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    manifest_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_doc_index_manifests_versions
                ON document_index_manifests(
                    active, embedding_model, embedding_dimension, index_version
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_doc_index_manifests_content_hash
                ON document_index_manifests(active, content_hash)
                """
            )

    def get(self, document_id: str) -> DocumentIndexManifestRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM document_index_manifests
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def save(
        self,
        *,
        manifest: IndexManifest,
        content_hash: str,
        parser_config_signature: str,
        chunk_config_signature: str,
        embedding_config_signature: str,
        access_signature: str,
        chunk_count: int,
        indexed_count: int,
    ) -> DocumentIndexManifestRecord:
        row = _manifest_row(
            manifest=manifest,
            content_hash=content_hash,
            parser_config_signature=parser_config_signature,
            chunk_config_signature=chunk_config_signature,
            embedding_config_signature=embedding_config_signature,
            access_signature=access_signature,
            chunk_count=chunk_count,
            indexed_count=indexed_count,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO document_index_manifests (
                    document_id, content_hash, document_hash, chunk_hashes_json,
                    parser_config_signature, chunk_config_signature,
                    embedding_config_signature, access_signature, embedding_model,
                    embedding_dimension, index_version, backend, dense_vector_name,
                    sparse_vector_name, sparse_top_n, chunk_count, indexed_count,
                    active, indexed_at, updated_at, deleted_at, manifest_json
                )
                VALUES (
                    :document_id, :content_hash, :document_hash, :chunk_hashes_json,
                    :parser_config_signature, :chunk_config_signature,
                    :embedding_config_signature, :access_signature, :embedding_model,
                    :embedding_dimension, :index_version, :backend, :dense_vector_name,
                    :sparse_vector_name, :sparse_top_n, :chunk_count, :indexed_count,
                    :active, :indexed_at, :updated_at, :deleted_at, :manifest_json
                )
                ON CONFLICT(document_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    document_hash = excluded.document_hash,
                    chunk_hashes_json = excluded.chunk_hashes_json,
                    parser_config_signature = excluded.parser_config_signature,
                    chunk_config_signature = excluded.chunk_config_signature,
                    embedding_config_signature = excluded.embedding_config_signature,
                    access_signature = excluded.access_signature,
                    embedding_model = excluded.embedding_model,
                    embedding_dimension = excluded.embedding_dimension,
                    index_version = excluded.index_version,
                    backend = excluded.backend,
                    dense_vector_name = excluded.dense_vector_name,
                    sparse_vector_name = excluded.sparse_vector_name,
                    sparse_top_n = excluded.sparse_top_n,
                    chunk_count = excluded.chunk_count,
                    indexed_count = excluded.indexed_count,
                    active = excluded.active,
                    indexed_at = excluded.indexed_at,
                    updated_at = excluded.updated_at,
                    deleted_at = excluded.deleted_at,
                    manifest_json = excluded.manifest_json
                """,
                row,
            )
        record = self.get(manifest.document_id)
        if record is None:
            raise KeyError(manifest.document_id)
        return record

    def update_access_signature(
        self,
        document_id: str,
        *,
        access_signature: str,
        manifest: IndexManifest | None = None,
    ) -> None:
        existing = self.get(document_id)
        if existing is None:
            return
        manifest_to_store = manifest or existing.manifest
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE document_index_manifests
                SET access_signature = ?,
                    manifest_json = ?,
                    updated_at = ?
                WHERE document_id = ?
                """,
                (
                    access_signature,
                    manifest_to_store.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                    document_id,
                ),
            )

    def mark_deleted(self, document_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE document_index_manifests
                SET active = 0, deleted_at = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (now, now, document_id),
            )

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _manifest_row(
    *,
    manifest: IndexManifest,
    content_hash: str,
    parser_config_signature: str,
    chunk_config_signature: str,
    embedding_config_signature: str,
    access_signature: str,
    chunk_count: int,
    indexed_count: int,
) -> dict[str, object | None]:
    now = datetime.now(timezone.utc).isoformat()
    dense_vector_name = str(manifest.metadata.get("dense_vector_name") or "dense")
    sparse_vector_name = str(manifest.metadata.get("sparse_vector_name") or "sparse")
    sparse_top_n = manifest.metadata.get("sparse_top_n")
    return {
        "document_id": manifest.document_id,
        "content_hash": content_hash,
        "document_hash": manifest.document_hash,
        "chunk_hashes_json": json.dumps(
            manifest.chunk_hashes,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "parser_config_signature": parser_config_signature,
        "chunk_config_signature": chunk_config_signature,
        "embedding_config_signature": embedding_config_signature,
        "access_signature": access_signature,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "index_version": manifest.index_version,
        "backend": manifest.backend.value,
        "dense_vector_name": dense_vector_name,
        "sparse_vector_name": sparse_vector_name,
        "sparse_top_n": int(sparse_top_n) if sparse_top_n is not None else None,
        "chunk_count": chunk_count,
        "indexed_count": indexed_count,
        "active": 1,
        "indexed_at": manifest.indexed_at.isoformat(),
        "updated_at": now,
        "deleted_at": None,
        "manifest_json": manifest.model_dump_json(),
    }


def _row_to_record(row: sqlite3.Row) -> DocumentIndexManifestRecord:
    data = dict(row)
    return DocumentIndexManifestRecord(
        document_id=str(data["document_id"]),
        content_hash=str(data["content_hash"]),
        document_hash=str(data["document_hash"]),
        chunk_hashes=json.loads(str(data["chunk_hashes_json"] or "{}")),
        parser_config_signature=str(data["parser_config_signature"]),
        chunk_config_signature=str(data["chunk_config_signature"]),
        embedding_config_signature=str(data["embedding_config_signature"]),
        access_signature=str(data["access_signature"]),
        embedding_model=str(data["embedding_model"]),
        embedding_dimension=int(data["embedding_dimension"]),
        index_version=str(data["index_version"]),
        backend=str(data["backend"]),
        dense_vector_name=str(data["dense_vector_name"]),
        sparse_vector_name=str(data["sparse_vector_name"]),
        sparse_top_n=(
            int(data["sparse_top_n"]) if data["sparse_top_n"] is not None else None
        ),
        chunk_count=int(data["chunk_count"]),
        indexed_count=int(data["indexed_count"]),
        active=bool(data["active"]),
        indexed_at=_parse_datetime(data.get("indexed_at")),
        updated_at=_parse_datetime(data.get("updated_at")) or datetime.now(timezone.utc),
        deleted_at=_parse_datetime(data.get("deleted_at")),
        manifest=IndexManifest.model_validate_json(str(data["manifest_json"])),
    )


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return datetime.fromisoformat(text)
