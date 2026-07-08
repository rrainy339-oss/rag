from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Any

from app.indexing.schemas import IndexBackend, IndexManifest, IndexingStatus


class CollectionManifestRepository:
    """Persist lightweight collection configuration for runtime startup."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_manifests (
                    collection_name TEXT PRIMARY KEY,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    index_version TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    dense_vector_name TEXT NOT NULL,
                    sparse_vector_name TEXT NOT NULL,
                    sparse_top_n INTEGER,
                    manifest_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get(self, collection_name: str) -> IndexManifest | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT manifest_json FROM collection_manifests
                WHERE collection_name = ?
                """,
                (collection_name,),
            ).fetchone()
        if row is None:
            return None
        return IndexManifest.model_validate_json(str(row["manifest_json"]))

    def save(self, collection_name: str, manifest: IndexManifest) -> IndexManifest:
        current = self.get(collection_name)
        if current is not None and _manifest_runtime_config(current) == _manifest_runtime_config(
            manifest
        ):
            return current

        row = _manifest_row(collection_name, manifest)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO collection_manifests (
                    collection_name, embedding_model, embedding_dimension,
                    index_version, backend, dense_vector_name, sparse_vector_name,
                    sparse_top_n, manifest_json, updated_at
                )
                VALUES (
                    :collection_name, :embedding_model, :embedding_dimension,
                    :index_version, :backend, :dense_vector_name, :sparse_vector_name,
                    :sparse_top_n, :manifest_json, :updated_at
                )
                ON CONFLICT(collection_name) DO UPDATE SET
                    embedding_model = excluded.embedding_model,
                    embedding_dimension = excluded.embedding_dimension,
                    index_version = excluded.index_version,
                    backend = excluded.backend,
                    dense_vector_name = excluded.dense_vector_name,
                    sparse_vector_name = excluded.sparse_vector_name,
                    sparse_top_n = excluded.sparse_top_n,
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                row,
            )
        return manifest

    def ensure(
        self,
        collection_name: str,
        *,
        embedding_model: str,
        embedding_dimension: int,
        dense_vector_name: str,
        sparse_vector_name: str,
        sparse_top_n: int | None,
        index_version: str = "idx-v1",
    ) -> IndexManifest:
        current = self.get(collection_name)
        if current is not None:
            return current
        manifest = build_collection_manifest(
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            sparse_top_n=sparse_top_n,
            index_version=index_version,
        )
        return self.save(collection_name, manifest)

    def signature(self, collection_name: str) -> tuple[Any, ...] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT embedding_model, embedding_dimension, index_version, backend,
                       dense_vector_name, sparse_vector_name, sparse_top_n, updated_at
                FROM collection_manifests
                WHERE collection_name = ?
                """,
                (collection_name,),
            ).fetchone()
        if row is None:
            return None
        return (
            row["embedding_model"],
            int(row["embedding_dimension"]),
            row["index_version"],
            row["backend"],
            row["dense_vector_name"],
            row["sparse_vector_name"],
            row["sparse_top_n"],
            row["updated_at"],
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


def build_collection_manifest(
    *,
    embedding_model: str,
    embedding_dimension: int,
    dense_vector_name: str,
    sparse_vector_name: str,
    sparse_top_n: int | None,
    index_version: str = "idx-v1",
) -> IndexManifest:
    return IndexManifest(
        document_id="default",
        source_uri=None,
        document_hash="collection-runtime-config",
        chunk_hashes={},
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        index_version=index_version,
        backend=IndexBackend.QDRANT_HYBRID,
        status=IndexingStatus.INDEXED,
        metadata={
            "combined": True,
            "dense_vector_name": dense_vector_name,
            "sparse_vector_name": sparse_vector_name,
            "sparse_top_n": sparse_top_n,
        },
    )


def _manifest_row(
    collection_name: str,
    manifest: IndexManifest,
) -> dict[str, object | None]:
    dense_vector_name = str(manifest.metadata.get("dense_vector_name") or "dense")
    sparse_vector_name = str(manifest.metadata.get("sparse_vector_name") or "sparse")
    sparse_top_n = manifest.metadata.get("sparse_top_n")
    return {
        "collection_name": collection_name,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "index_version": manifest.index_version,
        "backend": manifest.backend.value,
        "dense_vector_name": dense_vector_name,
        "sparse_vector_name": sparse_vector_name,
        "sparse_top_n": int(sparse_top_n) if sparse_top_n is not None else None,
        "manifest_json": manifest.model_dump_json(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _manifest_runtime_config(manifest: IndexManifest) -> tuple[object, ...]:
    return (
        manifest.embedding_model,
        manifest.embedding_dimension,
        manifest.index_version,
        manifest.backend.value,
        manifest.metadata.get("dense_vector_name"),
        manifest.metadata.get("sparse_vector_name"),
        manifest.metadata.get("sparse_top_n"),
    )
