from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Iterable

from app.chunking.schemas import Chunk


class ChunkRepository:
    """SQLite-backed chunk lookup used by online retrieval.

    The repository keeps the full chunk JSON for fidelity while indexing the
    fields needed for document-level replacement and parent expansion.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    parent_chunk_id TEXT,
                    chunk_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id, chunk_index)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id
                ON chunks(parent_chunk_id)
                """
            )

    def get(self, chunk_id: str | None) -> Chunk | None:
        if chunk_id is None:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return Chunk.model_validate_json(str(row["payload"]))

    def list_by_document(self, document_id: str) -> list[Chunk]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [Chunk.model_validate_json(str(row["payload"])) for row in rows]

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Iterable[Chunk],
    ) -> int:
        rows = [_chunk_row(chunk) for chunk in chunks]
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )
            if rows:
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, parent_chunk_id, chunk_type,
                        chunk_index, content_hash, payload, updated_at
                    )
                    VALUES (
                        :chunk_id, :document_id, :parent_chunk_id, :chunk_type,
                        :chunk_index, :content_hash, :payload, :updated_at
                    )
                    """,
                    rows,
                )
        return len(rows)

    def replace_all(self, chunks: Iterable[Chunk]) -> int:
        rows = [_chunk_row(chunk) for chunk in chunks]
        with self._connection() as connection:
            connection.execute("DELETE FROM chunks")
            if rows:
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, parent_chunk_id, chunk_type,
                        chunk_index, content_hash, payload, updated_at
                    )
                    VALUES (
                        :chunk_id, :document_id, :parent_chunk_id, :chunk_type,
                        :chunk_index, :content_hash, :payload, :updated_at
                    )
                    """,
                    rows,
                )
        return len(rows)

    def delete_by_document(self, document_id: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )
            return int(cursor.rowcount)

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"] if row is not None else 0)

    def count_by_document(self, document_id: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM chunks
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

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


def _chunk_row(chunk: Chunk) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "chunk_type": chunk.chunk_type.value,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
        "payload": chunk.model_dump_json(),
        "updated_at": now,
    }
