from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.documents.schemas import DocumentRecord, DocumentStatus


class DocumentRegistry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    tenant_id TEXT,
                    owner_id TEXT,
                    group_ids TEXT NOT NULL,
                    principal_ids TEXT NOT NULL,
                    classification TEXT,
                    status TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    access_signature TEXT,
                    dedupe_key TEXT,
                    source_path TEXT NOT NULL,
                    parsed_path TEXT,
                    chunks_path TEXT,
                    index_path TEXT,
                    error_message TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    indexed_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            _ensure_column(connection, "documents", "access_signature", "TEXT")
            _ensure_column(connection, "documents", "dedupe_key", "TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_dedupe_key
                ON documents(dedupe_key, status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                ON documents(tenant_id, content_hash, status)
                """
            )

    def create(self, record: DocumentRecord) -> DocumentRecord:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, filename, title, tenant_id, owner_id, group_ids,
                    principal_ids, classification, status, content_type, size_bytes,
                    content_hash, access_signature, dedupe_key, source_path,
                    parsed_path, chunks_path, index_path, error_message, chunk_count,
                    indexed_count, created_at, updated_at
                )
                VALUES (
                    :document_id, :filename, :title, :tenant_id, :owner_id,
                    :group_ids, :principal_ids, :classification, :status,
                    :content_type, :size_bytes, :content_hash, :access_signature,
                    :dedupe_key, :source_path, :parsed_path, :chunks_path,
                    :index_path, :error_message, :chunk_count, :indexed_count,
                    :created_at, :updated_at
                )
                """,
                _record_to_row(record),
            )
        return record

    def find_by_dedupe_key(self, dedupe_key: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE dedupe_key = ?
                  AND status NOT IN (?, ?)
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (
                    dedupe_key,
                    DocumentStatus.DELETED.value,
                    DocumentStatus.FAILED.value,
                ),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def find_by_content_hash(self, content_hash: str) -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE content_hash = ?
                  AND status NOT IN (?, ?)
                ORDER BY created_at ASC
                """,
                (
                    content_hash,
                    DocumentStatus.DELETED.value,
                    DocumentStatus.FAILED.value,
                ),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(self, *, include_deleted: bool = False) -> list[DocumentRecord]:
        query = "SELECT * FROM documents"
        params: tuple[Any, ...] = ()
        if not include_deleted:
            query += " WHERE status != ?"
            params = (DocumentStatus.DELETED.value,)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
    ) -> DocumentRecord:
        return self.update(
            document_id,
            status=status.value,
            error_message=error_message,
        )

    def update_artifacts(
        self,
        document_id: str,
        *,
        parsed_path: str | Path | None = None,
        chunks_path: str | Path | None = None,
        index_path: str | Path | None = None,
        chunk_count: int | None = None,
        indexed_count: int | None = None,
        status: DocumentStatus | None = None,
        error_message: str | None = None,
    ) -> DocumentRecord:
        values: dict[str, object | None] = {
            "parsed_path": str(parsed_path) if parsed_path is not None else None,
            "chunks_path": str(chunks_path) if chunks_path is not None else None,
            "index_path": str(index_path) if index_path is not None else None,
            "chunk_count": chunk_count,
            "indexed_count": indexed_count,
            "error_message": error_message,
        }
        if status is not None:
            values["status"] = status.value
        return self.update(
            document_id,
            **{key: value for key, value in values.items() if value is not None},
        )

    def update_permissions(
        self,
        document_id: str,
        *,
        tenant_id: str | None,
        owner_id: str | None,
        group_ids: list[str],
        principal_ids: list[str],
        classification: str | None,
        access_signature: str | None = None,
        dedupe_key: str | None = None,
    ) -> DocumentRecord:
        return self.update(
            document_id,
            tenant_id=tenant_id,
            owner_id=owner_id,
            group_ids=json.dumps(group_ids, ensure_ascii=False),
            principal_ids=json.dumps(principal_ids, ensure_ascii=False),
            classification=classification,
            access_signature=access_signature,
            dedupe_key=dedupe_key,
        )

    def update(self, document_id: str, **values: object) -> DocumentRecord:
        if not values:
            record = self.get(document_id)
            if record is None:
                raise KeyError(document_id)
            return record

        values["updated_at"] = datetime.now(timezone.utc).isoformat()
        assignments = ", ".join(f"{key} = :{key}" for key in values)
        params = {**values, "document_id": document_id}
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE documents SET {assignments} WHERE document_id = :document_id",
                params,
            )
            if cursor.rowcount == 0:
                raise KeyError(document_id)
        record = self.get(document_id)
        if record is None:
            raise KeyError(document_id)
        return record

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


def _record_to_row(record: DocumentRecord) -> dict[str, object | None]:
    data = record.model_dump()
    data["status"] = record.status.value
    data["group_ids"] = json.dumps(record.group_ids, ensure_ascii=False)
    data["principal_ids"] = json.dumps(record.principal_ids, ensure_ascii=False)
    data["created_at"] = record.created_at.isoformat()
    data["updated_at"] = record.updated_at.isoformat()
    return data


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    data = dict(row)
    data["group_ids"] = _json_list(data.get("group_ids"))
    data["principal_ids"] = _json_list(data.get("principal_ids"))
    return DocumentRecord.model_validate(data)


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded if str(item).strip()]


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )
