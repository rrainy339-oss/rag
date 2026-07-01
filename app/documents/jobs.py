from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.documents.schemas import (
    DocumentJob,
    DocumentJobStage,
    DocumentJobStatus,
    DocumentJobType,
)


TERMINAL_JOB_STATUSES = {
    DocumentJobStatus.SUCCEEDED,
    DocumentJobStatus.FAILED,
    DocumentJobStatus.CANCELLED,
}


class DocumentJobRegistry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_jobs (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    progress INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    locked_by TEXT,
                    locked_at TEXT,
                    heartbeat_at TEXT,
                    next_run_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_jobs_next_run
                ON document_jobs(status, next_run_at, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_jobs_document_id
                ON document_jobs(document_id, created_at)
                """
            )

    def create(
        self,
        *,
        document_id: str,
        job_type: DocumentJobType,
        max_attempts: int = 3,
    ) -> DocumentJob:
        now = _now()
        job = DocumentJob(
            job_id=str(uuid4()),
            document_id=document_id,
            job_type=job_type,
            max_attempts=max_attempts,
            next_run_at=now,
            created_at=now,
            updated_at=now,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO document_jobs (
                    job_id, document_id, job_type, status, stage, attempt,
                    max_attempts, progress, cancel_requested, error_message,
                    locked_by, locked_at, heartbeat_at, next_run_at, created_at,
                    updated_at, started_at, finished_at
                )
                VALUES (
                    :job_id, :document_id, :job_type, :status, :stage, :attempt,
                    :max_attempts, :progress, :cancel_requested, :error_message,
                    :locked_by, :locked_at, :heartbeat_at, :next_run_at,
                    :created_at, :updated_at, :started_at, :finished_at
                )
                """,
                _job_to_row(job),
            )
        return job

    def get(self, job_id: str) -> DocumentJob | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM document_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _row_to_job(row) if row is not None else None

    def latest_for_document(self, document_id: str) -> DocumentJob | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM document_jobs
                WHERE document_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return _row_to_job(row) if row is not None else None

    def list(
        self,
        *,
        document_id: str | None = None,
        include_terminal: bool = True,
        limit: int = 100,
    ) -> list[DocumentJob]:
        query = "SELECT * FROM document_jobs"
        clauses: list[str] = []
        params: list[object] = []
        if document_id:
            clauses.append("document_id = ?")
            params.append(document_id)
        if not include_terminal:
            clauses.append("status NOT IN (?, ?, ?)")
            params.extend(status.value for status in TERMINAL_JOB_STATUSES)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [_row_to_job(row) for row in rows]

    def claim_next(self, *, worker_id: str) -> DocumentJob | None:
        now = _now()
        runnable = (DocumentJobStatus.QUEUED.value, DocumentJobStatus.RETRYING.value)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM document_jobs
                WHERE status IN (?, ?)
                  AND cancel_requested = 0
                  AND next_run_at <= ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (*runnable, now.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return self.claim(_row_to_job(row).job_id, worker_id=worker_id)

    def claim(self, job_id: str, *, worker_id: str) -> DocumentJob | None:
        now = _now()
        runnable = (DocumentJobStatus.QUEUED.value, DocumentJobStatus.RETRYING.value)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM document_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            job = _row_to_job(row)
            started_at = job.started_at or now
            cursor = connection.execute(
                """
                UPDATE document_jobs
                SET status = ?, attempt = attempt + 1, locked_by = ?,
                    locked_at = ?, heartbeat_at = ?, started_at = ?,
                    updated_at = ?, error_message = NULL
                WHERE job_id = ?
                  AND status IN (?, ?)
                  AND cancel_requested = 0
                  AND next_run_at <= ?
                """,
                (
                    DocumentJobStatus.RUNNING.value,
                    worker_id,
                    now.isoformat(),
                    now.isoformat(),
                    started_at.isoformat(),
                    now.isoformat(),
                    job.job_id,
                    *runnable,
                    now.isoformat(),
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(job_id)

    def set_stage(
        self,
        job_id: str,
        *,
        stage: DocumentJobStage,
        progress: int,
    ) -> DocumentJob:
        return self.update(
            job_id,
            stage=stage.value,
            progress=max(0, min(100, progress)),
            heartbeat_at=_now().isoformat(),
        )

    def heartbeat(self, job_id: str) -> DocumentJob:
        return self.update(job_id, heartbeat_at=_now().isoformat())

    def request_cancel(self, job_id: str) -> DocumentJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        if job.status in {DocumentJobStatus.QUEUED, DocumentJobStatus.RETRYING}:
            return self.mark_cancelled(job_id)
        return self.update(
            job_id,
            status=DocumentJobStatus.CANCELLING.value,
            cancel_requested=1,
        )

    def retry(self, job_id: str) -> DocumentJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in {DocumentJobStatus.FAILED, DocumentJobStatus.CANCELLED}:
            raise ValueError("Only failed or cancelled jobs can be retried.")
        return self.update(
            job_id,
            status=DocumentJobStatus.QUEUED.value,
            stage=DocumentJobStage.QUEUED.value,
            progress=0,
            cancel_requested=0,
            error_message=None,
            locked_by=None,
            locked_at=None,
            heartbeat_at=None,
            next_run_at=_now().isoformat(),
            finished_at=None,
        )

    def mark_succeeded(self, job_id: str) -> DocumentJob:
        now = _now()
        return self.update(
            job_id,
            status=DocumentJobStatus.SUCCEEDED.value,
            stage=DocumentJobStage.COMPLETED.value,
            progress=100,
            locked_by=None,
            locked_at=None,
            heartbeat_at=now.isoformat(),
            finished_at=now.isoformat(),
            cancel_requested=0,
        )

    def mark_cancelled(self, job_id: str) -> DocumentJob:
        now = _now()
        return self.update(
            job_id,
            status=DocumentJobStatus.CANCELLED.value,
            stage=DocumentJobStage.CANCELLED.value,
            locked_by=None,
            locked_at=None,
            heartbeat_at=now.isoformat(),
            finished_at=now.isoformat(),
            cancel_requested=1,
        )

    def mark_failed(
        self,
        job_id: str,
        *,
        error_message: str,
        retry_delay_seconds: int,
    ) -> DocumentJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)

        now = _now()
        if job.attempt < job.max_attempts:
            return self.update(
                job_id,
                status=DocumentJobStatus.RETRYING.value,
                stage=DocumentJobStage.FAILED.value,
                error_message=error_message,
                locked_by=None,
                locked_at=None,
                heartbeat_at=now.isoformat(),
                next_run_at=(now + timedelta(seconds=retry_delay_seconds)).isoformat(),
            )
        return self.update(
            job_id,
            status=DocumentJobStatus.FAILED.value,
            stage=DocumentJobStage.FAILED.value,
            error_message=error_message,
            locked_by=None,
            locked_at=None,
            heartbeat_at=now.isoformat(),
            finished_at=now.isoformat(),
        )

    def update(self, job_id: str, **values: object) -> DocumentJob:
        if not values:
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job
        values["updated_at"] = _now().isoformat()
        assignments = ", ".join(f"{key} = :{key}" for key in values)
        params = {**values, "job_id": job_id}
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE document_jobs SET {assignments} WHERE job_id = :job_id",
                params,
            )
            if cursor.rowcount == 0:
                raise KeyError(job_id)
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

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


def _job_to_row(job: DocumentJob) -> dict[str, object | None]:
    data = job.model_dump()
    data["job_type"] = job.job_type.value
    data["status"] = job.status.value
    data["stage"] = job.stage.value
    data["cancel_requested"] = 1 if job.cancel_requested else 0
    for key in (
        "locked_at",
        "heartbeat_at",
        "next_run_at",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
    ):
        value = data.get(key)
        data[key] = value.isoformat() if isinstance(value, datetime) else value
    return data


def _row_to_job(row: sqlite3.Row) -> DocumentJob:
    data = dict(row)
    data["cancel_requested"] = bool(data.get("cancel_requested"))
    return DocumentJob.model_validate(data)


def _now() -> datetime:
    return datetime.now(timezone.utc)
