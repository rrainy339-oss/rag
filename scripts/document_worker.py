from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.settings import APISettings  # noqa: E402
from app.documents import DocumentService  # noqa: E402


_SHOULD_STOP = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run durable document ingestion jobs.")
    parser.add_argument(
        "--worker-id",
        default=f"document-worker-{os.getpid()}",
        help="Worker id written to document_jobs.locked_by.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds to sleep when no runnable job exists.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run at most one available job and exit.",
    )
    args = parser.parse_args(argv)

    _install_signal_handlers()
    service = DocumentService(APISettings.from_env())
    print(f"document worker started: {args.worker_id}", flush=True)

    while not _SHOULD_STOP:
        job = service.run_next_job(worker_id=args.worker_id)
        if job is None:
            if args.once:
                print("no runnable document job", flush=True)
                return 0
            time.sleep(max(0.1, args.interval))
            continue

        print(
            "job {job_id} document={document_id} status={status} "
            "stage={stage} progress={progress}".format(
                job_id=job.job_id,
                document_id=job.document_id,
                status=job.status.value,
                stage=job.stage.value,
                progress=job.progress,
            ),
            flush=True,
        )
        if args.once:
            return 0 if job.error_message is None else 1

    print("document worker stopped", flush=True)
    return 0


def _install_signal_handlers() -> None:
    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        global _SHOULD_STOP
        _SHOULD_STOP = True

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            signal.signal(signal_value, request_stop)


if __name__ == "__main__":
    raise SystemExit(main())
