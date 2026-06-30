from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from app.domain.schemas import AccessControl, RawDocument


DEFAULT_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".txt",
    ".csv",
}


class LocalFileConnector:
    def __init__(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
        supported_extensions: set[str] | None = None,
        access: AccessControl | None = None,
        access_by_filename: dict[str, AccessControl] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.recursive = recursive
        self.supported_extensions = {
            ext.lower() for ext in (supported_extensions or DEFAULT_SUPPORTED_EXTENSIONS)
        }
        self.access = access or AccessControl()
        self.access_by_filename: dict[str, AccessControl] = {}
        for filename, file_access in (access_by_filename or {}).items():
            self.access_by_filename[filename] = file_access
            self.access_by_filename[filename.lower()] = file_access

    def iter_documents(self) -> Iterable[RawDocument]:
        if self.root.is_file():
            candidates = [self.root]
        else:
            pattern = "**/*" if self.recursive else "*"
            candidates = [path for path in self.root.glob(pattern) if path.is_file()]

        for path in sorted(candidates):
            if path.suffix.lower() not in self.supported_extensions:
                continue
            yield self._to_raw_document(path)

    def _to_raw_document(self, path: Path) -> RawDocument:
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        return RawDocument(
            path=path,
            source_uri=path.as_uri(),
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            mime_type=mime_type,
            size_bytes=stat.st_size,
            content_hash=_sha256_file(path),
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            access=self._access_for(path),
            metadata={
                "connector": "local_file",
                "root": str(self.root),
            },
        )

    def _access_for(self, path: Path) -> AccessControl:
        return (
            self.access_by_filename.get(path.name)
            or self.access_by_filename.get(path.name.lower())
            or self.access_by_filename.get(path.stem)
            or self.access_by_filename.get(path.stem.lower())
            or self.access
        )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
