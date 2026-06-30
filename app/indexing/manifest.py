from __future__ import annotations

import json
from pathlib import Path

from app.indexing.schemas import IndexManifest


class InMemoryManifestStore:
    def __init__(self) -> None:
        self._manifests: dict[str, IndexManifest] = {}

    def get(self, document_id: str) -> IndexManifest | None:
        return self._manifests.get(document_id)

    def save(self, manifest: IndexManifest) -> None:
        self._manifests[manifest.document_id] = manifest


class JsonManifestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, document_id: str) -> IndexManifest | None:
        data = self._read()
        payload = data.get(document_id)
        return IndexManifest.model_validate(payload) if payload else None

    def save(self, manifest: IndexManifest) -> None:
        data = self._read()
        data[manifest.document_id] = manifest.model_dump(mode="json")
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

