from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.domain.schemas import RawDocument


class DocumentConnector(Protocol):
    def iter_documents(self) -> Iterable[RawDocument]:
        """Yield raw documents from the connector source."""

