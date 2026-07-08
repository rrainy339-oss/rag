from __future__ import annotations

import hashlib
import json
from typing import Any

from app.api.settings import APISettings


SIGNATURE_VERSION = "sig-v1"


def stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"signature_version": SIGNATURE_VERSION, **payload},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def access_signature(
    *,
    tenant_id: str | None,
    owner_id: str | None,
    group_ids: list[str],
    principal_ids: list[str],
    classification: str | None,
) -> str:
    return stable_signature(
        {
            "kind": "access",
            "tenant_id": _clean(tenant_id),
            "owner_id": _clean(owner_id),
            "group_ids": sorted(_clean_many(group_ids)),
            "principal_ids": sorted(_clean_many(principal_ids)),
            "classification": _clean(classification),
        }
    )


def dedupe_key(*, content_hash: str, access_signature_value: str) -> str:
    return stable_signature(
        {
            "kind": "document_dedupe",
            "content_hash": content_hash,
            "access_signature": access_signature_value,
        }
    )


def parser_config_signature(settings: APISettings) -> str:
    return stable_signature(
        {
            "kind": "parser_config",
            "parser": "docling",
            "pdf_backend": settings.documents_pdf_backend,
            "do_ocr": settings.documents_do_ocr,
            "artifacts_path": (
                str(settings.documents_docling_artifacts_path)
                if settings.documents_docling_artifacts_path
                else None
            ),
        }
    )


def chunk_config_signature(settings: APISettings) -> str:
    return stable_signature(
        {
            "kind": "chunk_config",
            "chunker": "structure-aware-v1",
            "child_target_tokens": settings.documents_child_target_tokens,
            "child_max_tokens": settings.documents_child_max_tokens,
            "parent_target_tokens": settings.documents_parent_target_tokens,
            "parent_max_tokens": settings.documents_parent_max_tokens,
        }
    )


def embedding_config_signature(
    *,
    settings: APISettings,
    embedding_model: str,
    embedding_dimension: int,
) -> str:
    return stable_signature(
        {
            "kind": "embedding_config",
            "provider": settings.embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "bge_max_length": settings.bge_max_length,
            "sparse_top_n": settings.sparse_top_n,
            "dense_vector_name": settings.dense_vector_name,
            "sparse_vector_name": settings.sparse_vector_name,
        }
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_many(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return cleaned
