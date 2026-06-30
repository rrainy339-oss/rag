from __future__ import annotations

import os
from typing import Any

from app.retrieval.rerankers.base import RerankerAPIError, candidate_document_text
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate


class VoyageReranker:
    """Remote Voyage AI reranker using the REST API."""

    def __init__(
        self,
        *,
        model_name: str = "rerank-2.5",
        api_key: str | None = None,
        endpoint: str = "https://api.voyageai.com/v1/rerank",
        timeout_seconds: float = 60.0,
        truncation: bool = True,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise RerankerAPIError(
                "Voyage reranking requires VOYAGE_API_KEY or --reranker-api-key."
            )
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.truncation = truncation

    def rerank(
        self,
        analyzed_query: AnalyzedQuery,
        candidates: list[RetrievalCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        payload = {
            "model": self.model_name,
            "query": analyzed_query.query.query,
            "documents": [candidate_document_text(candidate) for candidate in candidates],
            "top_k": top_k,
            "truncation": self.truncation,
        }
        data = _post_json(
            self.endpoint,
            payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds=self.timeout_seconds,
        )
        return _rank_from_remote_results(
            candidates,
            data.get("results") or [],
            model_name=self.model_name,
        )


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RerankerAPIError(
            "Remote reranking requires httpx. Install it in the project virtual "
            "environment with `pip install -e \".[reranking]\"`."
        ) from exc

    try:
        response = httpx.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RerankerAPIError(f"Voyage rerank request failed: {exc}") from exc
    return response.json()


def _rank_from_remote_results(
    candidates: list[RetrievalCandidate],
    results: list[dict[str, Any]],
    *,
    model_name: str,
) -> list[RetrievalCandidate]:
    ranked: list[RetrievalCandidate] = []
    for rank, result in enumerate(results, start=1):
        candidate = candidates[int(result["index"])]
        score = float(result.get("relevance_score", result.get("score", 0.0)))
        candidate.rerank_score = score
        candidate.final_score = score
        candidate.rank = rank
        candidate.metadata["reranker"] = model_name
        candidate.metadata["pre_rerank_score"] = candidate.fused_score
        ranked.append(candidate)
    return ranked
