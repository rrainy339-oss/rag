from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.answering.providers import (  # noqa: E402
    LLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)


DEFAULT_DATASET = Path("eval/datasets/xh202609_rag_eval.json")
DEFAULT_RUNS_DIR = Path("artifacts/eval/runs")

REFUSAL_PATTERNS = [
    "\u6587\u6863\u6ca1\u6709\u63d0\u5230",
    "\u6587\u6863\u672a\u63d0\u5230",
    "\u672a\u63d0\u5230",
    "\u6ca1\u6709\u63d0\u5230",
    "\u672a\u516c\u5e03",
    "\u6ca1\u6709\u516c\u5e03",
    "\u672a\u89c4\u5b9a",
    "\u6ca1\u6709\u89c4\u5b9a",
    "\u8d44\u6599\u4e0d\u8db3",
    "\u4fe1\u606f\u4e0d\u8db3",
    "\u65e0\u6cd5\u786e\u5b9a",
    "\u65e0\u6cd5\u56de\u7b54",
    "\u4e0d\u80fd\u786e\u5b9a",
    "not mentioned",
    "not specified",
    "insufficient",
]


@dataclass(frozen=True)
class RankedHit:
    rank: int
    kind: str
    context_id: str | None
    source_chunk_id: str | None
    chunk_id: str | None
    section_path: list[str]
    text: str


@dataclass(frozen=True)
class EvidenceHit:
    evidence_id: str
    rank: int | None
    section_rank: int | None
    value: float


def main() -> int:
    args = _parse_args()
    dataset = _load_json(args.dataset)
    output_dir = _resolve_output_dir(args, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    run_records = (
        list(_load_result_records(args.results))
        if args.results is not None
        else _run_dataset_against_api(dataset, args)
    )

    judge = _build_judge(args)
    scores: list[dict[str, Any]] = []
    for item in dataset["items"]:
        record = _record_for_item(run_records, item["id"])
        score = _score_item(item, record, dataset, judge, args)
        scores.append(score)
        print(
            f"{item['id']}: {score['final_score']:.1f}/100 "
            f"({', '.join(score['failure_reasons']) or 'ok'})"
        )

    summary = _summarize(dataset, scores, args, elapsed_seconds=time.perf_counter() - started)
    _write_outputs(output_dir, dataset, run_records, scores, summary)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and score a RAG evaluation dataset. The script can either call "
            "a running FastAPI RAG service or score a saved results.jsonl file."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--results",
        type=Path,
        help="Existing results.jsonl to score. If omitted, --api-base-url is required.",
    )

    api = parser.add_argument_group("RAG API run options")
    api.add_argument("--api-base-url", default=None, help="Example: http://127.0.0.1:8000")
    api.add_argument("--api-timeout", type=float, default=240.0)
    api.add_argument("--token", help="Bearer token for the RAG API.")
    api.add_argument("--username", help="Login username if token is not supplied.")
    api.add_argument("--password", help="Login password if token is not supplied.")
    api.add_argument("--auth-endpoint", default="/api/auth/chat/login")
    api.add_argument("--top-k", type=int, default=None)
    api.add_argument("--retrieve-only", action="store_true")
    api.add_argument("--rag-llm-provider", choices=["mock", "ollama", "openai-compatible"])
    api.add_argument("--rag-llm-base-url")
    api.add_argument("--rag-model")
    api.add_argument("--rag-api-key")
    api.add_argument("--rag-api-key-env", default="OPENAI_API_KEY")
    api.add_argument("--rag-temperature", type=float)
    api.add_argument("--rag-top-p", type=float)
    api.add_argument("--rag-max-tokens", type=int)

    judge = parser.add_argument_group("judge options")
    judge.add_argument(
        "--judge-provider",
        choices=["none", "ollama", "openai-compatible"],
        default="none",
        help="Use none for deterministic heuristic answer scoring.",
    )
    judge.add_argument("--judge-base-url", default="http://localhost:11434")
    judge.add_argument("--judge-model")
    judge.add_argument("--judge-api-key")
    judge.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    judge.add_argument("--judge-timeout", type=float, default=120.0)
    judge.add_argument("--judge-max-tokens", type=int, default=1200)
    judge.add_argument("--require-judge", action="store_true")

    scoring = parser.add_argument_group("scoring options")
    scoring.add_argument(
        "--citation-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether answerable items require citations for citation score.",
    )
    scoring.add_argument("--fail-on-api-error", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace, dataset: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RUNS_DIR / f"{dataset.get('dataset_id', 'rag_eval')}_{timestamp}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return data


def _load_result_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number} was not a JSON object.")
            records.append(parsed)
    return records


def _run_dataset_against_api(
    dataset: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.api_base_url:
        raise ValueError("Either --results or --api-base-url is required.")

    token = args.token or _login_for_token(args)
    records: list[dict[str, Any]] = []
    for item in dataset["items"]:
        record = _run_item_against_api(item, args, token)
        records.append(record)
        status = "error" if record.get("error") else "ok"
        print(f"ran {item['id']}: {status}")
        if record.get("error") and args.fail_on_api_error:
            raise RuntimeError(f"{item['id']} failed: {record['error']}")
    return records


def _login_for_token(args: argparse.Namespace) -> str | None:
    if not args.username and not args.password:
        return None
    if not args.username or not args.password:
        raise ValueError("--username and --password must be supplied together.")
    response = _post_json(
        _join_url(args.api_base_url, args.auth_endpoint),
        {"username": args.username, "password": args.password},
        timeout=args.api_timeout,
        token=None,
    )
    token = response.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("Login response did not include access_token.")
    return token


def _run_item_against_api(
    item: dict[str, Any],
    args: argparse.Namespace,
    token: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "item_id": item["id"],
        "query": item["query"],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    request_body = _request_body_for_item(item, args)
    try:
        t0 = time.perf_counter()
        retrieval = _post_json(
            _join_url(args.api_base_url, "/api/retrieve"),
            request_body,
            timeout=args.api_timeout,
            token=token,
        )
        record["retrieval"] = retrieval
        record["retrieve_elapsed_seconds"] = round(time.perf_counter() - t0, 6)

        if not args.retrieve_only:
            t1 = time.perf_counter()
            chat_body = dict(request_body)
            chat_body.update(_chat_overrides(args))
            chat = _post_json(
                _join_url(args.api_base_url, "/api/chat"),
                chat_body,
                timeout=args.api_timeout,
                token=token,
            )
            record["chat"] = chat
            record["chat_elapsed_seconds"] = round(time.perf_counter() - t1, 6)
    except Exception as exc:  # noqa: BLE001 - preserve batch run progress.
        record["error"] = str(exc)
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
    return record


def _request_body_for_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {"query": item["query"]}
    if args.top_k is not None:
        body["top_k"] = args.top_k
    body.update(item.get("request_overrides") or {})
    return body


def _chat_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.rag_llm_provider:
        overrides["llm_provider"] = args.rag_llm_provider
    if args.rag_llm_base_url:
        overrides["base_url"] = args.rag_llm_base_url
    if args.rag_model:
        overrides["model"] = args.rag_model
    api_key = args.rag_api_key or (
        os.environ.get(args.rag_api_key_env) if args.rag_api_key_env else None
    )
    if api_key:
        overrides["api_key"] = api_key
    if args.rag_temperature is not None:
        overrides["temperature"] = args.rag_temperature
    if args.rag_top_p is not None:
        overrides["top_p"] = args.rag_top_p
    if args.rag_max_tokens is not None:
        overrides["max_tokens"] = args.rag_max_tokens
    return overrides


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    token: str | None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc.reason}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Response from {url} was not a JSON object.")
    return parsed


def _join_url(base_url: str | None, path: str) -> str:
    if not base_url:
        raise ValueError("--api-base-url is required.")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _build_judge(args: argparse.Namespace) -> LLMProvider | None:
    if args.judge_provider == "none":
        return None
    if not args.judge_model:
        raise ValueError("--judge-model is required when --judge-provider is enabled.")
    if args.judge_provider == "ollama":
        return OllamaProvider(
            base_url=args.judge_base_url,
            model_name=args.judge_model,
            temperature=0.0,
            top_p=1.0,
            timeout=args.judge_timeout,
        )
    api_key = args.judge_api_key or (
        os.environ.get(args.judge_api_key_env) if args.judge_api_key_env else None
    )
    return OpenAICompatibleProvider(
        base_url=args.judge_base_url,
        model_name=args.judge_model,
        api_key=api_key,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
        timeout=args.judge_timeout,
    )


def _record_for_item(records: list[dict[str, Any]], item_id: str) -> dict[str, Any]:
    for record in records:
        if record.get("item_id") == item_id or record.get("id") == item_id:
            return record
    return {"item_id": item_id, "error": "missing_result_record"}


def _score_item(
    item: dict[str, Any],
    record: dict[str, Any],
    dataset: dict[str, Any],
    judge: LLMProvider | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    weights = dataset["score_profiles"][item["score_profile"]]["weights"]
    if record.get("error"):
        return _zero_score(item, str(record["error"]), weights)

    ranked_hits = _ranked_hits(record)
    answer = _answer_text(record)
    citations = _citations(record)
    judge_result = _judge_item(item, record, answer, judge, args)

    retrieval = _score_retrieval(item, ranked_hits, weights.get("retrieval", 0))
    answer_component = _score_answer(item, answer, weights.get("answer", 0), judge_result)
    citation_component = _score_citations(
        item,
        citations,
        record,
        weights.get("citation", 0),
        judge_result,
        args,
    )
    reliability = _score_reliability(
        item,
        answer,
        weights.get("reliability", 0),
        judge_result,
    )

    components = {
        "retrieval": retrieval["score"],
        "answer": answer_component["score"],
        "citation": citation_component["score"],
        "reliability": reliability["score"],
    }
    final_score = sum(components.values())
    failure_reasons = _failure_reasons(
        item,
        retrieval,
        answer_component,
        citation_component,
        reliability,
        judge_result,
        record,
    )
    return {
        "item_id": item["id"],
        "query": item["query"],
        "query_type": item["query_type"],
        "difficulty": item["difficulty"],
        "should_answer": item["should_answer"],
        "final_score": round(final_score, 4),
        "components": {key: round(value, 4) for key, value in components.items()},
        "retrieval": retrieval,
        "answer": answer_component,
        "citation": citation_component,
        "reliability": reliability,
        "judge": judge_result,
        "failure_reasons": failure_reasons,
        "error": record.get("error"),
    }


def _zero_score(
    item: dict[str, Any],
    error: str,
    weights: dict[str, Any],
) -> dict[str, Any]:
    components = {
        "retrieval": 0.0,
        "answer": 0.0,
        "citation": 0.0,
        "reliability": 0.0,
    }
    return {
        "item_id": item["id"],
        "query": item["query"],
        "query_type": item["query_type"],
        "difficulty": item["difficulty"],
        "should_answer": item["should_answer"],
        "final_score": 0.0,
        "components": components,
        "retrieval": {
            "score": 0.0,
            "weight": weights.get("retrieval", 0),
            "average_hit_value": 0.0,
            "evidence_hits": [
                {
                    "evidence_id": str(evidence.get("id", "")),
                    "rank": None,
                    "section_rank": None,
                    "value": 0.0,
                }
                for evidence in item.get("evidence", [])
                if isinstance(evidence, dict)
            ],
            "first_gold_rank": None,
            "ndcg_at_10": 0.0,
            "context_precision_at_5": 0.0,
        },
        "answer": {
            "score": 0.0,
            "weight": weights.get("answer", 0),
            "coverage": 0.0,
            "method": "not_scored",
            "point_scores": {},
        },
        "citation": {
            "score": 0.0,
            "weight": weights.get("citation", 0),
            "support_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "supported_points": [],
            "citation_count": 0,
            "method": "not_scored",
        },
        "reliability": {
            "score": 0.0,
            "weight": weights.get("reliability", 0),
            "value": 0.0,
            "forbidden_claims_present": [],
            "judge_forbidden_claims_present": [],
            "unsupported_claims": [],
            "refusal_detected": False,
            "refusal_correct": None,
        },
        "judge": None,
        "failure_reasons": ["format_error"],
        "error": error,
    }


def _ranked_hits(record: dict[str, Any]) -> list[RankedHit]:
    hits: list[RankedHit] = []
    retrieval = record.get("retrieval") if isinstance(record.get("retrieval"), dict) else {}
    chat = record.get("chat") if isinstance(record.get("chat"), dict) else {}

    contexts = _as_list(retrieval.get("contexts")) or _as_list(chat.get("contexts"))
    for index, context in enumerate(contexts, start=1):
        if not isinstance(context, dict):
            continue
        metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
        hits.append(
            RankedHit(
                rank=index,
                kind="context",
                context_id=_str_or_none(context.get("context_id")),
                source_chunk_id=_str_or_none(context.get("source_chunk_id")),
                chunk_id=None,
                section_path=_string_list(context.get("section_path")),
                text=_hit_text(context),
            )
        )
        source_rank = metadata.get("rank")
        if isinstance(source_rank, int):
            hits[-1]  # keep static analyzers quiet; source rank is reported elsewhere.

    candidates = _as_list(retrieval.get("candidates"))
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        rank = candidate.get("rank") if isinstance(candidate.get("rank"), int) else index
        hits.append(
            RankedHit(
                rank=rank,
                kind="candidate",
                context_id=None,
                source_chunk_id=None,
                chunk_id=_str_or_none(candidate.get("chunk_id")),
                section_path=_string_list(
                    (candidate.get("metadata") or {}).get("section_path")
                    if isinstance(candidate.get("metadata"), dict)
                    else None
                ),
                text=_hit_text(candidate),
            )
        )
    return hits


def _hit_text(payload: dict[str, Any]) -> str:
    parts = [
        payload.get("text"),
        payload.get("contextual_text"),
        payload.get("quote"),
    ]
    return "\n".join(str(part) for part in parts if isinstance(part, str))


def _score_retrieval(
    item: dict[str, Any],
    hits: list[RankedHit],
    weight: float,
) -> dict[str, Any]:
    evidence_hits: list[EvidenceHit] = []
    for evidence in item.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        first_rank = _first_gold_rank(evidence, hits)
        section_rank = _first_section_rank(evidence, hits)
        value = _hit_value(first_rank, section_rank)
        evidence_hits.append(
            EvidenceHit(
                evidence_id=str(evidence.get("id", "")),
                rank=first_rank,
                section_rank=section_rank,
                value=value,
            )
        )

    if not item.get("should_answer", True):
        score = weight
        average_value = 1.0
    elif evidence_hits:
        average_value = sum(hit.value for hit in evidence_hits) / len(evidence_hits)
        score = weight * average_value
    else:
        average_value = 0.0
        score = 0.0

    ranks = [hit.rank for hit in evidence_hits if hit.rank is not None]
    return {
        "score": round(score, 4),
        "weight": weight,
        "average_hit_value": round(average_value, 4),
        "evidence_hits": [
            {
                "evidence_id": hit.evidence_id,
                "rank": hit.rank,
                "section_rank": hit.section_rank,
                "value": hit.value,
            }
            for hit in evidence_hits
        ],
        "first_gold_rank": min(ranks) if ranks else None,
        "ndcg_at_10": round(_ndcg_at_10(item, hits), 6),
        "context_precision_at_5": round(_context_precision_at_5(item, hits), 6),
    }


def _first_gold_rank(evidence: dict[str, Any], hits: list[RankedHit]) -> int | None:
    gold_context_ids = set(_string_list(evidence.get("gold_context_ids")))
    gold_source_ids = set(_string_list(evidence.get("gold_source_chunk_ids")))
    ranks: list[int] = []
    for hit in hits:
        if hit.context_id and hit.context_id in gold_context_ids:
            ranks.append(hit.rank)
        if hit.source_chunk_id and hit.source_chunk_id in gold_source_ids:
            ranks.append(hit.rank)
        if hit.chunk_id and hit.chunk_id in gold_source_ids:
            ranks.append(hit.rank)
    return min(ranks) if ranks else None


def _first_section_rank(evidence: dict[str, Any], hits: list[RankedHit]) -> int | None:
    section = str(evidence.get("section") or "").strip()
    if not section:
        return None
    ranks: list[int] = []
    section_norm = _normalize_text(section)
    for hit in hits:
        path_norm = _normalize_text(" ".join(hit.section_path))
        if section_norm and section_norm in path_norm:
            ranks.append(hit.rank)
    return min(ranks) if ranks else None


def _hit_value(rank: int | None, section_rank: int | None) -> float:
    if rank is not None:
        if rank <= 3:
            return 1.0
        if rank <= 5:
            return 0.85
        if rank <= 10:
            return 0.67
    if section_rank is not None and section_rank <= 10:
        return 0.33
    return 0.0


def _ndcg_at_10(item: dict[str, Any], hits: list[RankedHit]) -> float:
    evidence = [ev for ev in item.get("evidence", []) if isinstance(ev, dict)]
    if not item.get("should_answer", True) or not evidence:
        return 0.0
    context_hits = [hit for hit in hits if hit.kind == "context" and hit.rank <= 10]
    relevant: list[int] = []
    matched_evidence: set[str] = set()
    for hit in sorted(context_hits, key=lambda h: h.rank):
        rel = 0
        for ev in evidence:
            ev_id = str(ev.get("id") or "")
            if ev_id in matched_evidence:
                continue
            if _hit_matches_evidence(hit, ev):
                rel = 1
                matched_evidence.add(ev_id)
                break
        relevant.append(rel)
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevant))
    ideal_count = min(len(evidence), 10)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def _context_precision_at_5(item: dict[str, Any], hits: list[RankedHit]) -> float:
    evidence = [ev for ev in item.get("evidence", []) if isinstance(ev, dict)]
    top_contexts = [hit for hit in hits if hit.kind == "context" and hit.rank <= 5]
    if not top_contexts:
        return 0.0
    relevant_count = 0
    for hit in top_contexts:
        if any(_hit_matches_evidence(hit, ev) for ev in evidence):
            relevant_count += 1
    return relevant_count / len(top_contexts)


def _hit_matches_evidence(hit: RankedHit, evidence: dict[str, Any]) -> bool:
    gold_context_ids = set(_string_list(evidence.get("gold_context_ids")))
    gold_source_ids = set(_string_list(evidence.get("gold_source_chunk_ids")))
    return bool(
        (hit.context_id and hit.context_id in gold_context_ids)
        or (hit.source_chunk_id and hit.source_chunk_id in gold_source_ids)
        or (hit.chunk_id and hit.chunk_id in gold_source_ids)
    )


def _score_answer(
    item: dict[str, Any],
    answer: str,
    weight: float,
    judge_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if judge_result and isinstance(judge_result.get("point_scores"), list):
        point_scores = _judge_point_scores(item, judge_result)
        method = "llm_judge"
    else:
        point_scores = _heuristic_point_scores(item, answer)
        method = "heuristic"
    weighted = _weighted_average_points(item, point_scores)
    return {
        "score": round(weight * weighted, 4),
        "weight": weight,
        "coverage": round(weighted, 6),
        "method": method,
        "point_scores": point_scores,
    }


def _judge_point_scores(
    item: dict[str, Any],
    judge_result: dict[str, Any],
) -> dict[str, float]:
    scores_by_id: dict[str, float] = {}
    for entry in judge_result.get("point_scores", []):
        if not isinstance(entry, dict):
            continue
        point_id = entry.get("point_id")
        score = entry.get("score")
        if isinstance(point_id, str) and isinstance(score, (int, float)):
            scores_by_id[point_id] = max(0.0, min(1.0, float(score)))
    return {
        str(point["id"]): scores_by_id.get(str(point["id"]), 0.0)
        for point in item.get("expected_points", [])
        if isinstance(point, dict)
    }


def _heuristic_point_scores(item: dict[str, Any], answer: str) -> dict[str, float]:
    answer_norm = _normalize_text(answer)
    is_refusal = _looks_like_refusal(answer)
    forbidden_present = _forbidden_claims_present(item, answer)
    scores: dict[str, float] = {}
    for point in item.get("expected_points", []):
        if not isinstance(point, dict):
            continue
        point_id = str(point["id"])
        point_text = str(point.get("text") or "")
        if not item.get("should_answer", True):
            if "不得" in point_text or "不应" in point_text:
                scores[point_id] = 1.0 if not forbidden_present else 0.0
            elif "未" in point_text or "\u6ca1\u6709" in point_text:
                scores[point_id] = 1.0 if is_refusal else 0.0
            else:
                scores[point_id] = 0.5 if is_refusal else 0.0
            continue

        similarity = _similarity(point_text, answer_norm)
        point_numbers = set(_numbers(point_text))
        if point_numbers and not point_numbers.issubset(set(_numbers(answer))):
            scores[point_id] = 0.0
        elif similarity >= 0.45:
            scores[point_id] = 1.0
        elif similarity >= 0.22:
            scores[point_id] = 0.5
        else:
            scores[point_id] = 0.0
    return scores


def _weighted_average_points(item: dict[str, Any], point_scores: dict[str, float]) -> float:
    total_weight = 0.0
    total_score = 0.0
    for point in item.get("expected_points", []):
        if not isinstance(point, dict):
            continue
        point_id = str(point["id"])
        weight = float(point.get("weight", 1.0))
        total_weight += weight
        total_score += weight * point_scores.get(point_id, 0.0)
    return total_score / total_weight if total_weight else 0.0


def _score_citations(
    item: dict[str, Any],
    citations: list[dict[str, Any]],
    record: dict[str, Any],
    weight: float,
    judge_result: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if weight <= 0 or not item.get("should_answer", True):
        return {
            "score": 0.0,
            "weight": weight,
            "support_rate": 1.0 if not item.get("should_answer", True) else 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "supported_points": [],
            "method": "not_applicable",
        }
    if not args.citation_required:
        return {
            "score": weight,
            "weight": weight,
            "support_rate": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "supported_points": [],
            "method": "disabled",
        }

    supported_points = _supported_points_by_citation(item, citations, record)
    expected_point_ids = [
        str(point["id"])
        for point in item.get("expected_points", [])
        if isinstance(point, dict)
    ]
    support_rate = (
        len(supported_points) / len(expected_point_ids) if expected_point_ids else 0.0
    )
    precision = _citation_precision(item, citations)
    return {
        "score": round(weight * support_rate, 4),
        "weight": weight,
        "support_rate": round(support_rate, 6),
        "precision": round(precision, 6),
        "recall": round(support_rate, 6),
        "supported_points": sorted(supported_points),
        "citation_count": len(citations),
        "method": "deterministic",
        "judge_citation_support": (
            judge_result.get("citation_support") if judge_result else None
        ),
    }


def _supported_points_by_citation(
    item: dict[str, Any],
    citations: list[dict[str, Any]],
    record: dict[str, Any],
) -> set[str]:
    citation_hits = [_citation_to_hit(citation) for citation in citations]
    supported: set[str] = set()
    for point in item.get("expected_points", []):
        if not isinstance(point, dict):
            continue
        for evidence in _evidence_for_point(item, point):
            if any(_hit_matches_evidence(hit, evidence) for hit in citation_hits):
                supported.add(str(point["id"]))
                break
            if any(_citation_quote_supports(citation, evidence) for citation in citations):
                supported.add(str(point["id"]))
                break
    if not supported and citations:
        # Some systems cite only Context labels. Try matching citation ids against chat contexts.
        cited_context_ids = {str(c.get("context_id")) for c in citations if c.get("context_id")}
        context_lookup = {
            str(context.get("context_id")): context
            for context in _chat_contexts(record)
            if isinstance(context, dict) and context.get("context_id")
        }
        derived = [_context_to_hit(context_lookup[cid]) for cid in cited_context_ids if cid in context_lookup]
        for point in item.get("expected_points", []):
            if not isinstance(point, dict):
                continue
            if any(
                _hit_matches_evidence(hit, evidence)
                for evidence in _evidence_for_point(item, point)
                for hit in derived
            ):
                supported.add(str(point["id"]))
    return supported


def _citation_precision(item: dict[str, Any], citations: list[dict[str, Any]]) -> float:
    if not citations:
        return 0.0
    evidence = [ev for ev in item.get("evidence", []) if isinstance(ev, dict)]
    supported = 0
    for citation in citations:
        hit = _citation_to_hit(citation)
        if any(_hit_matches_evidence(hit, ev) for ev in evidence) or any(
            _citation_quote_supports(citation, ev) for ev in evidence
        ):
            supported += 1
    return supported / len(citations)


def _evidence_for_point(item: dict[str, Any], point: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = set(_string_list(point.get("evidence_ids")))
    return [
        ev
        for ev in item.get("evidence", [])
        if isinstance(ev, dict) and str(ev.get("id")) in wanted
    ]


def _citation_to_hit(citation: dict[str, Any]) -> RankedHit:
    return RankedHit(
        rank=1,
        kind="citation",
        context_id=_str_or_none(citation.get("context_id")),
        source_chunk_id=_str_or_none(citation.get("source_chunk_id")),
        chunk_id=_str_or_none(citation.get("source_chunk_id")),
        section_path=_string_list(citation.get("section_path")),
        text=_hit_text(citation),
    )


def _context_to_hit(context: dict[str, Any]) -> RankedHit:
    return RankedHit(
        rank=1,
        kind="context",
        context_id=_str_or_none(context.get("context_id")),
        source_chunk_id=_str_or_none(context.get("source_chunk_id")),
        chunk_id=None,
        section_path=_string_list(context.get("section_path")),
        text=_hit_text(context),
    )


def _citation_quote_supports(citation: dict[str, Any], evidence: dict[str, Any]) -> bool:
    quote = _normalize_text(str(evidence.get("quote") or ""))
    citation_text = _normalize_text(_hit_text(citation))
    if not quote or not citation_text:
        return False
    return quote in citation_text or citation_text in quote


def _score_reliability(
    item: dict[str, Any],
    answer: str,
    weight: float,
    judge_result: dict[str, Any] | None,
) -> dict[str, Any]:
    forbidden_present = _forbidden_claims_present(item, answer)
    judge_forbidden = []
    unsupported_claims = []
    refusal_correct = None
    if judge_result:
        judge_forbidden = [
            str(value)
            for value in _as_list(judge_result.get("forbidden_claims_present"))
        ]
        unsupported_claims = [
            str(value)
            for value in _as_list(judge_result.get("unsupported_claims"))
        ]
        if isinstance(judge_result.get("refusal_correct"), bool):
            refusal_correct = bool(judge_result["refusal_correct"])

    if not item.get("should_answer", True):
        if forbidden_present or judge_forbidden:
            value = 0.0
        elif refusal_correct is True or _looks_like_refusal(answer):
            value = 1.0
        else:
            value = 0.3
    else:
        if forbidden_present or judge_forbidden:
            value = 0.0
        elif _looks_like_refusal(answer):
            value = 0.2
        elif unsupported_claims:
            value = 0.5
        else:
            value = 1.0
    return {
        "score": round(weight * value, 4),
        "weight": weight,
        "value": round(value, 6),
        "forbidden_claims_present": sorted(set(_forbidden_claims_present(item, answer))),
        "judge_forbidden_claims_present": judge_forbidden,
        "unsupported_claims": unsupported_claims,
        "refusal_detected": _looks_like_refusal(answer),
        "refusal_correct": refusal_correct,
    }


def _judge_item(
    item: dict[str, Any],
    record: dict[str, Any],
    answer: str,
    judge: LLMProvider | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if judge is None:
        return None
    prompt = _judge_prompt(item, record, answer)
    try:
        response = judge.generate(prompt)
        parsed = _extract_json_object(response.text)
        parsed["_judge_provider"] = response.provider
        parsed["_judge_model"] = response.model
        return parsed
    except Exception as exc:  # noqa: BLE001 - optional judge can fall back.
        if args.require_judge:
            raise
        return {
            "_judge_error": str(exc),
            "point_scores": [],
            "unsupported_claims": [],
            "forbidden_claims_present": [],
            "refusal_correct": None,
            "citation_support": [],
            "failure_reasons": ["judge_error"],
        }


def _judge_prompt(item: dict[str, Any], record: dict[str, Any], answer: str) -> str:
    evidence = [
        {
            "id": ev.get("id"),
            "section": ev.get("section"),
            "quote": ev.get("quote"),
        }
        for ev in item.get("evidence", [])
        if isinstance(ev, dict)
    ]
    points = [
        {
            "point_id": point.get("id"),
            "text": point.get("text"),
            "evidence_ids": point.get("evidence_ids", []),
        }
        for point in item.get("expected_points", [])
        if isinstance(point, dict)
    ]
    payload = {
        "question_id": item["id"],
        "question": item["query"],
        "should_answer": item["should_answer"],
        "expected_points": points,
        "evidence": evidence,
        "forbidden_claims": item.get("forbidden_claims", []),
        "answer": answer,
        "citations": _citations(record),
    }
    return (
        "You are a strict RAG evaluation judge. Grade only against the supplied "
        "expected points and evidence. Do not reward outside knowledge. Return "
        "JSON only, with no markdown.\n\n"
        "For each expected point, score 1.0 if fully covered and faithful, 0.5 "
        "if partially covered, and 0.0 if missing or contradicted. Identify "
        "unsupported claims and forbidden claims. For unanswerable items, "
        "refusal_correct should be true only when the answer clearly says the "
        "document does not provide the requested information and does not "
        "fabricate details.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "point_scores": [{"point_id": "p1", "score": 1.0, "rationale": "..."}],\n'
        '  "unsupported_claims": [],\n'
        '  "forbidden_claims_present": [],\n'
        '  "refusal_correct": true,\n'
        '  "citation_support": [{"point_id": "p1", "supported": true, "citation_ids": []}],\n'
        '  "failure_reasons": []\n'
        "}\n\n"
        f"Evaluation payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON was not an object.")
    return parsed


def _failure_reasons(
    item: dict[str, Any],
    retrieval: dict[str, Any],
    answer: dict[str, Any],
    citation: dict[str, Any],
    reliability: dict[str, Any],
    judge_result: dict[str, Any] | None,
    record: dict[str, Any],
) -> list[str]:
    reasons: set[str] = set()
    if record.get("error"):
        reasons.add("format_error")
    if item.get("should_answer", True) and retrieval.get("average_hit_value", 0.0) == 0.0:
        reasons.add("retrieval_miss")
        if item.get("query_type") == "table":
            reasons.add("table_parse_or_table_retrieval_failure")
    elif item.get("should_answer", True):
        for hit in retrieval.get("evidence_hits", []):
            if hit.get("rank") is None and hit.get("section_rank") is not None:
                reasons.add("wrong_section")
    if answer.get("coverage", 0.0) < 1.0:
        reasons.add("answer_missing_point")
    if citation.get("weight", 0) > 0:
        if citation.get("citation_count", 0) == 0:
            reasons.add("citation_missing")
        elif citation.get("support_rate", 0.0) < 1.0:
            reasons.add("citation_unsupported")
    if reliability.get("unsupported_claims"):
        reasons.add("unsupported_claim")
    if reliability.get("value", 1.0) <= 0.3 and not item.get("should_answer", True):
        reasons.add("refusal_failure")
    if judge_result:
        for reason in _as_list(judge_result.get("failure_reasons")):
            if isinstance(reason, str) and reason:
                reasons.add(reason)
    return sorted(reasons)


def _summarize(
    dataset: dict[str, Any],
    scores: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    answerable_scores = [score for score in scores if score["should_answer"]]
    unanswerable_scores = [score for score in scores if not score["should_answer"]]
    summary: dict[str, Any] = {
        "dataset_id": dataset.get("dataset_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "item_count": len(scores),
        "answerable_count": len(answerable_scores),
        "unanswerable_count": len(unanswerable_scores),
        "configuration": {
            "api_base_url": args.api_base_url,
            "results": str(args.results) if args.results else None,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "top_k": args.top_k,
            "rag_llm_provider": args.rag_llm_provider,
            "rag_model": args.rag_model,
        },
        "scores": _component_averages(scores),
        "retrieval_metrics": _retrieval_summary(answerable_scores),
        "answer_metrics": _answer_summary(scores, unanswerable_scores),
        "citation_metrics": _citation_summary(answerable_scores),
        "score_by_query_type": _group_score(scores, "query_type"),
        "score_by_difficulty": _group_score(scores, "difficulty"),
        "failure_reasons": dict(Counter(reason for s in scores for reason in s["failure_reasons"])),
        "worst_items": sorted(
            [
                {
                    "item_id": score["item_id"],
                    "query_type": score["query_type"],
                    "difficulty": score["difficulty"],
                    "final_score": score["final_score"],
                    "failure_reasons": score["failure_reasons"],
                }
                for score in scores
            ],
            key=lambda value: value["final_score"],
        )[:10],
    }
    return summary


def _component_averages(scores: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["retrieval", "answer", "citation", "reliability"]
    values = {"final_score_avg": _avg(score["final_score"] for score in scores)}
    for key in keys:
        values[f"{key}_score_avg"] = _avg(score["components"][key] for score in scores)
    return {key: round(value, 4) for key, value in values.items()}


def _retrieval_summary(scores: list[dict[str, Any]]) -> dict[str, float]:
    evidence_hits = [
        hit
        for score in scores
        for hit in score["retrieval"].get("evidence_hits", [])
    ]
    ranks = [hit.get("rank") for hit in evidence_hits]
    section_ranks = [hit.get("section_rank") for hit in evidence_hits]
    item_first_ranks = [score["retrieval"].get("first_gold_rank") for score in scores]
    return {
        "Recall@3": _rate(rank is not None and rank <= 3 for rank in ranks),
        "Recall@5": _rate(rank is not None and rank <= 5 for rank in ranks),
        "Recall@10": _rate(rank is not None and rank <= 10 for rank in ranks),
        "MRR@10": round(
            _avg((1.0 / rank) if isinstance(rank, int) and rank <= 10 else 0.0 for rank in item_first_ranks),
            6,
        ),
        "nDCG@10": round(_avg(score["retrieval"].get("ndcg_at_10", 0.0) for score in scores), 6),
        "ContextPrecision@5": round(
            _avg(score["retrieval"].get("context_precision_at_5", 0.0) for score in scores),
            6,
        ),
        "GoldSectionHitRate": _rate(
            rank is not None and rank <= 10 for rank in section_ranks
        ),
    }


def _answer_summary(
    scores: list[dict[str, Any]],
    unanswerable_scores: list[dict[str, Any]],
) -> dict[str, float]:
    unsupported_items = sum(
        1
        for score in scores
        if score["reliability"].get("unsupported_claims")
        or score["reliability"].get("judge_forbidden_claims_present")
    )
    refusal_correct = sum(
        1
        for score in unanswerable_scores
        if score["reliability"].get("value", 0.0) >= 1.0
    )
    return {
        "AnswerPointCoverage": round(
            _avg(score["answer"].get("coverage", 0.0) for score in scores),
            6,
        ),
        "UnsupportedClaimRate": round(
            unsupported_items / len(scores) if scores else 0.0,
            6,
        ),
        "RefusalAccuracy": round(
            refusal_correct / len(unanswerable_scores) if unanswerable_scores else 0.0,
            6,
        ),
    }


def _citation_summary(scores: list[dict[str, Any]]) -> dict[str, float]:
    scored = [score for score in scores if score["citation"].get("weight", 0) > 0]
    return {
        "CitationPrecision": round(
            _avg(score["citation"].get("precision", 0.0) for score in scored),
            6,
        ),
        "CitationRecall": round(
            _avg(score["citation"].get("recall", 0.0) for score in scored),
            6,
        ),
        "CitationSupportRate": round(
            _avg(score["citation"].get("support_rate", 0.0) for score in scored),
            6,
        ),
    }


def _group_score(scores: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        groups[str(score[key])].append(float(score["final_score"]))
    return {
        name: {"count": len(values), "avg_score": round(_avg(values), 4)}
        for name, values in sorted(groups.items())
    }


def _write_outputs(
    output_dir: Path,
    dataset: dict[str, Any],
    run_records: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    _write_jsonl(output_dir / "results.jsonl", run_records)
    _write_jsonl(output_dir / "scores.jsonl", scores)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(dataset, summary),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summary_markdown(dataset: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        "## Run Metadata",
        "",
        f"- dataset_id: `{summary.get('dataset_id')}`",
        f"- source_document: `{dataset.get('source_document', {}).get('source_file')}`",
        f"- source_sha256: `{dataset.get('source_document', {}).get('sha256')}`",
        f"- created_at: `{summary.get('created_at')}`",
        f"- elapsed_seconds: `{summary.get('elapsed_seconds')}`",
        "",
        "## Overall Score",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary["scores"].items():
        lines.append(f"| {key} | {value:.4f} |")

    lines.extend(["", "## Retrieval Metrics", "", "| Metric | Value |", "|---|---:|"])
    for key, value in summary["retrieval_metrics"].items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## Answer Metrics", "", "| Metric | Value |", "|---|---:|"])
    for key, value in summary["answer_metrics"].items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## Citation Metrics", "", "| Metric | Value |", "|---|---:|"])
    for key, value in summary["citation_metrics"].items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(["", "## Score By Query Type", "", "| query_type | count | avg_score |", "|---|---:|---:|"])
    for name, data in summary["score_by_query_type"].items():
        lines.append(f"| {name} | {data['count']} | {data['avg_score']:.4f} |")

    lines.extend(["", "## Score By Difficulty", "", "| difficulty | count | avg_score |", "|---|---:|---:|"])
    for name, data in summary["score_by_difficulty"].items():
        lines.append(f"| {name} | {data['count']} | {data['avg_score']:.4f} |")

    lines.extend(["", "## Failure Reasons", "", "| failure_reason | count |", "|---|---:|"])
    for reason, count in sorted(summary["failure_reasons"].items()):
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## Worst Items", "", "| item_id | type | difficulty | score | failures |", "|---|---|---|---:|---|"])
    for item in summary["worst_items"]:
        failures = ", ".join(item["failure_reasons"])
        lines.append(
            f"| {item['item_id']} | {item['query_type']} | {item['difficulty']} | "
            f"{item['final_score']:.4f} | {failures} |"
        )
    lines.append("")
    return "\n".join(lines)


def _answer_text(record: dict[str, Any]) -> str:
    chat = record.get("chat")
    if isinstance(chat, dict):
        value = chat.get("answer") or chat.get("content")
        if isinstance(value, str):
            return value
    value = record.get("answer")
    return value if isinstance(value, str) else ""


def _citations(record: dict[str, Any]) -> list[dict[str, Any]]:
    chat = record.get("chat")
    if isinstance(chat, dict):
        citations = chat.get("citations")
        if isinstance(citations, list):
            return [c for c in citations if isinstance(c, dict)]
    citations = record.get("citations")
    if isinstance(citations, list):
        return [c for c in citations if isinstance(c, dict)]
    return []


def _chat_contexts(record: dict[str, Any]) -> list[dict[str, Any]]:
    chat = record.get("chat")
    if isinstance(chat, dict) and isinstance(chat.get("contexts"), list):
        return [c for c in chat["contexts"] if isinstance(c, dict)]
    return []


def _forbidden_claims_present(item: dict[str, Any], answer: str) -> list[str]:
    answer_norm = _normalize_text(answer)
    present: list[str] = []
    for claim in item.get("forbidden_claims", []):
        claim_norm = _normalize_text(str(claim))
        if claim_norm and claim_norm in answer_norm:
            present.append(str(claim))
    return present


def _looks_like_refusal(answer: str) -> bool:
    answer_norm = _normalize_text(answer)
    return any(_normalize_text(pattern) in answer_norm for pattern in REFUSAL_PATTERNS)


def _similarity(expected: str, answer_norm: str) -> float:
    expected_norm = _normalize_text(expected)
    if not expected_norm or not answer_norm:
        return 0.0
    if expected_norm in answer_norm:
        return 1.0
    expected_tokens = _char_ngrams(expected_norm)
    answer_tokens = _char_ngrams(answer_norm)
    if not expected_tokens or not answer_tokens:
        return 0.0
    overlap = len(expected_tokens & answer_tokens)
    return overlap / len(expected_tokens)


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[`~!@#$%^&*()_=+\[\]{}\\|;:'\",.<>/?，。！？；：（）【】《》“”‘’、\-－—]", "", text)
    return text


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _avg(values: Any) -> float:
    collected = [float(value) for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _rate(values: Any) -> float:
    collected = list(values)
    return round(sum(1 for value in collected if value) / len(collected), 6) if collected else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
