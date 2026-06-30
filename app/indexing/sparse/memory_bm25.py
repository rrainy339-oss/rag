from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from app.indexing.metadata_filters import metadata_matches
from app.indexing.schemas import IndexRecord
from app.indexing.sparse.base import SparseSearchResult


class MemoryBM25Store:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.records: dict[str, IndexRecord] = {}
        self._doc_terms: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._term_doc_freq: dict[str, int] = defaultdict(int)

    def upsert(self, records: list[IndexRecord]) -> None:
        for record in records:
            if record.record_id in self.records:
                self._remove_record_terms(record.record_id)
            self.records[record.record_id] = record
            terms = Counter(tokenize(record.contextual_text))
            self._doc_terms[record.record_id] = terms
            self._doc_lengths[record.record_id] = sum(terms.values())
            for term in terms:
                self._term_doc_freq[term] += 1

    def delete_by_document(self, document_id: str) -> int:
        record_ids = [
            record_id
            for record_id, record in self.records.items()
            if record.document_id == document_id
        ]
        for record_id in record_ids:
            self._remove_record_terms(record_id)
            del self.records[record_id]
        return len(record_ids)

    def count(self) -> int:
        return len(self.records)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SparseSearchResult]:
        filters = filters or {}
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        average_length = self._average_doc_length()
        total_docs = len(self.records)
        results: list[SparseSearchResult] = []

        for record_id, record in self.records.items():
            if not metadata_matches(record.metadata, filters):
                continue
            terms = self._doc_terms.get(record_id, Counter())
            doc_length = self._doc_lengths.get(record_id, 0)
            score = 0.0
            for term, query_weight in query_terms.items():
                term_freq = terms.get(term, 0)
                if term_freq == 0:
                    continue
                doc_freq = self._term_doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_length / max(average_length, 1)
                )
                score += query_weight * idf * term_freq * (self.k1 + 1) / denom
            if score > 0:
                results.append(SparseSearchResult(record=record, score=score))

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def _remove_record_terms(self, record_id: str) -> None:
        terms = self._doc_terms.pop(record_id, Counter())
        self._doc_lengths.pop(record_id, None)
        for term in terms:
            self._term_doc_freq[term] -= 1
            if self._term_doc_freq[term] <= 0:
                del self._term_doc_freq[term]

    def _average_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)


def build_sparse_terms(text: str) -> dict[str, float]:
    counts = Counter(tokenize(text))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {term: count / total for term, count in sorted(counts.items())}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())
