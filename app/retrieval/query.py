from __future__ import annotations

import re

from app.retrieval.schemas import AnalyzedQuery, QueryType, RetrievalQuery


class QueryAnalyzer:
    def analyze(self, query: RetrievalQuery) -> AnalyzedQuery:
        text = query.query.strip()
        exact_terms = _extract_exact_terms(text)
        lower = text.lower()

        if any(
            word in lower
            for word in (
                "table",
                "表格",
                "多少",
                "金额",
                "数量",
                "比例",
                "比率",
                "ratio",
                "rate",
            )
        ):
            query_type = QueryType.TABLE
        elif any(word in lower for word in ("summarize", "summary", "总结", "概括")):
            query_type = QueryType.SUMMARY
        elif exact_terms:
            query_type = QueryType.EXACT
        else:
            query_type = QueryType.GENERAL

        return AnalyzedQuery(query=query, query_type=query_type, exact_terms=exact_terms)


def _extract_exact_terms(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z]{2,}[-_]?\d+[A-Z0-9_-]*\b",
        r"\b\d+(?:\.\d+){1,}\b",
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
    ]
    terms: list[str] = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text))

    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|「([^」]+)」|『([^』]+)』', text)
    for group in quoted:
        terms.extend(item for item in group if item)
    return list(dict.fromkeys(terms))
