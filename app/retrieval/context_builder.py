from __future__ import annotations

from app.retrieval.parent_expander import ParentExpander
from app.retrieval.schemas import ContextItem, RetrievalCandidate, RetrievalQuery


class ContextBuilder:
    def __init__(self, parent_expander: ParentExpander | None = None) -> None:
        self.parent_expander = parent_expander or ParentExpander()

    def build(
        self,
        candidates: list[RetrievalCandidate],
        *,
        final_top_k: int,
        query: RetrievalQuery | None = None,
    ) -> list[ContextItem]:
        contexts: list[ContextItem] = []
        seen: set[str] = set()

        for candidate in candidates:
            context = self.parent_expander.expand(candidate, query=query)
            if context.context_id in seen:
                continue
            contexts.append(context)
            seen.add(context.context_id)
            if len(contexts) >= final_top_k:
                break

        return contexts
