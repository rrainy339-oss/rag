from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.schemas import ContextItem, RetrievalResponse


@dataclass(frozen=True)
class PackedContext:
    label: str
    context_id: str
    source_chunk_id: str
    parent_chunk_id: str | None
    section_path: list[str]
    text: str
    score: float
    metadata: dict[str, object]


class ContextPacker:
    def __init__(
        self,
        *,
        top_contexts: int = 5,
        max_context_chars: int = 12000,
        max_chars_per_context: int = 3000,
    ) -> None:
        self.top_contexts = top_contexts
        self.max_context_chars = max_context_chars
        self.max_chars_per_context = max_chars_per_context

    def pack(self, response: RetrievalResponse) -> list[PackedContext]:
        packed: list[PackedContext] = []
        seen: set[str] = set()
        used_chars = 0

        for context in response.contexts:
            if len(packed) >= self.top_contexts:
                break
            if context.context_id in seen:
                continue
            text = _trim_text(context.text or context.contextual_text, self.max_chars_per_context)
            if not text:
                continue
            if used_chars + len(text) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= 0:
                    break
                text = _trim_text(text, remaining)
            packed.append(_to_packed_context(context, len(packed) + 1, text))
            seen.add(context.context_id)
            used_chars += len(text)

        return packed


def _to_packed_context(context: ContextItem, index: int, text: str) -> PackedContext:
    return PackedContext(
        label=f"Context {index}",
        context_id=context.context_id,
        source_chunk_id=context.source_chunk_id,
        parent_chunk_id=context.parent_chunk_id,
        section_path=list(context.section_path),
        text=text,
        score=float(context.score),
        metadata=dict(context.metadata),
    )


def _trim_text(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    return text[: max_chars - 12].rstrip() + " ...[truncated]"
