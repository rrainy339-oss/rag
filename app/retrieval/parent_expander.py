from __future__ import annotations

from app.chunking.schemas import ChunkType
from app.domain.permissions import access_visible_to_context, security_context_for_subject
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.schemas import ContextItem, RetrievalCandidate, RetrievalQuery


class ParentExpander:
    def __init__(self, chunk_store: ChunkStore | None = None) -> None:
        self.chunk_store = chunk_store or ChunkStore()

    def expand(
        self,
        candidate: RetrievalCandidate,
        *,
        query: RetrievalQuery | None = None,
    ) -> ContextItem:
        record = candidate.record
        security_context = (
            security_context_for_subject(
                tenant_id=query.tenant_id,
                user_id=query.user_id,
                group_ids=query.group_ids,
                max_classification=query.max_classification,
            )
            if query is not None
            else None
        )
        parent = self.chunk_store.get(record.parent_chunk_id)
        source_chunk = self.chunk_store.get(record.chunk_id)
        if parent is not None and security_context is not None:
            if not access_visible_to_context(parent.access, security_context):
                parent = None
        if source_chunk is not None and security_context is not None:
            if not access_visible_to_context(source_chunk.access, security_context):
                source_chunk = None

        if record.chunk_type == ChunkType.TABLE:
            text = record.text
            contextual_text = record.contextual_text
            if parent is not None:
                text = f"{record.text}\n\nRelated section context:\n{parent.text}"
                contextual_text = f"{record.contextual_text}\n\nRelated section context:\n{parent.contextual_text}"
            context_id = record.chunk_id
        elif parent is not None:
            text = parent.text
            contextual_text = parent.contextual_text
            context_id = parent.chunk_id
        elif source_chunk is not None:
            text = source_chunk.text
            contextual_text = source_chunk.contextual_text
            context_id = source_chunk.chunk_id
        else:
            text = record.text
            contextual_text = record.contextual_text
            context_id = record.chunk_id

        return ContextItem(
            context_id=context_id,
            source_chunk_id=record.chunk_id,
            parent_chunk_id=record.parent_chunk_id,
            chunk_type=record.chunk_type,
            text=text,
            contextual_text=contextual_text,
            score=candidate.final_score,
            source_scores=candidate.source_scores,
            section_path=list(record.metadata.get("section_path") or []),
            citations=record.citations,
            metadata={
                "record_id": record.record_id,
                "rank": candidate.rank,
                "fusion_boost": candidate.metadata.get("fusion_boost"),
                "expanded_from_parent": record.chunk_type != ChunkType.TABLE and parent is not None,
            },
        )
