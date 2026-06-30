from __future__ import annotations

from app.domain.permissions import access_visible_to_context, security_context_for_subject
from app.indexing.metadata_filters import metadata_matches
from app.indexing.schemas import IndexRecord
from app.retrieval.schemas import RetrievalQuery


def is_record_visible(record: IndexRecord, query: RetrievalQuery) -> bool:
    if not metadata_matches(record.metadata, query.metadata_filters):
        return False

    context = security_context_for_subject(
        tenant_id=query.tenant_id,
        user_id=query.user_id,
        group_ids=query.group_ids,
        max_classification=query.max_classification,
    )
    return access_visible_to_context(record.access, context)
