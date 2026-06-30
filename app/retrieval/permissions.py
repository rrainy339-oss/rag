from __future__ import annotations

from app.domain.permissions import (
    CLASSIFICATION_LEVEL_LTE_FILTER,
    PRINCIPAL_IDS_ANY_FILTER,
    security_context_for_subject,
)
from app.retrieval.schemas import RetrievalQuery


def filters_for_query(query: RetrievalQuery) -> dict[str, object]:
    context = security_context_for_subject(
        tenant_id=query.tenant_id,
        user_id=query.user_id,
        group_ids=query.group_ids,
        max_classification=query.max_classification,
    )
    filters = dict(query.metadata_filters)
    filters[PRINCIPAL_IDS_ANY_FILTER] = context.principal_ids
    if context.max_classification_level is not None:
        filters[CLASSIFICATION_LEVEL_LTE_FILTER] = context.max_classification_level
    return filters
