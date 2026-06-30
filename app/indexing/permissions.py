from __future__ import annotations

from app.domain.permissions import access_classification_level, access_principal_ids
from app.domain.schemas import AccessControl


def access_to_payload(access: AccessControl) -> dict[str, object]:
    return {
        "tenant_id": access.tenant_id,
        "allowed_user_ids": access.allowed_user_ids,
        "allowed_group_ids": access.allowed_group_ids,
        "classification": access.classification,
        "principal_ids": access_principal_ids(access),
        "classification_level": access_classification_level(access),
    }
