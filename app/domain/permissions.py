from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from app.domain.schemas import AccessControl


PUBLIC_PRINCIPAL = "public"
PRINCIPAL_IDS_ANY_FILTER = "principal_ids_any"
CLASSIFICATION_LEVEL_LTE_FILTER = "classification_level_lte"
UNKNOWN_CLASSIFICATION_LEVEL = 10_000

CLASSIFICATION_RANKS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
    "secret": 4,
}


@dataclass(frozen=True)
class SecurityContext:
    principal_ids: list[str]
    max_classification_level: int | None = None


def tenant_principal(tenant_id: str) -> str:
    return f"tenant:{tenant_id}"


def user_principal(user_id: str, *, tenant_id: str | None = None) -> str:
    if tenant_id:
        return f"user:{tenant_id}:{user_id}"
    return f"user:{user_id}"


def group_principal(group_id: str, *, tenant_id: str | None = None) -> str:
    if tenant_id:
        return f"group:{tenant_id}:{group_id}"
    return f"group:{group_id}"


def access_principal_ids(access: AccessControl) -> list[str]:
    tenant_id = _clean(access.tenant_id)
    user_ids = _clean_many(access.allowed_user_ids)
    group_ids = _clean_many(access.allowed_group_ids)

    principals: list[str] = []
    if user_ids or group_ids:
        principals.extend(
            user_principal(user_id, tenant_id=tenant_id) for user_id in user_ids
        )
        principals.extend(
            group_principal(group_id, tenant_id=tenant_id) for group_id in group_ids
        )
    elif tenant_id:
        principals.append(tenant_principal(tenant_id))
    else:
        principals.append(PUBLIC_PRINCIPAL)

    return _unique(principals)


def security_context_for_subject(
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    group_ids: Iterable[str] | None = None,
    max_classification: str | None = None,
) -> SecurityContext:
    clean_tenant_id = _clean(tenant_id)
    clean_user_id = _clean(user_id)
    clean_group_ids = _clean_many(group_ids or [])

    principals = [PUBLIC_PRINCIPAL]
    if clean_tenant_id:
        principals.append(tenant_principal(clean_tenant_id))
    if clean_user_id:
        principals.append(user_principal(clean_user_id, tenant_id=clean_tenant_id))
        principals.append(user_principal(clean_user_id))
    principals.extend(
        group_principal(group_id, tenant_id=clean_tenant_id)
        for group_id in clean_group_ids
    )
    principals.extend(group_principal(group_id) for group_id in clean_group_ids)

    return SecurityContext(
        principal_ids=_unique(principals),
        max_classification_level=clearance_level(max_classification),
    )


def access_classification_level(access: AccessControl) -> int:
    if not access.classification:
        return CLASSIFICATION_RANKS["public"]
    level = classification_level(access.classification)
    return level if level is not None else UNKNOWN_CLASSIFICATION_LEVEL


def classification_level(value: str | None) -> int | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return CLASSIFICATION_RANKS.get(cleaned.lower())


def clearance_level(max_classification: str | None) -> int | None:
    if not max_classification:
        return None
    level = classification_level(max_classification)
    return level if level is not None else UNKNOWN_CLASSIFICATION_LEVEL


def access_visible_to_context(
    access: AccessControl,
    context: SecurityContext,
) -> bool:
    if not set(access_principal_ids(access)).intersection(context.principal_ids):
        return False

    if context.max_classification_level is not None:
        if access_classification_level(access) > context.max_classification_level:
            return False

    return True


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_many(values: Iterable[str]) -> list[str]:
    return _unique(value for value in (_clean(value) for value in values) if value)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
