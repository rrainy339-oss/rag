from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from app.security.schemas import AuthMode


class SecuritySettings(BaseModel):
    auth_mode: AuthMode = "disabled"

    dev_subject: str = "dev-user"
    dev_tenant_id: str | None = None
    dev_user_id: str | None = "dev-user"
    dev_email: str | None = None
    dev_group_ids: list[str] = Field(default_factory=list)
    dev_roles: list[str] = Field(default_factory=lambda: ["developer"])
    dev_scopes: list[str] = Field(
        default_factory=lambda: [
            "rag:chat",
            "rag:retrieve",
            "rag:models",
            "rag:documents",
        ]
    )
    dev_max_classification: str | None = None
    dev_allow_header_override: bool = True

    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    oidc_subject_claim: str = "sub"
    oidc_user_id_claim: str = "sub"
    oidc_email_claim: str = "email"
    oidc_tenant_claim: str = "tenant_id"
    oidc_groups_claim: str = "groups"
    oidc_roles_claim: str = "roles"
    oidc_scopes_claim: str = "scope"
    oidc_max_classification_claim: str = "max_classification"

    audit_log_path: Path | None = None
    audit_log_query: bool = True

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        return cls(
            auth_mode=_str("RAG_AUTH_MODE", "disabled"),
            dev_subject=_str("RAG_DEV_SUBJECT", "dev-user"),
            dev_tenant_id=_optional_str("RAG_DEV_TENANT_ID"),
            dev_user_id=_optional_str("RAG_DEV_USER_ID") or "dev-user",
            dev_email=_optional_str("RAG_DEV_EMAIL"),
            dev_group_ids=_csv("RAG_DEV_GROUP_IDS", []),
            dev_roles=_csv("RAG_DEV_ROLES", ["developer"]),
            dev_scopes=_csv(
                "RAG_DEV_SCOPES",
                ["rag:chat", "rag:retrieve", "rag:models", "rag:documents"],
            ),
            dev_max_classification=_optional_str("RAG_DEV_MAX_CLASSIFICATION"),
            dev_allow_header_override=_bool("RAG_DEV_ALLOW_HEADER_OVERRIDE", True),
            oidc_issuer=_optional_str("RAG_OIDC_ISSUER"),
            oidc_audience=_optional_str("RAG_OIDC_AUDIENCE"),
            oidc_jwks_url=_optional_str("RAG_OIDC_JWKS_URL"),
            oidc_algorithms=_csv("RAG_OIDC_ALGORITHMS", ["RS256"]),
            oidc_subject_claim=_str("RAG_OIDC_SUBJECT_CLAIM", "sub"),
            oidc_user_id_claim=_str("RAG_OIDC_USER_ID_CLAIM", "sub"),
            oidc_email_claim=_str("RAG_OIDC_EMAIL_CLAIM", "email"),
            oidc_tenant_claim=_str("RAG_OIDC_TENANT_CLAIM", "tenant_id"),
            oidc_groups_claim=_str("RAG_OIDC_GROUPS_CLAIM", "groups"),
            oidc_roles_claim=_str("RAG_OIDC_ROLES_CLAIM", "roles"),
            oidc_scopes_claim=_str("RAG_OIDC_SCOPES_CLAIM", "scope"),
            oidc_max_classification_claim=_str(
                "RAG_OIDC_MAX_CLASSIFICATION_CLAIM",
                "max_classification",
            ),
            audit_log_path=_optional_path("RAG_AUDIT_LOG_PATH"),
            audit_log_query=_bool("RAG_AUDIT_LOG_QUERY", True),
        )


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return Path(value)
