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

    jwt_secret: str = "change-me-local-rag-jwt-secret"
    jwt_issuer: str = "enterprise-rag"
    jwt_audience: str = "enterprise-rag-api"
    jwt_expire_minutes: int = Field(default=480, ge=1)
    auth_db_path: Path = Path("artifacts/security/auth.sqlite3")
    auth_seed_default_users: bool = True

    jwt_admin_username: str = "admin"
    jwt_admin_password: str = "admin123"
    jwt_admin_subject: str = "admin"
    jwt_admin_tenant_id: str | None = "tenant-a"
    jwt_admin_user_id: str | None = "admin"
    jwt_admin_email: str | None = None
    jwt_admin_group_ids: list[str] = Field(default_factory=lambda: ["admin"])
    jwt_admin_roles: list[str] = Field(default_factory=lambda: ["document_manager"])
    jwt_admin_scopes: list[str] = Field(
        default_factory=lambda: [
            "rag:chat",
            "rag:retrieve",
            "rag:models",
            "rag:documents",
        ]
    )
    jwt_admin_max_classification: str | None = "secret"

    jwt_user_username: str = "user"
    jwt_user_password: str = "user123"
    jwt_user_subject: str = "user"
    jwt_user_tenant_id: str | None = "tenant-a"
    jwt_user_user_id: str | None = "user"
    jwt_user_email: str | None = None
    jwt_user_group_ids: list[str] = Field(default_factory=list)
    jwt_user_roles: list[str] = Field(default_factory=lambda: ["user"])
    jwt_user_scopes: list[str] = Field(
        default_factory=lambda: ["rag:chat", "rag:retrieve", "rag:models"]
    )
    jwt_user_max_classification: str | None = "internal"

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
            jwt_secret=_str("RAG_JWT_SECRET", "change-me-local-rag-jwt-secret"),
            jwt_issuer=_str("RAG_JWT_ISSUER", "enterprise-rag"),
            jwt_audience=_str("RAG_JWT_AUDIENCE", "enterprise-rag-api"),
            jwt_expire_minutes=_int("RAG_JWT_EXPIRE_MINUTES", 480),
            auth_db_path=_path(
                "RAG_AUTH_DB_PATH",
                cls.model_fields["auth_db_path"].default,
            ),
            auth_seed_default_users=_bool("RAG_AUTH_SEED_DEFAULT_USERS", True),
            jwt_admin_username=_str("RAG_JWT_ADMIN_USERNAME", "admin"),
            jwt_admin_password=_str("RAG_JWT_ADMIN_PASSWORD", "admin123"),
            jwt_admin_subject=_str("RAG_JWT_ADMIN_SUBJECT", "admin"),
            jwt_admin_tenant_id=_optional_str("RAG_JWT_ADMIN_TENANT_ID") or "tenant-a",
            jwt_admin_user_id=_optional_str("RAG_JWT_ADMIN_USER_ID") or "admin",
            jwt_admin_email=_optional_str("RAG_JWT_ADMIN_EMAIL"),
            jwt_admin_group_ids=_csv("RAG_JWT_ADMIN_GROUP_IDS", ["admin"]),
            jwt_admin_roles=_csv("RAG_JWT_ADMIN_ROLES", ["document_manager"]),
            jwt_admin_scopes=_csv(
                "RAG_JWT_ADMIN_SCOPES",
                ["rag:chat", "rag:retrieve", "rag:models", "rag:documents"],
            ),
            jwt_admin_max_classification=(
                _optional_str("RAG_JWT_ADMIN_MAX_CLASSIFICATION") or "secret"
            ),
            jwt_user_username=_str("RAG_JWT_USER_USERNAME", "user"),
            jwt_user_password=_str("RAG_JWT_USER_PASSWORD", "user123"),
            jwt_user_subject=_str("RAG_JWT_USER_SUBJECT", "user"),
            jwt_user_tenant_id=_optional_str("RAG_JWT_USER_TENANT_ID") or "tenant-a",
            jwt_user_user_id=_optional_str("RAG_JWT_USER_USER_ID") or "user",
            jwt_user_email=_optional_str("RAG_JWT_USER_EMAIL"),
            jwt_user_group_ids=_csv("RAG_JWT_USER_GROUP_IDS", []),
            jwt_user_roles=_csv("RAG_JWT_USER_ROLES", ["user"]),
            jwt_user_scopes=_csv(
                "RAG_JWT_USER_SCOPES",
                ["rag:chat", "rag:retrieve", "rag:models"],
            ),
            jwt_user_max_classification=(
                _optional_str("RAG_JWT_USER_MAX_CLASSIFICATION") or "internal"
            ),
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


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _csv(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _path(name: str, default: object) -> Path:
    value = os.getenv(name)
    return Path(value) if value else Path(default)


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return Path(value)
