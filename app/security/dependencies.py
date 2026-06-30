from __future__ import annotations

from typing import Any

from fastapi import Request

from app.security.config import SecuritySettings
from app.security.schemas import AuthError, Principal


def authenticate_request(request: Request, settings: SecuritySettings) -> Principal:
    if settings.auth_mode == "disabled":
        return Principal(
            subject="anonymous",
            auth_mode="disabled",
            enforce_permissions=False,
        )
    if settings.auth_mode == "dev":
        return _dev_principal(request, settings)
    if settings.auth_mode == "oidc":
        return _oidc_principal(request, settings)
    raise AuthError(f"Unsupported auth mode: {settings.auth_mode}", status_code=500)


def require_scope(principal: Principal, scope: str) -> None:
    if not principal.enforce_permissions:
        return
    if scope in principal.scopes or "rag:admin" in principal.scopes:
        return
    raise AuthError(f"Missing required scope: {scope}", status_code=403)


def _dev_principal(request: Request, settings: SecuritySettings) -> Principal:
    if not settings.dev_allow_header_override:
        return Principal(
            subject=settings.dev_subject,
            tenant_id=settings.dev_tenant_id,
            user_id=settings.dev_user_id,
            email=settings.dev_email,
            group_ids=settings.dev_group_ids,
            roles=settings.dev_roles,
            scopes=settings.dev_scopes,
            max_classification=settings.dev_max_classification,
            auth_mode="dev",
            enforce_permissions=True,
        )

    return Principal(
        subject=_header(request, "x-rag-subject") or settings.dev_subject,
        tenant_id=_header(request, "x-rag-tenant-id") or settings.dev_tenant_id,
        user_id=_header(request, "x-rag-user-id") or settings.dev_user_id,
        email=_header(request, "x-rag-email") or settings.dev_email,
        group_ids=_header_list(request, "x-rag-group-ids", settings.dev_group_ids),
        roles=_header_list(request, "x-rag-roles", settings.dev_roles),
        scopes=_header_list(request, "x-rag-scopes", settings.dev_scopes),
        max_classification=(
            _header(request, "x-rag-max-classification")
            or settings.dev_max_classification
        ),
        auth_mode="dev",
        enforce_permissions=True,
    )


def _oidc_principal(request: Request, settings: SecuritySettings) -> Principal:
    token = _bearer_token(request)
    claims = _decode_oidc_token(token, settings)
    return _principal_from_claims(claims, settings)


def _decode_oidc_token(token: str, settings: SecuritySettings) -> dict[str, Any]:
    if not settings.oidc_jwks_url:
        raise AuthError("OIDC JWKS URL is not configured.", status_code=500)
    try:
        import jwt
        from jwt import PyJWKClient
    except ImportError as exc:
        raise AuthError(
            "OIDC auth requires PyJWT[crypto]. Install the auth extra.",
            status_code=500,
        ) from exc

    try:
        signing_key = PyJWKClient(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=settings.oidc_algorithms,
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
            options={
                "verify_aud": bool(settings.oidc_audience),
                "verify_iss": bool(settings.oidc_issuer),
            },
        )
    except Exception as exc:
        raise AuthError("Invalid bearer token.", status_code=401) from exc


def _principal_from_claims(
    claims: dict[str, Any],
    settings: SecuritySettings,
) -> Principal:
    subject = _claim_str(claims, settings.oidc_subject_claim)
    if not subject:
        raise AuthError("Token is missing subject claim.", status_code=401)

    return Principal(
        subject=subject,
        tenant_id=_claim_str(claims, settings.oidc_tenant_claim),
        user_id=_claim_str(claims, settings.oidc_user_id_claim) or subject,
        email=_claim_str(claims, settings.oidc_email_claim),
        group_ids=_claim_list(claims, settings.oidc_groups_claim),
        roles=_claim_list(claims, settings.oidc_roles_claim),
        scopes=_claim_list(claims, settings.oidc_scopes_claim),
        max_classification=_claim_str(claims, settings.oidc_max_classification_claim),
        auth_mode="oidc",
        enforce_permissions=True,
    )


def _bearer_token(request: Request) -> str:
    value = request.headers.get("authorization") or request.headers.get("Authorization")
    if not value:
        raise AuthError("Missing Authorization bearer token.", status_code=401)
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Invalid Authorization header.", status_code=401)
    return token.strip()


def _header(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _header_list(request: Request, name: str, default: list[str]) -> list[str]:
    value = _header(request, name)
    if value is None:
        return list(default)
    return _split_values(value)


def _claim_str(claims: dict[str, Any], name: str) -> str | None:
    value = claims.get(name)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _claim_list(claims: dict[str, Any], name: str) -> list[str]:
    value = claims.get(name)
    if value is None:
        return []
    if isinstance(value, str):
        return _split_values(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _split_values(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]

