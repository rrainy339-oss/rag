"""Authentication, authorization, and audit helpers for the API layer."""

from app.security.audit import AuditLogger
from app.security.auth_service import AuthService
from app.security.config import SecuritySettings
from app.security.dependencies import authenticate_request, require_scope
from app.security.schemas import AuthError, Principal
from app.security.users import AuthUser, AuthUserStore, SeedAuthUser

__all__ = [
    "AuditLogger",
    "AuthError",
    "AuthService",
    "AuthUser",
    "AuthUserStore",
    "Principal",
    "SeedAuthUser",
    "SecuritySettings",
    "authenticate_request",
    "require_scope",
]
