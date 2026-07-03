from __future__ import annotations

from datetime import datetime, timezone

from app.security.config import SecuritySettings
from app.security.passwords import verify_password
from app.security.schemas import AuthError, Principal
from app.security.users import AuthUser, AuthUserStore, SeedAuthUser


class AuthService:
    def __init__(
        self,
        settings: SecuritySettings,
        *,
        store: AuthUserStore | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or AuthUserStore(settings.auth_db_path)
        if settings.auth_seed_default_users:
            self.store.seed_defaults(default_seed_users(settings))

    def authenticate_login(self, username: str, password: str, *, login_type: str) -> Principal:
        user = self.store.get_by_username(username.strip())
        if user is None:
            raise AuthError("Invalid credentials.", status_code=401)
        if not user.is_active:
            raise AuthError("User account is disabled.", status_code=403)
        if _is_locked(user):
            raise AuthError("User account is locked.", status_code=403)

        if not verify_password(password, user.password_hash):
            self.store.mark_login_failure(user.user_id)
            raise AuthError("Invalid credentials.", status_code=401)

        if not _login_type_allowed(user, login_type):
            raise AuthError("User is not allowed to access this frontend.", status_code=403)

        user = self.store.mark_login_success(user.user_id)
        return _principal_from_user(user)


def default_seed_users(settings: SecuritySettings) -> list[SeedAuthUser]:
    return [
        SeedAuthUser(
            user_id=settings.jwt_admin_user_id or settings.jwt_admin_subject,
            username=settings.jwt_admin_username,
            password=settings.jwt_admin_password,
            tenant_id=settings.jwt_admin_tenant_id,
            email=settings.jwt_admin_email,
            display_name=settings.jwt_admin_subject,
            group_ids=settings.jwt_admin_group_ids,
            roles=settings.jwt_admin_roles,
            scopes=settings.jwt_admin_scopes,
            max_classification=settings.jwt_admin_max_classification,
        ),
        SeedAuthUser(
            user_id=settings.jwt_user_user_id or settings.jwt_user_subject,
            username=settings.jwt_user_username,
            password=settings.jwt_user_password,
            tenant_id=settings.jwt_user_tenant_id,
            email=settings.jwt_user_email,
            display_name=settings.jwt_user_subject,
            group_ids=settings.jwt_user_group_ids,
            roles=settings.jwt_user_roles,
            scopes=settings.jwt_user_scopes,
            max_classification=settings.jwt_user_max_classification,
        ),
    ]


def _principal_from_user(user: AuthUser) -> Principal:
    return Principal(
        subject=user.user_id,
        tenant_id=user.tenant_id,
        user_id=user.user_id,
        email=user.email,
        group_ids=user.group_ids,
        roles=user.roles,
        scopes=user.scopes,
        max_classification=user.max_classification,
        auth_mode="jwt",
        enforce_permissions=True,
    )


def _login_type_allowed(user: AuthUser, login_type: str) -> bool:
    scopes = set(user.scopes)
    if "rag:admin" in scopes:
        return True
    if login_type == "admin":
        return "rag:documents" in scopes
    if login_type == "chat":
        return "rag:chat" in scopes
    return False


def _is_locked(user: AuthUser) -> bool:
    if user.locked_until is None:
        return False
    return user.locked_until > datetime.now(timezone.utc)
