from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from app.security.passwords import hash_password


class AuthUser(BaseModel):
    user_id: str
    username: str
    password_hash: str
    tenant_id: str | None = None
    email: str | None = None
    display_name: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    max_classification: str | None = None
    is_active: bool = True
    token_version: int = 1
    failed_login_count: int = 0
    locked_until: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login_at: datetime | None = None


class SeedAuthUser(BaseModel):
    user_id: str
    username: str
    password: str
    tenant_id: str | None = None
    email: str | None = None
    display_name: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    max_classification: str | None = None
    is_active: bool = True


class AuthUserStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    tenant_id TEXT,
                    email TEXT,
                    display_name TEXT,
                    group_ids TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    max_classification TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    token_version INTEGER NOT NULL DEFAULT 1,
                    failed_login_count INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_auth_users_username
                ON auth_users(username)
                """
            )

    def seed_defaults(self, users: list[SeedAuthUser]) -> None:
        for user in users:
            self.create_if_missing(user)

    def create_if_missing(self, seed_user: SeedAuthUser) -> AuthUser:
        existing = self.get_by_username(seed_user.username)
        if existing is not None:
            return existing
        now = datetime.now(timezone.utc)
        user = AuthUser(
            user_id=seed_user.user_id,
            username=seed_user.username,
            password_hash=hash_password(seed_user.password),
            tenant_id=seed_user.tenant_id,
            email=seed_user.email,
            display_name=seed_user.display_name,
            group_ids=seed_user.group_ids,
            roles=seed_user.roles,
            scopes=seed_user.scopes,
            max_classification=seed_user.max_classification,
            is_active=seed_user.is_active,
            created_at=now,
            updated_at=now,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO auth_users (
                    user_id, username, password_hash, tenant_id, email,
                    display_name, group_ids, roles, scopes, max_classification,
                    is_active, token_version, failed_login_count, locked_until,
                    created_at, updated_at, last_login_at
                )
                VALUES (
                    :user_id, :username, :password_hash, :tenant_id, :email,
                    :display_name, :group_ids, :roles, :scopes, :max_classification,
                    :is_active, :token_version, :failed_login_count, :locked_until,
                    :created_at, :updated_at, :last_login_at
                )
                """,
                _user_to_row(user),
            )
        return user

    def get_by_username(self, username: str) -> AuthUser | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM auth_users WHERE lower(username) = lower(?)",
                (username,),
            ).fetchone()
        return _row_to_user(row) if row is not None else None

    def mark_login_success(self, user_id: str) -> AuthUser:
        now = datetime.now(timezone.utc).isoformat()
        return self.update(
            user_id,
            failed_login_count=0,
            locked_until=None,
            last_login_at=now,
        )

    def mark_login_failure(self, user_id: str) -> AuthUser:
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        return self.update(
            user_id,
            failed_login_count=user.failed_login_count + 1,
        )

    def get(self, user_id: str) -> AuthUser | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM auth_users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _row_to_user(row) if row is not None else None

    def update(self, user_id: str, **values: object) -> AuthUser:
        if not values:
            user = self.get(user_id)
            if user is None:
                raise KeyError(user_id)
            return user
        values["updated_at"] = datetime.now(timezone.utc).isoformat()
        assignments = ", ".join(f"{key} = :{key}" for key in values)
        params = {**values, "user_id": user_id}
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE auth_users SET {assignments} WHERE user_id = :user_id",
                params,
            )
            if cursor.rowcount == 0:
                raise KeyError(user_id)
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        return user

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _user_to_row(user: AuthUser) -> dict[str, object | None]:
    data = user.model_dump()
    data["group_ids"] = json.dumps(user.group_ids, ensure_ascii=False)
    data["roles"] = json.dumps(user.roles, ensure_ascii=False)
    data["scopes"] = json.dumps(user.scopes, ensure_ascii=False)
    data["is_active"] = 1 if user.is_active else 0
    for key in ("locked_until", "created_at", "updated_at", "last_login_at"):
        value = data.get(key)
        data[key] = value.isoformat() if isinstance(value, datetime) else value
    return data


def _row_to_user(row: sqlite3.Row) -> AuthUser:
    data = dict(row)
    data["group_ids"] = _json_list(data.get("group_ids"))
    data["roles"] = _json_list(data.get("roles"))
    data["scopes"] = _json_list(data.get("scopes"))
    data["is_active"] = bool(data.get("is_active"))
    return AuthUser.model_validate(data)


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item).strip() for item in decoded if str(item).strip()]
