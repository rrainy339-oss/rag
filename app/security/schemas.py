from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AuthMode = Literal["disabled", "dev", "oidc"]


class Principal(BaseModel):
    subject: str
    tenant_id: str | None = None
    user_id: str | None = None
    email: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    max_classification: str | None = None
    auth_mode: AuthMode = "disabled"
    enforce_permissions: bool = True


class AuthError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code

