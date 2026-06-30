from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.security.config import SecuritySettings
from app.security.schemas import Principal


class AuditLogger:
    def __init__(self, *, path: Path | None = None, log_query: bool = True) -> None:
        self.path = path
        self.log_query = log_query

    @classmethod
    def from_settings(cls, settings: SecuritySettings) -> "AuditLogger":
        return cls(path=settings.audit_log_path, log_query=settings.audit_log_query)

    def log(
        self,
        *,
        event_type: str,
        request_id: str,
        principal: Principal,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.path is None:
            return

        event = {
            "event_type": event_type,
            "request_id": request_id,
            "subject": principal.subject,
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "group_ids": principal.group_ids,
            "roles": principal.roles,
            "auth_mode": principal.auth_mode,
            "query": query if self.log_query else None,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str))
            file.write("\n")

