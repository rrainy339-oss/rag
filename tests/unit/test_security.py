from __future__ import annotations

import unittest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.security import AuthError, SecuritySettings, authenticate_request, require_scope


class SecurityTest(unittest.TestCase):
    def test_dev_auth_reads_header_identity(self) -> None:
        request = _request(
            headers={
                "x-rag-subject": "alice",
                "x-rag-tenant-id": "tenant-a",
                "x-rag-user-id": "u-alice",
                "x-rag-group-ids": "hr,finance",
                "x-rag-scopes": "rag:chat,rag:retrieve",
                "x-rag-max-classification": "confidential",
            }
        )

        principal = authenticate_request(
            request,
            SecuritySettings(auth_mode="dev"),
        )

        self.assertEqual(principal.subject, "alice")
        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.user_id, "u-alice")
        self.assertEqual(principal.group_ids, ["hr", "finance"])
        self.assertEqual(principal.max_classification, "confidential")

    def test_disabled_auth_does_not_enforce_permissions(self) -> None:
        principal = authenticate_request(_request(), SecuritySettings(auth_mode="disabled"))

        require_scope(principal, "rag:admin")

        self.assertFalse(principal.enforce_permissions)

    def test_scope_check_rejects_missing_scope(self) -> None:
        principal = authenticate_request(
            _request(headers={"x-rag-scopes": "rag:retrieve"}),
            SecuritySettings(auth_mode="dev"),
        )

        with self.assertRaises(AuthError) as error:
            require_scope(principal, "rag:chat")

        self.assertEqual(error.exception.status_code, 403)

    def test_oidc_requires_bearer_token(self) -> None:
        with self.assertRaises(AuthError) as error:
            authenticate_request(
                _request(),
                SecuritySettings(auth_mode="oidc", oidc_jwks_url="https://issuer/jwks"),
            )

        self.assertEqual(error.exception.status_code, 401)


def _request(headers: dict[str, str] | None = None) -> Request:
    app = FastAPI()

    @app.get("/probe")
    def probe(request: Request) -> dict[str, object]:
        app.state.request_object = request
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/probe", headers=headers or {})
        response.raise_for_status()
        return _last_request(app)


def _last_request(app: FastAPI) -> Request:
    value = getattr(app.state, "request_object", None)
    if value is None:
        raise AssertionError("test request was not captured")
    return value


if __name__ == "__main__":
    unittest.main()
