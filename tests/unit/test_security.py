from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.security import AuthError, AuthService, SecuritySettings, authenticate_request, require_scope
from app.security.jwt import encode_jwt
from app.security.passwords import hash_password, verify_password
from app.security.users import AuthUserStore, SeedAuthUser


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

    def test_jwt_auth_reads_bearer_claims(self) -> None:
        token = encode_jwt(
            {
                "sub": "alice",
                "tenant_id": "tenant-a",
                "user_id": "u-alice",
                "group_ids": ["hr", "finance"],
                "roles": ["user"],
                "scopes": ["rag:chat", "rag:retrieve"],
                "max_classification": "confidential",
            },
            secret="test-secret",
            issuer="enterprise-rag",
            audience="enterprise-rag-api",
            expires_in_seconds=60,
        )

        principal = authenticate_request(
            _request(headers={"authorization": f"Bearer {token}"}),
            SecuritySettings(auth_mode="jwt", jwt_secret="test-secret"),
        )

        self.assertEqual(principal.subject, "alice")
        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.group_ids, ["hr", "finance"])
        self.assertEqual(principal.scopes, ["rag:chat", "rag:retrieve"])
        self.assertEqual(principal.auth_mode, "jwt")

    def test_jwt_auth_rejects_missing_bearer_token(self) -> None:
        with self.assertRaises(AuthError) as error:
            authenticate_request(
                _request(),
                SecuritySettings(auth_mode="jwt", jwt_secret="test-secret"),
            )

        self.assertEqual(error.exception.status_code, 401)

    def test_password_hash_verification(self) -> None:
        password_hash = hash_password("secret-pass")

        self.assertTrue(verify_password("secret-pass", password_hash))
        self.assertFalse(verify_password("wrong-pass", password_hash))

    def test_auth_service_authenticates_database_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SecuritySettings(
                auth_mode="jwt",
                auth_db_path=Path(temp_dir) / "auth.sqlite3",
                auth_seed_default_users=False,
            )
            store = AuthUserStore(settings.auth_db_path)
            store.seed_defaults(
                [
                    SeedAuthUser(
                        user_id="admin-db",
                        username="admin-db",
                        password="db-pass",
                        tenant_id="tenant-db",
                        email="admin@example.com",
                        group_ids=["admin", "legal"],
                        roles=["document_manager"],
                        scopes=["rag:chat", "rag:documents"],
                        max_classification="secret",
                    )
                ]
            )
            auth_service = AuthService(settings, store=store)

            principal = auth_service.authenticate_login(
                "admin-db",
                "db-pass",
                login_type="admin",
            )

        self.assertEqual(principal.subject, "admin-db")
        self.assertEqual(principal.tenant_id, "tenant-db")
        self.assertEqual(principal.email, "admin@example.com")
        self.assertEqual(principal.group_ids, ["admin", "legal"])
        self.assertIn("rag:documents", principal.scopes)

    def test_auth_service_rejects_chat_only_user_for_admin_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SecuritySettings(
                auth_mode="jwt",
                auth_db_path=Path(temp_dir) / "auth.sqlite3",
                auth_seed_default_users=False,
            )
            store = AuthUserStore(settings.auth_db_path)
            store.seed_defaults(
                [
                    SeedAuthUser(
                        user_id="chat-db",
                        username="chat-db",
                        password="db-pass",
                        scopes=["rag:chat", "rag:retrieve"],
                    )
                ]
            )
            auth_service = AuthService(settings, store=store)

            with self.assertRaises(AuthError) as error:
                auth_service.authenticate_login(
                    "chat-db",
                    "db-pass",
                    login_type="admin",
                )

        self.assertEqual(error.exception.status_code, 403)


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
