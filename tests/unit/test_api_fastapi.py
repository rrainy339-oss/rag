from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.unit.test_api_runtime import _settings, runtime_dependency_patches


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed.")
class FastAPIAppTest(unittest.TestCase):
    def test_chat_endpoint_returns_frontend_compatible_payload(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main

        with tempfile.TemporaryDirectory() as temp_dir:
            api_main.settings, indexing_result = _settings(Path(temp_dir))
            api_main.reset_runtime_for_tests()
            with runtime_dependency_patches(indexing_result):
                client = TestClient(api_main.app)
                response = client.post(
                    "/api/chat",
                    json={
                        "query": "health insurance",
                        "tenant_id": "tenant-a",
                        "group_ids": ["hr"],
                        "top_k": 2,
                        "model": "mock-model",
                        "temperature": 0.1,
                    },
                    headers={"x-request-id": "api-test"},
                )
                api_main.reset_runtime_for_tests()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request_id"], "api-test")
        self.assertIn("answer", payload)
        self.assertIn("content", payload)
        self.assertIn("citations", payload)
        self.assertIn("contexts", payload)

    def test_models_endpoint_returns_model_names(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main

        client = TestClient(api_main.app)
        with patch(
            "urllib.request.urlopen",
            return_value=_FakeHTTPResponse({"models": [{"name": "llama3.1"}]}),
        ):
            response = client.post(
                "/api/models",
                json={
                    "llm_provider": "ollama",
                    "base_url": "http://localhost:11434",
                },
                headers={"x-request-id": "models-test"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request_id"], "models-test")
        self.assertEqual(payload["models"], ["llama3.1"])

    def test_me_endpoint_returns_principal_capabilities(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main
        from app.security import SecuritySettings

        api_main.security_settings = SecuritySettings(auth_mode="dev")
        api_main.audit_logger = api_main.AuditLogger.from_settings(
            api_main.security_settings
        )
        client = TestClient(api_main.app)
        response = client.get(
            "/api/me",
            headers={
                "x-request-id": "me-test",
                "x-rag-subject": "admin-a",
                "x-rag-tenant-id": "tenant-a",
                "x-rag-user-id": "admin-a",
                "x-rag-group-ids": "admin,legal",
                "x-rag-scopes": "rag:chat,rag:documents",
                "x-rag-max-classification": "secret",
            },
        )
        api_main.security_settings = SecuritySettings(auth_mode="disabled")
        api_main.audit_logger = api_main.AuditLogger.from_settings(
            api_main.security_settings
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request_id"], "me-test")
        self.assertEqual(payload["subject"], "admin-a")
        self.assertEqual(payload["tenant_id"], "tenant-a")
        self.assertEqual(payload["group_ids"], ["admin", "legal"])
        self.assertTrue(payload["can_chat"])
        self.assertTrue(payload["can_manage_documents"])

    def test_jwt_login_issues_tokens_for_separate_frontends(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main
        from app.security import SecuritySettings

        api_main.security_settings = SecuritySettings(
            auth_mode="jwt",
            jwt_secret="test-secret",
            jwt_admin_username="admin",
            jwt_admin_password="admin-pass",
            jwt_user_username="user",
            jwt_user_password="user-pass",
        )
        api_main.audit_logger = api_main.AuditLogger.from_settings(
            api_main.security_settings
        )
        client = TestClient(api_main.app)

        admin_login = client.post(
            "/api/auth/admin/login",
            json={"username": "admin", "password": "admin-pass"},
            headers={"x-request-id": "admin-login-test"},
        )
        chat_login = client.post(
            "/api/auth/chat/login",
            json={"username": "user", "password": "user-pass"},
            headers={"x-request-id": "chat-login-test"},
        )

        admin_token = admin_login.json()["access_token"]
        user_token = chat_login.json()["access_token"]
        admin_me = client.get(
            "/api/me",
            headers={
                "authorization": f"Bearer {admin_token}",
                "x-request-id": "admin-me-test",
            },
        )
        user_me = client.get(
            "/api/me",
            headers={
                "authorization": f"Bearer {user_token}",
                "x-request-id": "user-me-test",
            },
        )
        user_documents = client.get(
            "/api/documents",
            headers={
                "authorization": f"Bearer {user_token}",
                "x-request-id": "user-documents-test",
            },
        )
        api_main.security_settings = SecuritySettings(auth_mode="disabled")
        api_main.audit_logger = api_main.AuditLogger.from_settings(
            api_main.security_settings
        )

        self.assertEqual(admin_login.status_code, 200)
        self.assertEqual(chat_login.status_code, 200)
        self.assertTrue(admin_me.json()["can_manage_documents"])
        self.assertTrue(admin_me.json()["can_chat"])
        self.assertFalse(user_me.json()["can_manage_documents"])
        self.assertTrue(user_me.json()["can_chat"])
        self.assertEqual(user_documents.status_code, 403)

    def test_document_upload_accepts_permission_fields(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main

        with tempfile.TemporaryDirectory() as temp_dir:
            settings, _ = _settings(Path(temp_dir))
            settings_data = settings.model_dump()
            settings_data.update(
                {
                    "documents_db_path": Path(temp_dir) / "documents.sqlite3",
                    "documents_storage_dir": Path(temp_dir) / "documents",
                    "documents_collection_index_path": Path(temp_dir)
                    / "collection"
                    / "index.json",
                    "documents_collection_chunks_path": Path(temp_dir)
                    / "collection"
                    / "chunks.json",
                }
            )
            api_main.settings = api_main.APISettings(**settings_data)
            api_main.reset_runtime_for_tests()
            with TestClient(api_main.app) as client:
                response = client.post(
                    "/api/documents",
                    files={"file": ("policy.txt", b"hello", "text/plain")},
                    data={
                        "title": "Policy",
                        "tenant_id": "tenant-a",
                        "owner_id": "owner-1",
                        "group_ids": "legal, compliance",
                        "principal_ids": "user-1, user-2",
                        "classification": "confidential",
                    },
                    headers={"x-request-id": "upload-test"},
                )
            api_main.reset_runtime_for_tests()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request_id"], "upload-test")
        self.assertEqual(payload["tenant_id"], "tenant-a")
        self.assertEqual(payload["owner_id"], "owner-1")
        self.assertEqual(payload["group_ids"], ["legal", "compliance"])
        self.assertEqual(payload["principal_ids"], ["user-1", "user-2"])
        self.assertEqual(payload["classification"], "confidential")
        self.assertEqual(payload["job"]["status"], "queued")
        self.assertEqual(payload["job"]["job_type"], "ingest")

    def test_document_job_management_endpoints(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main

        with tempfile.TemporaryDirectory() as temp_dir:
            settings, _ = _settings(Path(temp_dir))
            settings_data = settings.model_dump()
            settings_data.update(
                {
                    "documents_db_path": Path(temp_dir) / "documents.sqlite3",
                    "documents_storage_dir": Path(temp_dir) / "documents",
                    "documents_collection_index_path": Path(temp_dir)
                    / "collection"
                    / "index.json",
                    "documents_collection_chunks_path": Path(temp_dir)
                    / "collection"
                    / "chunks.json",
                }
            )
            api_main.settings = api_main.APISettings(**settings_data)
            api_main.reset_runtime_for_tests()
            with TestClient(api_main.app) as client:
                upload = client.post(
                    "/api/documents",
                    files={"file": ("policy.txt", b"hello", "text/plain")},
                    data={"title": "Policy"},
                )
                payload = upload.json()
                document_id = payload["document_id"]
                job_id = payload["job"]["job_id"]
                listed = client.get("/api/jobs")
                cancelled = client.post(f"/api/jobs/{job_id}/cancel")
                retried = client.post(f"/api/jobs/{job_id}/retry")
                reindex = client.post(f"/api/documents/{document_id}/reindex")
            api_main.reset_runtime_for_tests()

        self.assertEqual(upload.status_code, 200)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["jobs"][0]["job_id"], job_id)
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json()["status"], "queued")
        self.assertEqual(reindex.status_code, 200)
        self.assertEqual(reindex.json()["job"]["job_type"], "reindex")
        self.assertEqual(reindex.json()["job"]["status"], "queued")

    def test_dev_auth_principal_overrides_request_acl_fields(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main
        from app.security import SecuritySettings

        with tempfile.TemporaryDirectory() as temp_dir:
            api_main.settings, indexing_result = _settings(Path(temp_dir))
            api_main.security_settings = SecuritySettings(auth_mode="dev")
            api_main.audit_logger = api_main.AuditLogger.from_settings(
                api_main.security_settings
            )
            api_main.reset_runtime_for_tests()
            with runtime_dependency_patches(indexing_result):
                client = TestClient(api_main.app)
                response = client.post(
                    "/api/chat",
                    json={
                        "query": "health insurance",
                        "tenant_id": "tenant-a",
                        "group_ids": ["hr"],
                        "top_k": 2,
                        "llm_provider": "mock",
                    },
                    headers={
                        "x-request-id": "dev-auth-test",
                        "x-rag-tenant-id": "tenant-a",
                        "x-rag-group-ids": "engineering",
                        "x-rag-scopes": "rag:chat",
                    },
                )
                api_main.reset_runtime_for_tests()
            api_main.security_settings = SecuritySettings(auth_mode="disabled")
            api_main.audit_logger = api_main.AuditLogger.from_settings(
                api_main.security_settings
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["contexts"], [])
        self.assertIn("no_contexts", payload["warnings"])


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
