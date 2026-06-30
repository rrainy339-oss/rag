from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.unit.test_api_runtime import _settings


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed.")
class FastAPIAppTest(unittest.TestCase):
    def test_chat_endpoint_returns_frontend_compatible_payload(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main

        with tempfile.TemporaryDirectory() as temp_dir:
            api_main.settings = _settings(Path(temp_dir))
            api_main.reset_runtime_for_tests()
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

    def test_document_upload_accepts_permission_fields(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main

        with tempfile.TemporaryDirectory() as temp_dir:
            settings_data = _settings(Path(temp_dir)).model_dump()
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
                    "documents_backend": "memory",
                    "documents_embedding_provider": "hashing",
                }
            )
            api_main.settings = api_main.APISettings(**settings_data)
            api_main.reset_runtime_for_tests()
            with TestClient(api_main.app) as client:
                with patch.object(
                    api_main.DocumentService,
                    "process_document",
                    return_value=None,
                ):
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

    def test_dev_auth_principal_overrides_request_acl_fields(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import main as api_main
        from app.security import SecuritySettings

        with tempfile.TemporaryDirectory() as temp_dir:
            api_main.settings = _settings(Path(temp_dir))
            api_main.security_settings = SecuritySettings(auth_mode="dev")
            api_main.audit_logger = api_main.AuditLogger.from_settings(
                api_main.security_settings
            )
            api_main.reset_runtime_for_tests()
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
