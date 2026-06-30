from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.api.runtime import RAGRuntime
from app.api.schemas import ChatRequest, RetrievalRequest
from app.api.settings import APISettings
from app.indexing.config import IndexingConfig
from app.indexing.embeddings.hashing import HashingEmbeddingProvider
from app.indexing.indexer import EnterpriseIndexer
from tests.unit.test_retrieval import _load_sample_chunks


class APIRuntimeTest(unittest.TestCase):
    def test_runtime_retrieves_contexts_from_artifact_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = _runtime(Path(temp_dir))
            try:
                response = runtime.retrieve(
                    RetrievalRequest(
                        query="health insurance",
                        tenant_id="tenant-a",
                        group_ids=["hr"],
                        top_k=2,
                    ),
                    request_id="test-request",
                )
            finally:
                runtime.close()

        self.assertEqual(response.request_id, "test-request")
        self.assertTrue(response.contexts)
        self.assertLessEqual(len(response.contexts), 2)
        self.assertIn("reranker", response.stats)

    def test_runtime_chat_returns_answer_and_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = _runtime(Path(temp_dir))
            try:
                response = runtime.chat(
                    ChatRequest(
                        query="health insurance",
                        tenant_id="tenant-a",
                        group_ids=["hr"],
                        top_k=2,
                    ),
                    request_id="chat-request",
                )
            finally:
                runtime.close()

        self.assertEqual(response.request_id, "chat-request")
        self.assertEqual(response.provider, "mock")
        self.assertTrue(response.answer)
        self.assertTrue(response.citations)
        self.assertEqual(response.citations[0].label, "Context 1")

    def test_runtime_chat_request_can_choose_mock_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_data = _settings(Path(temp_dir)).model_dump()
            settings_data["llm_provider"] = "ollama"
            runtime = RAGRuntime(APISettings(**settings_data))
            try:
                response = runtime.chat(
                    ChatRequest(
                        query="health insurance",
                        tenant_id="tenant-a",
                        group_ids=["hr"],
                        llm_provider="mock",
                        model="mock-selected",
                    ),
                    request_id="provider-request",
                )
            finally:
                runtime.close()

        self.assertEqual(response.provider, "mock")
        self.assertEqual(response.model, "mock-selected")


def _runtime(temp_dir: Path) -> RAGRuntime:
    return RAGRuntime(_settings(temp_dir))


def _settings(temp_dir: Path) -> APISettings:
    index_path, chunks_path = _write_runtime_artifacts(temp_dir)
    return APISettings(
        index_path=index_path,
        chunks_path=chunks_path,
        llm_provider="mock",
        final_top_k=2,
    )


def _write_runtime_artifacts(temp_dir: Path) -> tuple[Path, Path]:
    chunking_result = _load_sample_chunks()
    indexer = EnterpriseIndexer(
        config=IndexingConfig(embedding_dimension=64),
        embedding_provider=HashingEmbeddingProvider(64),
    )
    indexing_result = indexer.index(chunking_result)
    chunks_path = temp_dir / "sample.chunks.json"
    index_path = temp_dir / "sample.index.json"
    chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
    index_path.write_text(indexing_result.to_json(), encoding="utf-8")
    return index_path, chunks_path


if __name__ == "__main__":
    unittest.main()
