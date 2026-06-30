from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.api.runtime import RAGRuntime
from app.api.schemas import ChatRequest, RetrievalRequest
from app.api.settings import APISettings
from app.indexing.config import IndexingConfig
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.schemas import IndexingResult
from tests.unit.hybrid_fakes import CapturingHybridVectorStore, FakeHybridEmbeddingProvider
from tests.unit.test_retrieval import _load_sample_chunks


class APIRuntimeTest(unittest.TestCase):
    def test_runtime_retrieves_contexts_from_qdrant_hybrid_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, indexing_result = _settings(Path(temp_dir))
            with runtime_dependency_patches(indexing_result):
                runtime = RAGRuntime(settings)
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
        self.assertEqual(response.stats["retriever"], "qdrant_hybrid")
        self.assertIn("reranker", response.stats)

    def test_runtime_chat_returns_answer_and_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings, indexing_result = _settings(Path(temp_dir))
            with runtime_dependency_patches(indexing_result):
                runtime = RAGRuntime(settings)
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
            settings, indexing_result = _settings(Path(temp_dir))
            settings_data = settings.model_dump()
            settings_data["llm_provider"] = "ollama"
            with runtime_dependency_patches(indexing_result):
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


def _settings(temp_dir: Path) -> tuple[APISettings, IndexingResult]:
    index_path, chunks_path, indexing_result = _write_runtime_artifacts(temp_dir)
    return (
        APISettings(
            index_path=index_path,
            chunks_path=chunks_path,
            llm_provider="mock",
            final_top_k=2,
            qdrant_collection="test_runtime_chunks",
        ),
        indexing_result,
    )


def _write_runtime_artifacts(temp_dir: Path) -> tuple[Path, Path, IndexingResult]:
    chunking_result = _load_sample_chunks()
    provider = FakeHybridEmbeddingProvider()
    vector_store = CapturingHybridVectorStore()
    indexer = EnterpriseIndexer(
        embedding_provider=provider,
        vector_store=vector_store,
        config=IndexingConfig(
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension,
        ),
    )
    indexing_result = indexer.index(chunking_result)
    chunks_path = temp_dir / "sample.chunks.json"
    index_path = temp_dir / "sample.index.json"
    chunks_path.write_text(chunking_result.to_json(), encoding="utf-8")
    index_path.write_text(indexing_result.to_json(), encoding="utf-8")
    return index_path, chunks_path, indexing_result


@contextmanager
def runtime_dependency_patches(indexing_result: IndexingResult):
    store = CapturingHybridVectorStore(indexing_result.records)
    with patch("app.api.runtime.BGEM3EmbeddingProvider", FakeHybridEmbeddingProvider):
        with patch("app.api.runtime.QdrantHybridVectorStore", return_value=store):
            yield


if __name__ == "__main__":
    unittest.main()
