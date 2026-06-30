from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from app.answering import (
    AnswerPipeline,
    ContextPacker,
    MockLLMProvider,
    list_ollama_models,
    list_openai_compatible_models,
)
from app.answering.schemas import ConfidenceLevel
from app.chunking.schemas import ChunkType
from app.retrieval.schemas import ContextItem, QueryType, RetrievalResponse


class AnswerPipelineTest(unittest.TestCase):
    def test_answer_uses_mock_llm_and_extracts_context_citations(self) -> None:
        pipeline = AnswerPipeline(
            llm_provider=MockLLMProvider(
                "现有方法主要包括静态分析和动态分析。[Context 1] "
                "局限是覆盖不足。[Context 2]"
            ),
            context_packer=ContextPacker(top_contexts=2),
        )

        result = pipeline.answer(_retrieval_response())

        self.assertEqual(result.confidence, ConfidenceLevel.HIGH)
        self.assertEqual(result.used_context_ids, ["ctx-1", "ctx-2"])
        self.assertEqual(len(result.citations), 2)
        self.assertEqual(result.citations[0].source_chunk_id, "chunk-1")
        self.assertEqual(result.warnings, [])

    def test_answer_warns_when_llm_omits_citations(self) -> None:
        pipeline = AnswerPipeline(
            llm_provider=MockLLMProvider("这是没有引用的回答。"),
            context_packer=ContextPacker(top_contexts=2),
        )

        result = pipeline.answer(_retrieval_response())

        self.assertEqual(result.confidence, ConfidenceLevel.LOW)
        self.assertEqual(result.used_context_ids, [])
        self.assertIn("answer_has_no_context_citations", result.warnings)

    def test_answer_refuses_when_no_contexts_exist(self) -> None:
        pipeline = AnswerPipeline(llm_provider=MockLLMProvider("should not be used"))

        result = pipeline.answer(
            RetrievalResponse(
                query="问题",
                query_type=QueryType.GENERAL,
                contexts=[],
                candidates=[],
            )
        )

        self.assertEqual(result.confidence, ConfidenceLevel.LOW)
        self.assertIn("不足以回答", result.answer)
        self.assertIn("no_contexts", result.warnings)

    def test_lists_ollama_models(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_FakeHTTPResponse(
                {"models": [{"name": "llama3.1"}, {"model": "qwen2.5"}]}
            ),
        ):
            models = list_ollama_models(base_url="http://localhost:11434")

        self.assertEqual(models, ["llama3.1", "qwen2.5"])

    def test_lists_openai_compatible_models(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_FakeHTTPResponse(
                {"data": [{"id": "gpt-4o-mini"}, {"id": "qwen-plus"}]}
            ),
        ):
            models = list_openai_compatible_models(
                base_url="https://api.example.test/v1",
                api_key="test-key",
            )

        self.assertEqual(models, ["gpt-4o-mini", "qwen-plus"])


def _retrieval_response() -> RetrievalResponse:
    return RetrievalResponse(
        query="现有链上程序安全分析方法主要分为哪三类？",
        query_type=QueryType.GENERAL,
        contexts=[
            ContextItem(
                context_id="ctx-1",
                source_chunk_id="chunk-1",
                chunk_type=ChunkType.CHILD,
                text="静态分析方法通过分析程序代码发现风险，但难以覆盖运行时行为。",
                contextual_text="静态分析方法通过分析程序代码发现风险。",
                score=0.9,
                section_path=["相关工作"],
            ),
            ContextItem(
                context_id="ctx-2",
                source_chunk_id="chunk-2",
                chunk_type=ChunkType.CHILD,
                text="动态分析方法依赖执行路径，局限是路径覆盖不足。",
                contextual_text="动态分析方法依赖执行路径。",
                score=0.8,
                section_path=["相关工作"],
            ),
        ],
        candidates=[],
    )


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
