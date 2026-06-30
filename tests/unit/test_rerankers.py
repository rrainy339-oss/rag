from __future__ import annotations

import unittest

from app.chunking.schemas import ChunkType
from app.indexing.schemas import IndexRecord
from app.retrieval.rerankers import RerankerConfig, RerankerProvider, build_reranker
from app.retrieval.rerankers.bge import BGEReranker, _ensure_prepare_for_model
from app.retrieval.rerankers.noop import NoopReranker
from app.retrieval.rerankers.qwen import QwenReranker
from app.retrieval.schemas import AnalyzedQuery, RetrievalCandidate, RetrievalQuery


class RerankerAdaptersTest(unittest.TestCase):
    def test_bge_reranker_applies_cross_encoder_scores(self) -> None:
        reranker = BGEReranker(model_name="fake-bge", model=_FakeBGEModel())
        candidates = [
            _candidate("a", "unrelated policy", fused_score=0.9),
            _candidate("b", "target evidence about remote work", fused_score=0.1),
        ]

        ranked = reranker.rerank(_query("target"), candidates, top_k=2)

        self.assertEqual([item.record.chunk_id for item in ranked], ["b", "a"])
        self.assertEqual(ranked[0].rerank_score, 3.0)
        self.assertEqual(ranked[0].final_score, 3.0)
        self.assertEqual(ranked[0].metadata["reranker"], "fake-bge")
        self.assertEqual(ranked[0].metadata["pre_rerank_score"], 0.1)

    def test_qwen_reranker_applies_cross_encoder_scores(self) -> None:
        reranker = QwenReranker(model_name="fake-qwen", model=_FakeQwenModel())
        candidates = [
            _candidate("a", "target document", fused_score=0.1),
            _candidate("b", "unrelated document", fused_score=0.9),
        ]

        ranked = reranker.rerank(_query("target"), candidates, top_k=1)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].record.chunk_id, "a")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[0].metadata["reranker"], "fake-qwen")

    def test_factory_keeps_noop_as_default(self) -> None:
        reranker = build_reranker(RerankerConfig(provider=RerankerProvider.NOOP))

        self.assertIsInstance(reranker, NoopReranker)
        self.assertEqual(reranker.model_name, "noop-reranker")

    def test_bge_reranker_patches_transformers_v5_tokenizer_api(self) -> None:
        model = _FakeModelWithTokenizer()

        _ensure_prepare_for_model(model)
        item = model.tokenizer.prepare_for_model(
            [10, 11],
            [20, 21, 22],
            truncation="only_second",
            max_length=7,
            padding=False,
        )

        self.assertEqual(item["input_ids"], [0, 10, 11, 2, 2, 20, 2])
        self.assertEqual(item["attention_mask"], [1, 1, 1, 1, 1, 1, 1])


class _FakeBGEModel:
    def compute_score(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [3.0 if "target evidence" in passage else -1.0 for _, passage in pairs]


class _FakeQwenModel:
    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        return [2.0 if "target" in passage else -2.0 for _, passage in pairs]


class _FakeTokenizerWithoutPrepare:
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1


class _FakeModelWithTokenizer:
    tokenizer = _FakeTokenizerWithoutPrepare()


def _query(text: str) -> AnalyzedQuery:
    return AnalyzedQuery(query=RetrievalQuery(query=text))


def _candidate(chunk_id: str, text: str, *, fused_score: float) -> RetrievalCandidate:
    record = IndexRecord(
        record_id=f"idx_{chunk_id}",
        chunk_id=chunk_id,
        document_id="doc-test",
        chunk_type=ChunkType.CHILD,
        text=text,
        contextual_text=text,
        vector=[0.0, 1.0],
        content_hash=f"{chunk_id}-hash",
        embedding_model="test",
        embedding_dimension=2,
        index_version="test",
    )
    return RetrievalCandidate(record=record, fused_score=fused_score, final_score=fused_score)


if __name__ == "__main__":
    unittest.main()
