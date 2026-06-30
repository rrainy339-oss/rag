from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.schemas import ChunkingResult
from app.indexing.embeddings.bge import BGEM3EmbeddingProvider
from app.indexing.embeddings.hashing import HashingEmbeddingProvider
from app.indexing.schemas import IndexingResult
from app.indexing.sparse.memory_bm25 import MemoryBM25Store
from app.indexing.vectorstores.memory import MemoryVectorStore
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore, QdrantVectorStore
from app.retrieval.chunk_store import ChunkStore
from app.retrieval.config import RetrievalConfig
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dense import DenseRetriever
from app.retrieval.parent_expander import ParentExpander
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.qdrant_hybrid import (
    QdrantHybridRetrievalPipeline,
    QdrantHybridRetriever,
)
from app.retrieval.rerankers import RerankerConfig, RerankerProvider, build_reranker
from app.retrieval.schemas import RetrievalQuery
from app.retrieval.sparse import SparseRetriever


def main() -> int:
    args = _parse_args()
    args.backend = _normalize_backend(args.backend)

    indexing_result = IndexingResult.model_validate_json(
        args.index.read_text(encoding="utf-8")
    )
    chunking_result = ChunkingResult.model_validate_json(
        args.chunks.read_text(encoding="utf-8")
    )

    vector_store = _build_vector_store(args, indexing_result)
    embedding_provider = _build_embedding_provider(args, indexing_result)
    chunk_store = ChunkStore.from_chunking_result(chunking_result)
    reranker = _build_reranker(args)

    config = RetrievalConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        fusion_top_k=args.fusion_top_k,
        final_top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        retriever_oversample=args.retriever_oversample,
        release_dense_model_before_rerank=args.reranker in {"bge", "qwen"},
    )

    if args.backend == "qdrant_hybrid":
        pipeline = QdrantHybridRetrievalPipeline(
            hybrid_retriever=QdrantHybridRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            ),
            config=config,
            reranker=reranker,
            context_builder=ContextBuilder(ParentExpander(chunk_store)),
        )
    else:
        sparse_store = MemoryBM25Store()
        if args.backend == "memory":
            vector_store.upsert(indexing_result.records)
        sparse_store.upsert(indexing_result.records)
        pipeline = RetrievalPipeline(
            dense_retriever=DenseRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            ),
            sparse_retriever=SparseRetriever(sparse_store=sparse_store),
            config=config,
            reranker=reranker,
            context_builder=ContextBuilder(ParentExpander(chunk_store)),
        )

    response = pipeline.retrieve(
        RetrievalQuery(
            query=args.query,
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            group_ids=args.group_ids,
            max_classification=args.max_classification,
            metadata_filters=_parse_metadata_filters(args.metadata_filter),
        )
    )

    output_json = response.to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output_json)

    print(
        "retrieved "
        f"{len(response.contexts)} context(s), "
        f"{response.stats['fused_count']} fused candidate(s)"
    )
    if hasattr(vector_store, "close"):
        vector_store.close()
    return 0


def _build_embedding_provider(args: argparse.Namespace, result: IndexingResult) -> object:
    provider = args.embedding_provider
    if provider == "auto":
        if result.manifest.embedding_model == "hashing-embedding":
            provider = "hashing"
        elif "bge-m3" in result.manifest.embedding_model.lower():
            provider = "bge-m3"
        else:
            raise ValueError(
                "Cannot infer embedding provider from manifest model "
                f"{result.manifest.embedding_model!r}; pass --embedding-provider."
            )

    if provider == "hashing":
        return HashingEmbeddingProvider(result.manifest.embedding_dimension)
    if provider == "bge-m3":
        return BGEM3EmbeddingProvider(
            model_name=args.bge_model or result.manifest.embedding_model,
            use_fp16=args.bge_use_fp16,
            max_length=args.bge_max_length,
            batch_size=args.bge_batch_size,
            cache_dir=str(args.bge_cache_dir) if args.bge_cache_dir else None,
            sparse_top_n=args.sparse_top_n,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _build_vector_store(args: argparse.Namespace, result: IndexingResult) -> object:
    if args.backend == "memory":
        return MemoryVectorStore()
    if args.backend == "qdrant":
        qdrant_path = args.qdrant_path
        if args.qdrant_url is None and qdrant_path is None:
            qdrant_path = args.index.parent / "qdrant"
        return QdrantVectorStore(
            url=args.qdrant_url,
            path=qdrant_path,
            api_key=args.qdrant_api_key,
            collection_name=args.qdrant_collection,
            vector_size=result.manifest.embedding_dimension,
            timeout=args.qdrant_timeout,
        )
    if args.backend == "qdrant_hybrid":
        qdrant_path = args.qdrant_path
        if args.qdrant_url is None and qdrant_path is None:
            qdrant_path = args.index.parent / "qdrant_hybrid"
        return QdrantHybridVectorStore(
            url=args.qdrant_url,
            path=qdrant_path,
            api_key=args.qdrant_api_key,
            collection_name=args.qdrant_collection,
            vector_size=result.manifest.embedding_dimension,
            dense_vector_name=args.dense_vector_name,
            sparse_vector_name=args.sparse_vector_name,
            timeout=args.qdrant_timeout,
        )
    raise ValueError(f"Unsupported vector backend: {args.backend}")


def _build_reranker(args: argparse.Namespace) -> object:
    return build_reranker(
        RerankerConfig(
            provider=RerankerProvider(args.reranker),
            model_name=args.reranker_model,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            use_fp16=args.reranker_use_fp16,
            normalize=args.reranker_normalize,
            cache_dir=str(args.reranker_cache_dir) if args.reranker_cache_dir else None,
            device=args.reranker_device,
            api_key=args.reranker_api_key,
            timeout_seconds=args.reranker_timeout,
            max_tokens_per_doc=args.reranker_max_tokens_per_doc,
            instruction=args.reranker_instruction,
        )
    )


def _parse_metadata_filters(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata filter must use key=value syntax: {value}")
        key, raw = value.split("=", 1)
        filters[key] = raw
    return filters


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve contexts from local index JSON.")
    parser.add_argument("--index", type=Path, required=True, help="Index JSON file.")
    parser.add_argument("--chunks", type=Path, required=True, help="Chunks JSON file.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--output", type=Path, help="Optional response JSON output path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-top-k", type=int, default=40)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--sparse-top-k", type=int, default=50)
    parser.add_argument("--fusion-top-k", type=int, default=60)
    parser.add_argument("--retriever-oversample", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=["memory", "qdrant", "qdrant_hybrid", "qdrant-hybrid"],
        default="memory",
        help="Vector store backend.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "hashing", "bge-m3"],
        default="auto",
    )
    parser.add_argument("--bge-model")
    parser.add_argument("--bge-batch-size", type=int, default=12)
    parser.add_argument("--bge-max-length", type=int, default=8192)
    parser.add_argument("--bge-cache-dir", type=Path)
    parser.add_argument("--sparse-top-n", type=int, default=512)
    parser.add_argument(
        "--bge-use-fp16",
        action="store_true",
        help="Use fp16 for BGE-M3, useful on compatible GPUs.",
    )
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--qdrant-collection", default="rag_chunks")
    parser.add_argument("--qdrant-timeout", type=int)
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="sparse")
    parser.add_argument(
        "--reranker",
        choices=[provider.value for provider in RerankerProvider],
        default=RerankerProvider.NOOP.value,
    )
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int)
    parser.add_argument("--reranker-cache-dir", type=Path)
    parser.add_argument("--reranker-device")
    parser.add_argument("--reranker-api-key")
    parser.add_argument("--reranker-timeout", type=float, default=60.0)
    parser.add_argument("--reranker-max-tokens-per-doc", type=int, default=4096)
    parser.add_argument("--reranker-instruction")
    parser.add_argument(
        "--reranker-use-fp16",
        action="store_true",
        help="Use fp16 for local rerankers on compatible GPUs.",
    )
    parser.add_argument(
        "--reranker-normalize",
        action="store_true",
        help="Normalize local reranker scores when supported.",
    )
    parser.add_argument("--tenant-id")
    parser.add_argument("--user-id")
    parser.add_argument("--group-id", dest="group_ids", action="append", default=[])
    parser.add_argument("--max-classification")
    parser.add_argument("--metadata-filter", action="append", default=[])
    return parser.parse_args()


def _normalize_backend(value: str) -> str:
    return "qdrant_hybrid" if value == "qdrant-hybrid" else value


if __name__ == "__main__":
    raise SystemExit(main())
