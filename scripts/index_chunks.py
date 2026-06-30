from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.schemas import ChunkingResult
from app.indexing.schemas import IndexBackend
from app.indexing.config import IndexingConfig
from app.indexing.embeddings.bge import BGEM3EmbeddingProvider
from app.indexing.embeddings.hashing import HashingEmbeddingProvider
from app.indexing.indexer import EnterpriseIndexer
from app.indexing.manifest import JsonManifestStore
from app.indexing.sparse.memory_bm25 import MemoryBM25Store
from app.indexing.vectorstores.memory import MemoryVectorStore
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore, QdrantVectorStore


def main() -> int:
    args = _parse_args()
    args.backend = _normalize_backend(args.backend)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_provider = _build_embedding_provider(args)
    vector_store = _build_vector_store(args, output_dir, embedding_provider.dimension)
    sparse_store = (
        None
        if args.backend == IndexBackend.QDRANT_HYBRID.value
        else MemoryBM25Store()
    )
    manifest_store = JsonManifestStore(output_dir / "index_manifest.json")
    config = IndexingConfig(
        backend=IndexBackend(args.backend),
        embedding_model=embedding_provider.model_name,
        embedding_dimension=embedding_provider.dimension,
        embedding_batch_size=args.embedding_batch_size,
        dense_vector_name=args.dense_vector_name,
        sparse_vector_name=args.sparse_vector_name,
        sparse_top_n=args.sparse_top_n,
        force_reindex=not args.skip_unchanged,
    )
    indexer = EnterpriseIndexer(
        config=config,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        sparse_store=sparse_store,
        manifest_store=manifest_store,
    )

    input_files = _iter_input_files(args.input)
    for input_file in input_files:
        chunking_result = ChunkingResult.model_validate_json(
            input_file.read_text(encoding="utf-8")
        )
        result = indexer.index(chunking_result)
        output_path = output_dir / input_file.name.replace(".chunks.json", ".index.json")
        if output_path == output_dir / input_file.name:
            output_path = output_dir / f"{input_file.stem}.index.json"
        output_path.write_text(result.to_json(), encoding="utf-8")
        print(f"wrote {output_path}")
        print(
            f"{result.status.value}: indexed={result.indexed_count} "
            f"skipped={result.skipped_count} deleted={result.deleted_count}"
        )

    print(f"indexed {len(input_files)} chunk file(s)")
    print(f"{args.backend} vector records: {vector_store.count()}")
    if sparse_store is not None:
        print(f"memory BM25 records: {sparse_store.count()}")
    if hasattr(vector_store, "close"):
        vector_store.close()
    return 0


def _build_embedding_provider(args: argparse.Namespace) -> object:
    if args.embedding_provider == "hashing":
        return HashingEmbeddingProvider(args.embedding_dimension)
    if args.embedding_provider == "bge-m3":
        return BGEM3EmbeddingProvider(
            model_name=args.bge_model,
            use_fp16=args.bge_use_fp16,
            max_length=args.bge_max_length,
            batch_size=args.bge_batch_size,
            cache_dir=str(args.bge_cache_dir) if args.bge_cache_dir else None,
            sparse_top_n=args.sparse_top_n,
        )
    raise ValueError(f"Unsupported embedding provider: {args.embedding_provider}")


def _build_vector_store(
    args: argparse.Namespace,
    output_dir: Path,
    embedding_dimension: int,
) -> object:
    if args.backend == IndexBackend.MEMORY.value:
        return MemoryVectorStore()
    if args.backend == IndexBackend.QDRANT.value:
        qdrant_path = args.qdrant_path
        if args.qdrant_url is None and qdrant_path is None:
            qdrant_path = output_dir / "qdrant"
        return QdrantVectorStore(
            url=args.qdrant_url,
            path=qdrant_path,
            api_key=args.qdrant_api_key,
            collection_name=args.qdrant_collection,
            vector_size=embedding_dimension,
            timeout=args.qdrant_timeout,
        )
    if args.backend == IndexBackend.QDRANT_HYBRID.value:
        qdrant_path = args.qdrant_path
        if args.qdrant_url is None and qdrant_path is None:
            qdrant_path = output_dir / "qdrant_hybrid"
        return QdrantHybridVectorStore(
            url=args.qdrant_url,
            path=qdrant_path,
            api_key=args.qdrant_api_key,
            collection_name=args.qdrant_collection,
            vector_size=embedding_dimension,
            dense_vector_name=args.dense_vector_name,
            sparse_vector_name=args.sparse_vector_name,
            timeout=args.qdrant_timeout,
        )
    raise ValueError(f"Unsupported index backend: {args.backend}")


def _iter_input_files(path: Path) -> list[Path]:
    resolved = path.resolve()
    if resolved.is_file():
        return [resolved]
    return sorted(resolved.glob("*.chunks.json"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index chunk JSON files.")
    parser.add_argument("--input", type=Path, required=True, help="Chunks JSON file or folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "bge-m3"],
        default="hashing",
        help="Embedding implementation. Use bge-m3 for real dense embeddings.",
    )
    parser.add_argument("--embedding-dimension", type=int, default=384)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in IndexBackend] + ["qdrant-hybrid"],
        default=IndexBackend.MEMORY.value,
        help="Vector store backend.",
    )
    parser.add_argument("--bge-model", default="BAAI/bge-m3")
    parser.add_argument("--bge-batch-size", type=int, default=12)
    parser.add_argument("--bge-max-length", type=int, default=8192)
    parser.add_argument("--bge-cache-dir", type=Path)
    parser.add_argument(
        "--sparse-top-n",
        type=int,
        default=512,
        help="Keep the top N BGE-M3 sparse weights per chunk for qdrant_hybrid.",
    )
    parser.add_argument(
        "--bge-use-fp16",
        action="store_true",
        help="Use fp16 for BGE-M3, useful on compatible GPUs.",
    )
    parser.add_argument("--qdrant-url", help="Qdrant server URL, for example http://localhost:6333.")
    parser.add_argument("--qdrant-path", type=Path, help="Local persistent Qdrant storage path.")
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--qdrant-collection", default="rag_chunks")
    parser.add_argument("--qdrant-timeout", type=int)
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="sparse")
    parser.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="Skip indexing when the manifest already matches the chunks and model.",
    )
    return parser.parse_args()


def _normalize_backend(value: str) -> str:
    return "qdrant_hybrid" if value == "qdrant-hybrid" else value


if __name__ == "__main__":
    raise SystemExit(main())
