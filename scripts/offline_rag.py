from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.schemas import ChunkingResult
from app.connectors.local_file import DEFAULT_SUPPORTED_EXTENSIONS
from app.indexing.schemas import (
    IndexBackend,
    IndexManifest,
    IndexingResult,
    IndexingStatus,
)
from app.parsing.docling_parser import SUPPORTED_PDF_BACKENDS
from app.retrieval.rerankers import RerankerProvider


DEFAULT_DOCLING_ARTIFACTS = Path(r"D:\AI\docling")
DEFAULT_HF_HOME = Path(r"D:\AI\huggingface")


def main() -> int:
    args = _parse_args()
    args.backend = _normalize_backend(args.backend)

    input_path = _project_path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    source_files = _iter_source_files(input_path, recursive=not args.no_recursive)
    if not source_files:
        raise SystemExit(f"No supported source files found under: {input_path}")
    _ensure_unique_stems(source_files)

    run_name = args.run_name or _safe_file_name(input_path.stem)
    run_dir = _project_path(args.work_dir) / run_name
    parsed_dir = _project_path(args.parsed_output) if args.parsed_output else run_dir / "parsed"
    chunks_dir = _project_path(args.chunks_output) if args.chunks_output else run_dir / "chunks"
    index_dir = _project_path(args.index_output) if args.index_output else run_dir / "index"
    retrieval_dir = (
        _project_path(args.retrieval_output)
        if args.retrieval_output
        else run_dir / "retrieval"
    )
    retrieval_output = _project_path(args.output) if args.output else (
        retrieval_dir / f"{run_name}.retrieval.json"
    )
    answer_dir = run_dir / "answers"
    answer_output = _project_path(args.answer_output) if args.answer_output else (
        answer_dir / f"{run_name}.answer.json"
    )
    answer_markdown_output = (
        _project_path(args.answer_markdown_output)
        if args.answer_markdown_output
        else answer_dir / f"{run_name}.answer.md"
    )

    docling_artifacts_path = _first_existing_path(
        args.docling_artifacts_path,
        DEFAULT_DOCLING_ARTIFACTS,
    )
    hf_home = _first_existing_path(args.hf_home, DEFAULT_HF_HOME)
    hf_cache_dir = _project_path(args.hf_cache_dir) if args.hf_cache_dir else None
    if hf_cache_dir is None and hf_home is not None:
        hf_cache_dir = hf_home / "hub"

    bge_model = None
    if args.embedding_provider == "bge-m3":
        bge_model = _resolve_model_path(
            model_arg=args.bge_model,
            cache_dir=hf_cache_dir,
            repo_id=args.bge_model_id,
            offline=args.offline,
            flag_name="bge-model",
        )

    reranker_model = None
    if args.reranker == RerankerProvider.BGE.value:
        reranker_model = _resolve_model_path(
            model_arg=args.reranker_model,
            cache_dir=hf_cache_dir,
            repo_id=args.reranker_model_id,
            offline=args.offline,
            flag_name="reranker-model",
        )

    env = _build_env(
        hf_home=hf_home,
        hf_cache_dir=hf_cache_dir,
        docling_artifacts_path=docling_artifacts_path,
        offline=args.offline,
    )

    qdrant_path = _project_path(args.qdrant_path) if args.qdrant_path else None
    if (
        args.backend in {IndexBackend.QDRANT.value, IndexBackend.QDRANT_HYBRID.value}
        and args.qdrant_url is None
    ):
        qdrant_path = qdrant_path or run_dir / (
            "qdrant_hybrid"
            if args.backend == IndexBackend.QDRANT_HYBRID.value
            else "qdrant"
        )
    qdrant_collection = args.qdrant_collection or _safe_collection_name(
        f"{run_name}_hybrid"
        if args.backend == IndexBackend.QDRANT_HYBRID.value
        else run_name
    )

    _run_step(
        "parse",
        _parse_command(
            input_path=input_path,
            output_dir=parsed_dir,
            pdf_backend=args.pdf_backend,
            ocr=args.ocr,
            artifacts_path=docling_artifacts_path,
            no_recursive=args.no_recursive,
            permissions=_project_path(args.permissions) if args.permissions else None,
            document_tenant_id=args.document_tenant_id,
            document_allowed_user_ids=args.document_allowed_user_ids,
            document_allowed_group_ids=args.document_allowed_group_ids,
            document_classification=args.document_classification,
        ),
        env,
    )

    parsed_files = [parsed_dir / f"{path.stem}.parsed.json" for path in source_files]
    _require_files("parsed", parsed_files)

    for parsed_file in parsed_files:
        _run_step(
            f"chunk {parsed_file.name}",
            _chunk_command(parsed_file=parsed_file, output_dir=chunks_dir, args=args),
            env,
        )

    chunk_files = [chunks_dir / f"{path.stem}.chunks.json" for path in source_files]
    _require_files("chunks", chunk_files)

    for chunk_file in chunk_files:
        _run_step(
            f"index {chunk_file.name}",
            _index_command(
                chunk_file=chunk_file,
                output_dir=index_dir,
                args=args,
                bge_model=bge_model,
                bge_cache_dir=hf_cache_dir,
                qdrant_path=qdrant_path,
                qdrant_collection=qdrant_collection,
            ),
            env,
        )

    index_files = [index_dir / f"{path.stem}.index.json" for path in source_files]
    _require_files("index", index_files)

    retrieval_chunks = _combine_chunks_if_needed(chunk_files, chunks_dir, run_name)
    retrieval_index = _combine_indexes_if_needed(index_files, index_dir, run_name)

    _run_step(
        "retrieve",
        _retrieve_command(
            index_file=retrieval_index,
            chunks_file=retrieval_chunks,
            output_file=retrieval_output,
            args=args,
            bge_model=bge_model,
            bge_cache_dir=hf_cache_dir,
            reranker_model=reranker_model,
            reranker_cache_dir=hf_cache_dir,
            qdrant_path=qdrant_path,
            qdrant_collection=qdrant_collection,
        ),
        env,
    )

    if args.print_contexts:
        _print_context_texts(retrieval_output)

    if args.answer:
        _run_step(
            "answer",
            _answer_command(
                retrieval_file=retrieval_output,
                output_file=answer_output,
                markdown_output_file=answer_markdown_output,
                args=args,
            ),
            env,
        )

    print("\nDone.")
    print(f"Parsed: {parsed_dir}")
    print(f"Chunks: {chunks_dir}")
    print(f"Index: {index_dir}")
    print(f"Retrieval JSON: {retrieval_output}")
    if args.answer:
        print(f"Answer JSON: {answer_output}")
        print(f"Answer Markdown: {answer_markdown_output}")
    return 0


def _parse_command(
    *,
    input_path: Path,
    output_dir: Path,
    pdf_backend: str,
    ocr: bool,
    artifacts_path: Path | None,
    no_recursive: bool,
    permissions: Path | None,
    document_tenant_id: str | None,
    document_allowed_user_ids: list[str],
    document_allowed_group_ids: list[str],
    document_classification: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "parse_local.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--pdf-backend",
        pdf_backend,
        "--ocr" if ocr else "--no-ocr",
    ]
    if artifacts_path is not None:
        command.extend(["--artifacts-path", str(artifacts_path)])
    if no_recursive:
        command.append("--no-recursive")
    if permissions is not None:
        command.extend(["--permissions", str(permissions)])
    if document_tenant_id:
        command.extend(["--tenant-id", document_tenant_id])
    for user_id in document_allowed_user_ids:
        command.extend(["--allowed-user-id", user_id])
    for group_id in document_allowed_group_ids:
        command.extend(["--allowed-group-id", group_id])
    if document_classification:
        command.extend(["--classification", document_classification])
    return command


def _chunk_command(*, parsed_file: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "chunk_parsed.py"),
        "--input",
        str(parsed_file),
        "--output",
        str(output_dir),
        "--child-target-tokens",
        str(args.child_target_tokens),
        "--child-max-tokens",
        str(args.child_max_tokens),
        "--parent-target-tokens",
        str(args.parent_target_tokens),
        "--parent-max-tokens",
        str(args.parent_max_tokens),
    ]


def _index_command(
    *,
    chunk_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
    bge_model: str | None,
    bge_cache_dir: Path | None,
    qdrant_path: Path | None,
    qdrant_collection: str,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "index_chunks.py"),
        "--input",
        str(chunk_file),
        "--output",
        str(output_dir),
        "--embedding-provider",
        args.embedding_provider,
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        "--backend",
        args.backend,
    ]
    if args.embedding_provider == "hashing":
        command.extend(["--embedding-dimension", str(args.embedding_dimension)])
    if args.embedding_provider == "bge-m3":
        command.extend(
            [
                "--bge-model",
                str(bge_model),
                "--bge-batch-size",
                str(args.bge_batch_size),
                "--bge-max-length",
                str(args.bge_max_length),
                "--sparse-top-n",
                str(args.sparse_top_n),
            ]
        )
        if bge_cache_dir is not None:
            command.extend(["--bge-cache-dir", str(bge_cache_dir)])
        if args.bge_use_fp16:
            command.append("--bge-use-fp16")
    command.extend(_qdrant_args(args, qdrant_path, qdrant_collection))
    return command


def _retrieve_command(
    *,
    index_file: Path,
    chunks_file: Path,
    output_file: Path,
    args: argparse.Namespace,
    bge_model: str | None,
    bge_cache_dir: Path | None,
    reranker_model: str | None,
    reranker_cache_dir: Path | None,
    qdrant_path: Path | None,
    qdrant_collection: str,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "retrieve.py"),
        "--index",
        str(index_file),
        "--chunks",
        str(chunks_file),
        "--query",
        args.query,
        "--output",
        str(output_file),
        "--top-k",
        str(args.top_k),
        "--rerank-top-k",
        str(args.rerank_top_k),
        "--dense-top-k",
        str(args.dense_top_k),
        "--sparse-top-k",
        str(args.sparse_top_k),
        "--fusion-top-k",
        str(args.fusion_top_k),
        "--retriever-oversample",
        str(args.retriever_oversample),
        "--backend",
        args.backend,
        "--embedding-provider",
        args.embedding_provider,
        "--reranker",
        args.reranker,
    ]
    if args.embedding_provider == "bge-m3":
        command.extend(
            [
                "--bge-model",
                str(bge_model),
                "--bge-batch-size",
                str(args.bge_batch_size),
                "--bge-max-length",
                str(args.bge_max_length),
                "--sparse-top-n",
                str(args.sparse_top_n),
            ]
        )
        if bge_cache_dir is not None:
            command.extend(["--bge-cache-dir", str(bge_cache_dir)])
        if args.bge_use_fp16:
            command.append("--bge-use-fp16")
    command.extend(_qdrant_args(args, qdrant_path, qdrant_collection))
    if args.reranker != RerankerProvider.NOOP.value:
        if reranker_model is not None:
            command.extend(["--reranker-model", str(reranker_model)])
        command.extend(
            [
                "--reranker-batch-size",
                str(args.reranker_batch_size),
                "--reranker-max-tokens-per-doc",
                str(args.reranker_max_tokens_per_doc),
            ]
        )
        if args.reranker_max_length is not None:
            command.extend(["--reranker-max-length", str(args.reranker_max_length)])
        if reranker_cache_dir is not None:
            command.extend(["--reranker-cache-dir", str(reranker_cache_dir)])
        if args.reranker_device:
            command.extend(["--reranker-device", args.reranker_device])
        if args.reranker_use_fp16:
            command.append("--reranker-use-fp16")
        if args.reranker_normalize:
            command.append("--reranker-normalize")
    if args.tenant_id:
        command.extend(["--tenant-id", args.tenant_id])
    if args.user_id:
        command.extend(["--user-id", args.user_id])
    for group_id in args.group_ids:
        command.extend(["--group-id", group_id])
    if args.max_classification:
        command.extend(["--max-classification", args.max_classification])
    for metadata_filter in args.metadata_filter:
        command.extend(["--metadata-filter", metadata_filter])
    return command


def _qdrant_args(
    args: argparse.Namespace,
    qdrant_path: Path | None,
    qdrant_collection: str,
) -> list[str]:
    if args.backend not in {IndexBackend.QDRANT.value, IndexBackend.QDRANT_HYBRID.value}:
        return []
    command: list[str] = ["--qdrant-collection", qdrant_collection]
    if args.qdrant_url:
        command.extend(["--qdrant-url", args.qdrant_url])
    elif qdrant_path is not None:
        command.extend(["--qdrant-path", str(qdrant_path)])
    if args.qdrant_api_key:
        command.extend(["--qdrant-api-key", args.qdrant_api_key])
    if args.qdrant_timeout is not None:
        command.extend(["--qdrant-timeout", str(args.qdrant_timeout)])
    if args.backend == IndexBackend.QDRANT_HYBRID.value:
        command.extend(
            [
                "--dense-vector-name",
                args.dense_vector_name,
                "--sparse-vector-name",
                args.sparse_vector_name,
            ]
        )
    return command


def _answer_command(
    *,
    retrieval_file: Path,
    output_file: Path,
    markdown_output_file: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "answer.py"),
        "--retrieval",
        str(retrieval_file),
        "--query",
        args.query,
        "--output",
        str(output_file),
        "--markdown-output",
        str(markdown_output_file),
        "--top-contexts",
        str(args.answer_top_contexts),
        "--max-context-chars",
        str(args.answer_max_context_chars),
        "--max-chars-per-context",
        str(args.answer_max_chars_per_context),
        "--language",
        args.answer_language,
        "--llm-provider",
        args.llm_provider,
    ]
    if args.include_prompt:
        command.append("--include-prompt")
    if args.llm_provider == "mock":
        command.extend(["--mock-response", args.mock_response])
        return command

    command.extend(
        [
            "--base-url",
            args.llm_base_url,
            "--model",
            args.llm_model,
            "--temperature",
            str(args.llm_temperature),
            "--top-p",
            str(args.llm_top_p),
            "--max-tokens",
            str(args.llm_max_tokens),
            "--timeout",
            str(args.llm_timeout),
        ]
    )
    if args.llm_api_key:
        command.extend(["--api-key", args.llm_api_key])
    if args.llm_api_key_env:
        command.extend(["--api-key-env", args.llm_api_key_env])
    return command


def _run_step(name: str, command: list[str], env: dict[str, str]) -> None:
    print(f"\n== {name} ==", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _combine_chunks_if_needed(files: list[Path], output_dir: Path, run_name: str) -> Path:
    if len(files) == 1:
        return files[0]

    results = [
        ChunkingResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in files
    ]
    chunks = [chunk for result in results for chunk in result.chunks]
    combined = ChunkingResult(
        document_id=run_name,
        chunks=chunks,
        stats={
            "combined": True,
            "document_count": len(results),
            "chunk_count": len(chunks),
            "source_files": [str(path) for path in files],
            "source_document_ids": [result.document_id for result in results],
        },
    )
    output_path = output_dir / f"{run_name}.combined.chunks.json"
    output_path.write_text(combined.to_json(), encoding="utf-8")
    print(f"wrote {output_path}")
    return output_path


def _combine_indexes_if_needed(files: list[Path], output_dir: Path, run_name: str) -> Path:
    if len(files) == 1:
        return files[0]

    results = [
        IndexingResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in files
    ]
    first = results[0]
    records = [record for result in results for record in result.records]
    for result in results[1:]:
        if result.manifest.embedding_model != first.manifest.embedding_model:
            raise SystemExit("Cannot combine indexes with different embedding models.")
        if result.manifest.embedding_dimension != first.manifest.embedding_dimension:
            raise SystemExit("Cannot combine indexes with different embedding dimensions.")
        if result.manifest.index_version != first.manifest.index_version:
            raise SystemExit("Cannot combine indexes with different index versions.")
        if result.manifest.backend != first.manifest.backend:
            raise SystemExit("Cannot combine indexes with different backends.")

    manifest = IndexManifest(
        document_id=run_name,
        source_uri=None,
        document_hash=_hash_values(
            [result.document_id for result in results]
            + [record.content_hash for record in records]
        ),
        chunk_hashes={record.chunk_id: record.content_hash for record in records},
        embedding_model=first.manifest.embedding_model,
        embedding_dimension=first.manifest.embedding_dimension,
        index_version=first.manifest.index_version,
        backend=first.manifest.backend,
        status=IndexingStatus.INDEXED,
        metadata={
            "combined": True,
            "document_count": len(results),
            "record_count": len(records),
            "source_files": [str(path) for path in files],
            "source_document_ids": [result.document_id for result in results],
        },
    )
    combined = IndexingResult(
        document_id=run_name,
        status=IndexingStatus.INDEXED,
        indexed_count=len(records),
        skipped_count=sum(result.skipped_count for result in results),
        deleted_count=sum(result.deleted_count for result in results),
        manifest=manifest,
        records=records,
    )
    output_path = output_dir / f"{run_name}.combined.index.json"
    output_path.write_text(combined.to_json(), encoding="utf-8")
    print(f"wrote {output_path}")
    return output_path


def _resolve_model_path(
    *,
    model_arg: str | None,
    cache_dir: Path | None,
    repo_id: str,
    offline: bool,
    flag_name: str,
) -> str:
    if model_arg:
        return model_arg
    if cache_dir is not None:
        repo_dir = cache_dir / ("models--" + repo_id.replace("/", "--"))
        refs_main = repo_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot = repo_dir / "snapshots" / revision
            if snapshot.exists():
                return str(snapshot)
        if offline:
            raise SystemExit(
                f"Offline model cache not found for {repo_id} under {cache_dir}. "
                f"Pass --{flag_name} or run once with --no-offline."
            )
    if offline:
        raise SystemExit(
            f"Offline mode needs a local path for {repo_id}. "
            f"Pass --{flag_name} or --hf-cache-dir."
        )
    return repo_id


def _build_env(
    *,
    hf_home: Path | None,
    hf_cache_dir: Path | None,
    docling_artifacts_path: Path | None,
    offline: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    if hf_cache_dir is not None:
        env["HF_HUB_CACHE"] = str(hf_cache_dir)
    if docling_artifacts_path is not None:
        env["DOCLING_ARTIFACTS_PATH"] = str(docling_artifacts_path)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        env.pop("HF_DATASETS_OFFLINE", None)
    return env


def _iter_source_files(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        candidates = [path]
    else:
        pattern = "**/*" if recursive else "*"
        candidates = [item for item in path.glob(pattern) if item.is_file()]
    return [
        item
        for item in sorted(candidates)
        if item.suffix.lower() in DEFAULT_SUPPORTED_EXTENSIONS
    ]


def _ensure_unique_stems(paths: list[Path]) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        existing = seen.get(path.stem)
        if existing is not None:
            raise SystemExit(
                "Input files must have unique file stems because stage outputs "
                f"use stem-based names: {existing} and {path}"
            )
        seen[path.stem] = path


def _require_files(label: str, files: list[Path]) -> None:
    missing = [path for path in files if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing {label} output file(s):\n{formatted}")


def _print_context_texts(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    contexts = data.get("contexts") or []
    print("\n== retrieved texts ==")
    for index, context in enumerate(contexts, start=1):
        text = str(context.get("text") or "").strip()
        print(f"\n--- context {index} ---")
        print(text)


def _hash_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def _first_existing_path(primary: Path | None, fallback: Path) -> Path | None:
    if primary is not None:
        return _project_path(primary)
    return fallback if fallback.exists() else None


def _project_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _safe_file_name(value: str) -> str:
    cleaned = "".join(char if char not in '<>:"/\\|?*' else "_" for char in value)
    return cleaned.strip(" .") or "offline_rag"


def _safe_collection_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value)
    cleaned = cleaned.strip("_") or "offline_rag"
    return f"offline_rag_{cleaned}"[:64]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local parsing, chunking, indexing, and retrieval in one command."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source file or folder.")
    parser.add_argument("--query", required=True, help="Retrieval query.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts") / "offline_rag",
        help="Base output folder for this one-command workflow.",
    )
    parser.add_argument("--run-name", help="Output subfolder name. Defaults to input stem.")
    parser.add_argument("--parsed-output", type=Path, help="Override parsed output folder.")
    parser.add_argument("--chunks-output", type=Path, help="Override chunks output folder.")
    parser.add_argument("--index-output", type=Path, help="Override index output folder.")
    parser.add_argument("--retrieval-output", type=Path, help="Override retrieval output folder.")
    parser.add_argument("--output", type=Path, help="Override retrieval JSON output path.")
    parser.add_argument("--no-recursive", action="store_true")

    parser.add_argument(
        "--pdf-backend",
        choices=sorted(SUPPORTED_PDF_BACKENDS),
        default="pypdfium2",
    )
    parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable Docling OCR. Default: disabled.",
    )
    parser.add_argument(
        "--artifacts-path",
        "--docling-artifacts-path",
        dest="docling_artifacts_path",
        type=Path,
        help="Local Docling artifacts path, for example D:\\AI\\docling.",
    )
    parser.add_argument(
        "--permissions",
        type=Path,
        help=(
            "Document access JSON sidecar. It maps filenames to tenant_id, "
            "allowed_user_ids, allowed_group_ids, and classification."
        ),
    )
    parser.add_argument("--document-tenant-id")
    parser.add_argument(
        "--document-allowed-user-id",
        dest="document_allowed_user_ids",
        action="append",
        default=[],
        help="Default allowed user id for parsed documents. Can be repeated.",
    )
    parser.add_argument(
        "--document-allowed-group-id",
        dest="document_allowed_group_ids",
        action="append",
        default=[],
        help="Default allowed group id for parsed documents. Can be repeated.",
    )
    parser.add_argument("--document-classification")

    parser.add_argument("--child-target-tokens", type=int, default=450)
    parser.add_argument("--child-max-tokens", type=int, default=650)
    parser.add_argument("--parent-target-tokens", type=int, default=1400)
    parser.add_argument("--parent-max-tokens", type=int, default=2000)

    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in IndexBackend] + ["qdrant-hybrid"],
        default=IndexBackend.QDRANT.value,
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "bge-m3"],
        default="bge-m3",
    )
    parser.add_argument("--embedding-dimension", type=int, default=384)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--bge-model", help="Local BGE-M3 snapshot path or model id.")
    parser.add_argument("--bge-model-id", default="BAAI/bge-m3")
    parser.add_argument("--bge-batch-size", type=int, default=12)
    parser.add_argument("--bge-max-length", type=int, default=8192)
    parser.add_argument("--sparse-top-n", type=int, default=512)
    parser.add_argument("--bge-use-fp16", action="store_true")

    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--qdrant-collection")
    parser.add_argument("--qdrant-timeout", type=int)
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="sparse")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-top-k", type=int, default=40)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--sparse-top-k", type=int, default=50)
    parser.add_argument("--fusion-top-k", type=int, default=60)
    parser.add_argument("--retriever-oversample", type=int, default=3)
    parser.add_argument(
        "--tenant-id",
        help="Trusted query tenant id used for retrieval-time permission filtering.",
    )
    parser.add_argument(
        "--user-id",
        help="Trusted query user id used for retrieval-time permission filtering.",
    )
    parser.add_argument(
        "--group-id",
        dest="group_ids",
        action="append",
        default=[],
        help="Trusted query group id. Can be repeated.",
    )
    parser.add_argument("--max-classification")
    parser.add_argument("--metadata-filter", action="append", default=[])
    parser.add_argument(
        "--reranker",
        choices=[provider.value for provider in RerankerProvider],
        default=RerankerProvider.BGE.value,
    )
    parser.add_argument("--reranker-model", help="Local reranker snapshot path or model id.")
    parser.add_argument("--reranker-model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--reranker-max-length", type=int, default=256)
    parser.add_argument("--reranker-device")
    parser.add_argument("--reranker-max-tokens-per-doc", type=int, default=4096)
    parser.add_argument("--reranker-use-fp16", action="store_true")
    parser.add_argument("--reranker-normalize", action="store_true")

    parser.add_argument("--hf-home", type=Path, help="Hugging Face home path.")
    parser.add_argument("--hf-cache-dir", type=Path, help="Hugging Face hub cache path.")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set Hugging Face offline env vars. Default: enabled.",
    )
    parser.add_argument(
        "--print-contexts",
        action="store_true",
        help="Print retrieved context text after writing the JSON output.",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Generate a cited LLM answer after retrieval.",
    )
    parser.add_argument("--answer-output", type=Path)
    parser.add_argument("--answer-markdown-output", type=Path)
    parser.add_argument("--answer-top-contexts", type=int, default=5)
    parser.add_argument("--answer-max-context-chars", type=int, default=12000)
    parser.add_argument("--answer-max-chars-per-context", type=int, default=3000)
    parser.add_argument("--answer-language", default="zh")
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument(
        "--llm-provider",
        choices=["openai-compatible", "mock"],
        default="openai-compatible",
    )
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="local-model")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--mock-response", default="这是一个 mock answer。[Context 1]")
    return parser.parse_args()


def _normalize_backend(value: str) -> str:
    return "qdrant_hybrid" if value == "qdrant-hybrid" else value


if __name__ == "__main__":
    raise SystemExit(main())
