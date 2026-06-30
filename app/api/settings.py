from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


BackendName = Literal["memory", "qdrant", "qdrant_hybrid"]
EmbeddingProviderName = Literal["auto", "hashing", "bge-m3"]
LLMProviderName = Literal["mock", "ollama", "openai-compatible"]
RerankerName = Literal["noop", "bge", "qwen", "cohere", "voyage"]


class APISettings(BaseModel):
    index_path: Path = Path("artifacts/collections/default/index.json")
    chunks_path: Path = Path("artifacts/collections/default/chunks.json")

    documents_db_path: Path = Path("artifacts/documents/documents.sqlite3")
    documents_storage_dir: Path = Path("artifacts/documents")
    documents_collection_index_path: Path = Path("artifacts/collections/default/index.json")
    documents_collection_chunks_path: Path = Path("artifacts/collections/default/chunks.json")
    documents_pdf_backend: str | None = "pypdfium2"
    documents_do_ocr: bool | None = False
    documents_docling_artifacts_path: Path | None = None
    documents_child_target_tokens: int = Field(default=450, ge=80)
    documents_child_max_tokens: int = Field(default=650, ge=100)
    documents_parent_target_tokens: int = Field(default=1400, ge=200)
    documents_parent_max_tokens: int = Field(default=2000, ge=300)
    documents_backend: BackendName = "qdrant_hybrid"
    documents_embedding_provider: Literal["hashing", "bge-m3"] = "bge-m3"
    ingestion_embedding_dimension: int = Field(default=384, ge=8)
    documents_embedding_batch_size: int = Field(default=32, ge=1)

    backend: BackendName = "memory"
    embedding_provider: EmbeddingProviderName = "auto"
    bge_model: str | None = None
    bge_cache_dir: Path | None = None
    bge_batch_size: int = Field(default=12, ge=1)
    bge_max_length: int = Field(default=8192, ge=1)
    bge_use_fp16: bool = False
    sparse_top_n: int | None = Field(default=512, ge=1)

    qdrant_url: str | None = None
    qdrant_path: Path | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_chunks"
    qdrant_timeout: int | None = None
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"

    reranker: RerankerName = "noop"
    reranker_model: str | None = None
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_max_length: int | None = Field(default=None, ge=1)
    reranker_cache_dir: Path | None = None
    reranker_device: str | None = None
    reranker_api_key: str | None = None
    reranker_timeout: float = Field(default=60.0, gt=0)
    reranker_max_tokens_per_doc: int = Field(default=4096, ge=1)
    reranker_instruction: str | None = None
    reranker_use_fp16: bool = False
    reranker_normalize: bool = False

    dense_top_k: int = Field(default=50, ge=1)
    sparse_top_k: int = Field(default=50, ge=1)
    fusion_top_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=40, ge=1)
    final_top_k: int = Field(default=5, ge=1)
    retriever_oversample: int = Field(default=3, ge=1)
    release_dense_model_before_rerank: bool = False

    llm_provider: LLMProviderName = "mock"
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "local-model"
    llm_api_key: str | None = None
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_top_p: float = Field(default=1.0, ge=0, le=1)
    llm_max_tokens: int = Field(default=1024, ge=1)
    llm_timeout: float = Field(default=120.0, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    answer_top_contexts: int = Field(default=5, ge=1)
    answer_max_context_chars: int = Field(default=12000, ge=1)
    answer_max_chars_per_context: int = Field(default=3000, ge=1)
    answer_language: str = "zh"
    include_prompt: bool = False

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "APISettings":
        return cls(
            index_path=_path("RAG_API_INDEX_PATH", cls.model_fields["index_path"].default),
            chunks_path=_path("RAG_API_CHUNKS_PATH", cls.model_fields["chunks_path"].default),
            documents_db_path=_path(
                "RAG_DOCUMENTS_DB_PATH",
                cls.model_fields["documents_db_path"].default,
            ),
            documents_storage_dir=_path(
                "RAG_DOCUMENTS_STORAGE_DIR",
                cls.model_fields["documents_storage_dir"].default,
            ),
            documents_collection_index_path=_path(
                "RAG_DOCUMENTS_COLLECTION_INDEX_PATH",
                cls.model_fields["documents_collection_index_path"].default,
            ),
            documents_collection_chunks_path=_path(
                "RAG_DOCUMENTS_COLLECTION_CHUNKS_PATH",
                cls.model_fields["documents_collection_chunks_path"].default,
            ),
            documents_pdf_backend=_optional_str("RAG_DOCUMENTS_PDF_BACKEND") or "pypdfium2",
            documents_do_ocr=_optional_bool("RAG_DOCUMENTS_DO_OCR", False),
            documents_docling_artifacts_path=_optional_path(
                "RAG_DOCUMENTS_DOCLING_ARTIFACTS_PATH"
            ),
            documents_child_target_tokens=_int("RAG_DOCUMENTS_CHILD_TARGET_TOKENS", 450),
            documents_child_max_tokens=_int("RAG_DOCUMENTS_CHILD_MAX_TOKENS", 650),
            documents_parent_target_tokens=_int("RAG_DOCUMENTS_PARENT_TARGET_TOKENS", 1400),
            documents_parent_max_tokens=_int("RAG_DOCUMENTS_PARENT_MAX_TOKENS", 2000),
            documents_backend=_str("RAG_DOCUMENTS_BACKEND", "qdrant_hybrid"),
            documents_embedding_provider=_str(
                "RAG_DOCUMENTS_EMBEDDING_PROVIDER",
                "bge-m3",
            ),
            ingestion_embedding_dimension=_int("RAG_DOCUMENTS_EMBEDDING_DIMENSION", 384),
            documents_embedding_batch_size=_int("RAG_DOCUMENTS_EMBEDDING_BATCH_SIZE", 32),
            backend=_str("RAG_API_BACKEND", "memory"),
            embedding_provider=_str("RAG_API_EMBEDDING_PROVIDER", "auto"),
            bge_model=_optional_str("RAG_API_BGE_MODEL"),
            bge_cache_dir=_optional_path("RAG_API_BGE_CACHE_DIR"),
            bge_batch_size=_int("RAG_API_BGE_BATCH_SIZE", 12),
            bge_max_length=_int("RAG_API_BGE_MAX_LENGTH", 8192),
            bge_use_fp16=_bool("RAG_API_BGE_USE_FP16", False),
            sparse_top_n=_optional_int("RAG_API_SPARSE_TOP_N", 512),
            qdrant_url=_optional_str("RAG_API_QDRANT_URL"),
            qdrant_path=_optional_path("RAG_API_QDRANT_PATH"),
            qdrant_api_key=_optional_str("RAG_API_QDRANT_API_KEY"),
            qdrant_collection=_str("RAG_API_QDRANT_COLLECTION", "rag_chunks"),
            qdrant_timeout=_optional_int("RAG_API_QDRANT_TIMEOUT"),
            dense_vector_name=_str("RAG_API_DENSE_VECTOR_NAME", "dense"),
            sparse_vector_name=_str("RAG_API_SPARSE_VECTOR_NAME", "sparse"),
            reranker=_str("RAG_API_RERANKER", "noop"),
            reranker_model=_optional_str("RAG_API_RERANKER_MODEL"),
            reranker_batch_size=_int("RAG_API_RERANKER_BATCH_SIZE", 16),
            reranker_max_length=_optional_int("RAG_API_RERANKER_MAX_LENGTH"),
            reranker_cache_dir=_optional_path("RAG_API_RERANKER_CACHE_DIR"),
            reranker_device=_optional_str("RAG_API_RERANKER_DEVICE"),
            reranker_api_key=_optional_str("RAG_API_RERANKER_API_KEY"),
            reranker_timeout=_float("RAG_API_RERANKER_TIMEOUT", 60.0),
            reranker_max_tokens_per_doc=_int("RAG_API_RERANKER_MAX_TOKENS_PER_DOC", 4096),
            reranker_instruction=_optional_str("RAG_API_RERANKER_INSTRUCTION"),
            reranker_use_fp16=_bool("RAG_API_RERANKER_USE_FP16", False),
            reranker_normalize=_bool("RAG_API_RERANKER_NORMALIZE", False),
            dense_top_k=_int("RAG_API_DENSE_TOP_K", 50),
            sparse_top_k=_int("RAG_API_SPARSE_TOP_K", 50),
            fusion_top_k=_int("RAG_API_FUSION_TOP_K", 60),
            rerank_top_k=_int("RAG_API_RERANK_TOP_K", 40),
            final_top_k=_int("RAG_API_FINAL_TOP_K", 5),
            retriever_oversample=_int("RAG_API_RETRIEVER_OVERSAMPLE", 3),
            release_dense_model_before_rerank=_bool(
                "RAG_API_RELEASE_DENSE_MODEL_BEFORE_RERANK", False
            ),
            llm_provider=_str("RAG_API_LLM_PROVIDER", "mock"),
            llm_base_url=_str("RAG_API_LLM_BASE_URL", "http://localhost:8000/v1"),
            llm_model=_str("RAG_API_LLM_MODEL", "local-model"),
            llm_api_key=_optional_str("RAG_API_LLM_API_KEY"),
            llm_api_key_env=_str("RAG_API_LLM_API_KEY_ENV", "OPENAI_API_KEY"),
            llm_temperature=_float("RAG_API_LLM_TEMPERATURE", 0.1),
            llm_top_p=_float("RAG_API_LLM_TOP_P", 1.0),
            llm_max_tokens=_int("RAG_API_LLM_MAX_TOKENS", 1024),
            llm_timeout=_float("RAG_API_LLM_TIMEOUT", 120.0),
            ollama_base_url=_str("RAG_API_OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_str("RAG_API_OLLAMA_MODEL", "llama3.1"),
            answer_top_contexts=_int("RAG_API_ANSWER_TOP_CONTEXTS", 5),
            answer_max_context_chars=_int("RAG_API_ANSWER_MAX_CONTEXT_CHARS", 12000),
            answer_max_chars_per_context=_int(
                "RAG_API_ANSWER_MAX_CHARS_PER_CONTEXT", 3000
            ),
            answer_language=_str("RAG_API_ANSWER_LANGUAGE", "zh"),
            include_prompt=_bool("RAG_API_INCLUDE_PROMPT", False),
            cors_origins=_csv("RAG_API_CORS_ORIGINS", ["*"]),
        )


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _optional_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: object) -> Path:
    value = os.getenv(name)
    return Path(value) if value else Path(default)


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _csv(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]
