from __future__ import annotations

import os
from pathlib import Path

from app.answering import (
    AnswerPipeline,
    ContextPacker,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    PromptBuilder,
    list_ollama_models,
    list_openai_compatible_models,
)
from app.answering.providers import LLMProvider
from app.api.errors import RuntimeConfigurationError
from app.api.schemas import (
    CandidateResponse,
    ChatRequest,
    ChatResponse,
    CitationResponse,
    ContextResponse,
    ModelListRequest,
    ModelListResponse,
    RetrievalAPIResponse,
    RetrievalRequest,
)
from app.api.settings import APISettings
from app.documents.chunks import ChunkRepository
from app.documents.manifest import (
    CollectionManifestRepository,
    build_collection_manifest,
)
from app.indexing.embeddings.bge import BGEM3EmbeddingProvider
from app.indexing.schemas import IndexManifest, IndexingResult
from app.indexing.vectorstores.qdrant import QdrantHybridVectorStore
from app.retrieval.config import RetrievalConfig
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.parent_expander import ParentExpander
from app.retrieval.qdrant_hybrid import (
    QdrantHybridRetrievalPipeline,
    QdrantHybridRetriever,
)
from app.retrieval.rerankers import RerankerConfig, RerankerProvider, build_reranker
from app.retrieval.schemas import RetrievalQuery, RetrievalResponse
from app.security.schemas import Principal


class RAGRuntime:
    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.manifest_repository = CollectionManifestRepository(
            settings.documents_db_path
        )
        self.collection_manifest = _load_collection_manifest(
            settings,
            self.manifest_repository,
        )
        self.embedding_provider = self._build_embedding_provider()
        self.vector_store = self._build_vector_store()
        self.chunk_store = ChunkRepository(settings.documents_db_path)
        self.reranker = self._build_reranker()

    def retrieve(
        self,
        request: RetrievalRequest,
        *,
        request_id: str,
        principal: Principal | None = None,
    ) -> RetrievalAPIResponse:
        response = self._retrieve_response(request, principal=principal)
        return _to_retrieval_api_response(response, request_id=request_id)

    def chat(
        self,
        request: ChatRequest,
        *,
        request_id: str,
        principal: Principal | None = None,
    ) -> ChatResponse:
        retrieval_response = self._retrieve_response(request, principal=principal)
        answer_pipeline = self._build_answer_pipeline(request)
        answer = answer_pipeline.answer(retrieval_response, query=request.query)
        contexts = [_to_context_response(context) for context in retrieval_response.contexts]
        citations = [
            CitationResponse(
                label=citation.label,
                text=citation.quote,
                quote=citation.quote,
                context_id=citation.context_id,
                source_chunk_id=citation.source_chunk_id,
                parent_chunk_id=citation.parent_chunk_id,
                section_path=citation.section_path,
                score=citation.score,
                metadata=citation.metadata,
            )
            for citation in answer.citations
        ]
        stats = {
            **retrieval_response.stats,
            "confidence": answer.confidence.value,
            "provider": answer.provider,
            "model": answer.model,
        }
        return ChatResponse(
            answer=answer.answer,
            content=answer.answer,
            citations=citations,
            contexts=contexts,
            stats=stats,
            warnings=answer.warnings,
            confidence=answer.confidence.value,
            provider=answer.provider,
            model=answer.model,
            request_id=request_id,
            prompt=answer.prompt,
        )

    def close(self) -> None:
        for resource in (self.vector_store, self.embedding_provider):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def _retrieve_response(
        self,
        request: RetrievalRequest,
        *,
        principal: Principal | None = None,
    ) -> RetrievalResponse:
        pipeline = self._build_retrieval_pipeline(request)
        return pipeline.retrieve(_retrieval_query(request, principal))

    def _build_retrieval_pipeline(self, request: RetrievalRequest) -> object:
        config = RetrievalConfig(
            dense_top_k=self.settings.dense_top_k,
            sparse_top_k=self.settings.sparse_top_k,
            fusion_top_k=self.settings.fusion_top_k,
            rerank_top_k=self.settings.rerank_top_k,
            final_top_k=request.top_k or self.settings.final_top_k,
            retriever_oversample=self.settings.retriever_oversample,
            release_dense_model_before_rerank=(
                self.settings.release_dense_model_before_rerank
            ),
        )
        context_builder = ContextBuilder(ParentExpander(self.chunk_store))
        return QdrantHybridRetrievalPipeline(
            hybrid_retriever=QdrantHybridRetriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
            ),
            config=config,
            reranker=self.reranker,
            context_builder=context_builder,
        )

    def _build_answer_pipeline(self, request: ChatRequest) -> AnswerPipeline:
        include_prompt = (
            self.settings.include_prompt
            if request.include_prompt is None
            else request.include_prompt
        )
        return AnswerPipeline(
            llm_provider=self._build_llm_provider(request),
            context_packer=ContextPacker(
                top_contexts=self.settings.answer_top_contexts,
                max_context_chars=self.settings.answer_max_context_chars,
                max_chars_per_context=self.settings.answer_max_chars_per_context,
            ),
            prompt_builder=PromptBuilder(language=self.settings.answer_language),
            include_prompt=include_prompt,
        )

    def _build_embedding_provider(self) -> object:
        if "bge-m3" not in self.collection_manifest.embedding_model.lower():
            raise RuntimeConfigurationError(
                "Runtime requires a BGE-M3 hybrid index manifest, got "
                f"{self.collection_manifest.embedding_model!r}."
            )
        return BGEM3EmbeddingProvider(
            model_name=(
                self.settings.bge_model
                or self.collection_manifest.embedding_model
            ),
            use_fp16=self.settings.bge_use_fp16,
            max_length=self.settings.bge_max_length,
            batch_size=self.settings.bge_batch_size,
            cache_dir=(
                str(self.settings.bge_cache_dir)
                if self.settings.bge_cache_dir
                else None
            ),
            sparse_top_n=self.settings.sparse_top_n,
        )

    def _build_vector_store(self) -> object:
        qdrant_path = self.settings.qdrant_path
        if self.settings.qdrant_url is None and qdrant_path is None:
            qdrant_path = self.settings.index_path.parent / "qdrant_hybrid"
        return QdrantHybridVectorStore(
            url=self.settings.qdrant_url,
            path=qdrant_path,
            api_key=self.settings.qdrant_api_key,
            collection_name=self.settings.qdrant_collection,
            vector_size=self.collection_manifest.embedding_dimension,
            dense_vector_name=self.settings.dense_vector_name,
            sparse_vector_name=self.settings.sparse_vector_name,
            timeout=self.settings.qdrant_timeout,
        )

    def _build_reranker(self) -> object:
        return build_reranker(
            RerankerConfig(
                provider=RerankerProvider(self.settings.reranker),
                model_name=self.settings.reranker_model,
                batch_size=self.settings.reranker_batch_size,
                max_length=self.settings.reranker_max_length,
                use_fp16=self.settings.reranker_use_fp16,
                normalize=self.settings.reranker_normalize,
                cache_dir=(
                    str(self.settings.reranker_cache_dir)
                    if self.settings.reranker_cache_dir
                    else None
                ),
                device=self.settings.reranker_device,
                api_key=self.settings.reranker_api_key,
                timeout_seconds=self.settings.reranker_timeout,
                max_tokens_per_doc=self.settings.reranker_max_tokens_per_doc,
                instruction=self.settings.reranker_instruction,
            )
        )

    def _build_llm_provider(self, request: ChatRequest) -> LLMProvider:
        provider = request.llm_provider or self.settings.llm_provider
        if provider == "mock":
            return MockLLMProvider(
                "Development mock answer based on retrieved context. [Context 1]",
                model_name=request.model or "mock-llm",
            )
        if provider == "ollama":
            return OllamaProvider(
                base_url=request.base_url or self.settings.ollama_base_url,
                model_name=request.model or self.settings.ollama_model,
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else self.settings.llm_temperature
                ),
                top_p=request.top_p if request.top_p is not None else self.settings.llm_top_p,
                timeout=self.settings.llm_timeout,
            )
        if provider == "openai-compatible":
            api_key = request.api_key or self.settings.llm_api_key
            if api_key is None and self.settings.llm_api_key_env:
                api_key = os.environ.get(self.settings.llm_api_key_env)
            return OpenAICompatibleProvider(
                base_url=request.base_url or self.settings.llm_base_url,
                model_name=request.model or self.settings.llm_model,
                api_key=api_key,
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else self.settings.llm_temperature
                ),
                top_p=request.top_p if request.top_p is not None else self.settings.llm_top_p,
                max_tokens=(
                    request.max_tokens
                    if request.max_tokens is not None
                    else self.settings.llm_max_tokens
                ),
                timeout=self.settings.llm_timeout,
            )
        raise RuntimeConfigurationError(f"Unsupported LLM provider: {provider}")


def list_models_for_request(
    request: ModelListRequest,
    *,
    settings: APISettings,
    request_id: str,
) -> ModelListResponse:
    provider = request.llm_provider
    timeout = request.timeout or 30.0
    if provider == "mock":
        return ModelListResponse(
            llm_provider=provider,
            base_url="",
            models=["mock-llm"],
            request_id=request_id,
        )
    if provider == "ollama":
        base_url = request.base_url or settings.ollama_base_url
        return ModelListResponse(
            llm_provider=provider,
            base_url=base_url,
            models=list_ollama_models(base_url=base_url, timeout=timeout),
            request_id=request_id,
        )
    if provider == "openai-compatible":
        base_url = request.base_url or settings.llm_base_url
        api_key = request.api_key or settings.llm_api_key
        if api_key is None and settings.llm_api_key_env:
            api_key = os.environ.get(settings.llm_api_key_env)
        return ModelListResponse(
            llm_provider=provider,
            base_url=base_url,
            models=list_openai_compatible_models(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            ),
            request_id=request_id,
        )
    raise RuntimeConfigurationError(f"Unsupported LLM provider: {provider}")


def _retrieval_query(
    request: RetrievalRequest,
    principal: Principal | None,
) -> RetrievalQuery:
    if principal is not None and principal.enforce_permissions:
        return RetrievalQuery(
            query=request.query,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            group_ids=principal.group_ids,
            max_classification=principal.max_classification,
            metadata_filters=request.metadata_filters,
        )
    return RetrievalQuery(
        query=request.query,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        group_ids=request.group_ids,
        max_classification=request.max_classification,
        metadata_filters=request.metadata_filters,
    )


def _load_collection_manifest(
    settings: APISettings,
    repository: CollectionManifestRepository,
) -> IndexManifest:
    manifest = repository.get(settings.qdrant_collection)
    if manifest is not None:
        return manifest
    if settings.index_path.exists():
        manifest = _load_indexing_result(settings.index_path).manifest
        repository.save(settings.qdrant_collection, manifest)
        return manifest
    manifest = build_collection_manifest(
        embedding_model=settings.bge_model or "BAAI/bge-m3",
        embedding_dimension=BGEM3EmbeddingProvider.dimension,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        sparse_top_n=settings.sparse_top_n,
        index_version=settings.documents_index_version,
    )
    repository.save(settings.qdrant_collection, manifest)
    return manifest


def _load_indexing_result(path: Path) -> IndexingResult:
    if not path.exists():
        raise RuntimeConfigurationError(f"Index file does not exist: {path}")
    return IndexingResult.model_validate_json(path.read_text(encoding="utf-8"))


def _to_retrieval_api_response(
    response: RetrievalResponse,
    *,
    request_id: str,
) -> RetrievalAPIResponse:
    return RetrievalAPIResponse(
        query=response.query,
        query_type=response.query_type.value,
        contexts=[_to_context_response(context) for context in response.contexts],
        candidates=[
            CandidateResponse(
                record_id=candidate.record.record_id,
                chunk_id=candidate.record.chunk_id,
                document_id=candidate.record.document_id,
                parent_chunk_id=candidate.record.parent_chunk_id,
                chunk_type=candidate.record.chunk_type.value,
                final_score=candidate.final_score,
                fused_score=candidate.fused_score,
                rerank_score=candidate.rerank_score,
                rank=candidate.rank,
                source_scores=candidate.source_scores,
                metadata=candidate.metadata,
            )
            for candidate in response.candidates
        ],
        stats=response.stats,
        request_id=request_id,
    )


def _to_context_response(context: object) -> ContextResponse:
    return ContextResponse(
        context_id=context.context_id,
        source_chunk_id=context.source_chunk_id,
        parent_chunk_id=context.parent_chunk_id,
        chunk_type=context.chunk_type.value,
        text=context.text,
        contextual_text=context.contextual_text,
        score=context.score,
        source_scores=context.source_scores,
        section_path=context.section_path,
        metadata=context.metadata,
    )
