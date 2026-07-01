from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.documents.schemas import DocumentJob, DocumentRecord


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_id: str | None = None
    user_id: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    max_classification: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatRequest(RetrievalRequest):
    llm_provider: str | None = Field(default=None)
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    include_prompt: bool | None = None


class CitationResponse(BaseModel):
    label: str
    text: str
    quote: str
    context_id: str
    source_chunk_id: str
    parent_chunk_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextResponse(BaseModel):
    context_id: str
    source_chunk_id: str
    parent_chunk_id: str | None = None
    chunk_type: str
    text: str
    contextual_text: str
    score: float
    source_scores: dict[str, float] = Field(default_factory=dict)
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateResponse(BaseModel):
    record_id: str
    chunk_id: str
    document_id: str
    parent_chunk_id: str | None = None
    chunk_type: str
    final_score: float
    fused_score: float
    rerank_score: float | None = None
    rank: int | None = None
    source_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalAPIResponse(BaseModel):
    query: str
    query_type: str
    contexts: list[ContextResponse]
    candidates: list[CandidateResponse]
    stats: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    request_id: str


class ChatResponse(BaseModel):
    answer: str
    content: str
    citations: list[CitationResponse]
    contexts: list[ContextResponse]
    stats: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    confidence: str
    provider: str
    model: str
    request_id: str
    prompt: str | None = None


class ModelListRequest(BaseModel):
    llm_provider: str = Field(default="ollama")
    base_url: str | None = None
    api_key: str | None = None
    timeout: float | None = Field(default=None, gt=0)


class ModelListResponse(BaseModel):
    llm_provider: str
    base_url: str
    models: list[str]
    request_id: str


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class PrincipalResponse(BaseModel):
    subject: str
    tenant_id: str | None = None
    user_id: str | None = None
    email: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    max_classification: str | None = None
    auth_mode: str
    enforce_permissions: bool
    can_chat: bool
    can_manage_documents: bool
    request_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    principal: PrincipalResponse
    request_id: str


class DocumentJobResponse(DocumentJob):
    request_id: str | None = None


class DocumentJobListResponse(BaseModel):
    jobs: list[DocumentJobResponse]
    request_id: str


class DocumentResponse(DocumentRecord):
    request_id: str | None = None
    job: DocumentJobResponse | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    request_id: str


class DocumentPermissionUpdate(BaseModel):
    tenant_id: str | None = None
    owner_id: str | None = None
    group_ids: list[str] = Field(default_factory=list)
    principal_ids: list[str] = Field(default_factory=list)
    classification: str | None = None


class HealthResponse(BaseModel):
    status: str
    runtime_loaded: bool
    index_path: str
    chunks_path: str
    backend: str
    llm_provider: str
    auth_mode: str
