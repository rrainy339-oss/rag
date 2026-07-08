from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.answering.providers import LLMProviderError
from app.api.errors import RuntimeConfigurationError, RuntimeUnavailableError
from app.api.runtime import RAGRuntime, list_models_for_request
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentJobListResponse,
    DocumentJobResponse,
    DocumentListResponse,
    DocumentPermissionUpdate,
    DocumentResponse,
    HealthResponse,
    LoginRequest,
    ModelListRequest,
    ModelListResponse,
    PrincipalResponse,
    RetrievalAPIResponse,
    RetrievalRequest,
    TokenResponse,
)
from app.api.settings import APISettings
from app.documents import DocumentJobType, DocumentService
from app.security import (
    AuditLogger,
    AuthService,
    AuthError,
    Principal,
    SecuritySettings,
    authenticate_request,
    require_scope,
)
from app.security.jwt import encode_jwt


settings = APISettings.from_env()
security_settings = SecuritySettings.from_env()
audit_logger = AuditLogger.from_settings(security_settings)
_runtime: RAGRuntime | None = None
_runtime_collection_signature: tuple[object, ...] | None = None
_document_service: DocumentService | None = None
_auth_service: AuthService | None = None


def close_runtime() -> None:
    global _runtime, _runtime_collection_signature
    if _runtime is not None:
        _runtime.close()
        _runtime = None
    _runtime_collection_signature = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_document_service().ensure_collection_files()
    try:
        yield
    finally:
        close_runtime()


def get_runtime() -> RAGRuntime:
    global _runtime, _runtime_collection_signature
    try:
        service = get_document_service()
        service.ensure_collection_files()
        current_signature = service.collection_manifest_signature()
        if (
            _runtime is not None
            and _runtime_collection_signature != current_signature
        ):
            close_runtime()
        if _runtime is None:
            _runtime = RAGRuntime(settings)
            _runtime_collection_signature = service.collection_manifest_signature()
        return _runtime
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService(
            settings,
            on_collection_updated=close_runtime,
        )
    return _document_service


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(security_settings)
    return _auth_service


def get_current_principal(request: Request) -> Principal:
    try:
        return authenticate_request(request, security_settings)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


app = FastAPI(
    title="Enterprise RAG API",
    version="0.1.0",
    description="Thin FastAPI layer over the local enterprise RAG runtime.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        runtime_loaded=_runtime is not None,
        index_path=str(settings.index_path),
        chunks_path=str(settings.chunks_path),
        backend=settings.backend,
        llm_provider=settings.llm_provider,
        auth_mode=security_settings.auth_mode,
    )


@app.post("/api/auth/admin/login", response_model=TokenResponse)
def admin_login(
    request_body: LoginRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    try:
        principal = _principal_for_login(
            request_body,
            login_type="admin",
            auth_service=auth_service,
        )
        return _issue_token(principal, request_id=request.state.request_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/auth/chat/login", response_model=TokenResponse)
def chat_login(
    request_body: LoginRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    try:
        principal = _principal_for_login(
            request_body,
            login_type="chat",
            auth_service=auth_service,
        )
        return _issue_token(principal, request_id=request.state.request_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/me", response_model=PrincipalResponse)
def me(
    request: Request,
    principal: Principal = Depends(get_current_principal),
) -> PrincipalResponse:
    return _to_principal_response(principal, request_id=request.state.request_id)


@app.post("/api/retrieve", response_model=RetrievalAPIResponse)
def retrieve(
    request_body: RetrievalRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    runtime: RAGRuntime = Depends(get_runtime),
) -> RetrievalAPIResponse:
    try:
        require_scope(principal, "rag:retrieve")
        response = runtime.retrieve(
            request_body,
            request_id=request.state.request_id,
            principal=principal,
        )
        audit_logger.log(
            event_type="retrieve",
            request_id=request.state.request_id,
            principal=principal,
            query=request_body.query,
            metadata={
                "context_count": len(response.contexts),
                "candidate_count": len(response.candidates),
            },
        )
        return response
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request_body: ChatRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    runtime: RAGRuntime = Depends(get_runtime),
) -> ChatResponse:
    try:
        require_scope(principal, "rag:chat")
        response = runtime.chat(
            request_body,
            request_id=request.state.request_id,
            principal=principal,
        )
        audit_logger.log(
            event_type="chat",
            request_id=request.state.request_id,
            principal=principal,
            query=request_body.query,
            metadata={
                "context_count": len(response.contexts),
                "provider": response.provider,
                "model": response.model,
                "confidence": response.confidence,
            },
        )
        return response
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/models", response_model=ModelListResponse)
def models(
    request_body: ModelListRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
) -> ModelListResponse:
    try:
        require_scope(principal, "rag:models")
        response = list_models_for_request(
            request_body,
            settings=settings,
            request_id=request.state.request_id,
        )
        audit_logger.log(
            event_type="models",
            request_id=request.state.request_id,
            principal=principal,
            metadata={
                "provider": response.llm_provider,
                "base_url": response.base_url,
                "model_count": len(response.models),
            },
        )
        return response
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    request: Request,
    include_deleted: bool = False,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    try:
        require_scope(principal, "rag:documents")
        documents = [
            _to_document_response(record, job=service.latest_job(record.document_id))
            for record in service.list_documents(include_deleted=include_deleted)
        ]
        return DocumentListResponse(
            documents=documents,
            request_id=request.state.request_id,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    try:
        require_scope(principal, "rag:documents")
        record = service.get_document(document_id)
        if record is None:
            raise KeyError(document_id)
        return _to_document_response(
            record,
            request_id=request.state.request_id,
            job=service.latest_job(record.document_id),
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/documents", response_model=DocumentResponse)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    tenant_id: str | None = Form(default=None),
    owner_id: str | None = Form(default=None),
    group_ids: list[str] | None = Form(default=None),
    principal_ids: list[str] | None = Form(default=None),
    classification: str | None = Form(default=None),
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    try:
        require_scope(principal, "rag:documents")
        trusted_tenant_id = tenant_id or (
            principal.tenant_id if principal.enforce_permissions else None
        )
        trusted_owner_id = owner_id or (
            principal.user_id or principal.subject
            if principal.enforce_permissions
            else None
        )
        create_result = service.create_document_result(
            filename=file.filename or "document",
            content_type=file.content_type,
            file=file.file,
            title=title,
            tenant_id=trusted_tenant_id,
            owner_id=trusted_owner_id,
            group_ids=_split_form_values(group_ids),
            principal_ids=_split_form_values(principal_ids),
            classification=classification,
        )
        record = create_result.record
        job = (
            service.enqueue_document(record.document_id)
            if create_result.created
            else service.latest_job(record.document_id)
        )
        audit_logger.log(
            event_type="document_upload",
            request_id=request.state.request_id,
            principal=principal,
            metadata={
                "document_id": record.document_id,
                "filename": record.filename,
                "created": create_result.created,
                "tenant_id": record.tenant_id,
                "group_ids": record.group_ids,
                "classification": record.classification,
                "job_id": job.job_id if job is not None else None,
            },
        )
        return _to_document_response(record, request_id=request.state.request_id, job=job)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.patch("/api/documents/{document_id}/permissions", response_model=DocumentResponse)
def update_document_permissions(
    document_id: str,
    request_body: DocumentPermissionUpdate,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    try:
        require_scope(principal, "rag:documents")
        record = service.update_permissions(
            document_id,
            tenant_id=request_body.tenant_id,
            owner_id=request_body.owner_id,
            group_ids=request_body.group_ids,
            principal_ids=request_body.principal_ids,
            classification=request_body.classification,
        )
        audit_logger.log(
            event_type="document_permissions_update",
            request_id=request.state.request_id,
            principal=principal,
            metadata={
                "document_id": record.document_id,
                "tenant_id": record.tenant_id,
                "group_ids": record.group_ids,
                "principal_ids": record.principal_ids,
                "classification": record.classification,
            },
        )
        return _to_document_response(
            record,
            request_id=request.state.request_id,
            job=service.latest_job(record.document_id),
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/documents/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    try:
        require_scope(principal, "rag:documents")
        record = service.get_document(document_id)
        if record is None:
            raise KeyError(document_id)
        job = service.enqueue_document(document_id, job_type=DocumentJobType.REINDEX)
        audit_logger.log(
            event_type="document_reindex",
            request_id=request.state.request_id,
            principal=principal,
            metadata={"document_id": document_id, "job_id": job.job_id},
        )
        return _to_document_response(record, request_id=request.state.request_id, job=job)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/jobs", response_model=DocumentJobListResponse)
def list_jobs(
    request: Request,
    document_id: str | None = None,
    include_terminal: bool = True,
    limit: int = 100,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentJobListResponse:
    try:
        require_scope(principal, "rag:documents")
        jobs = service.list_jobs(
            document_id=document_id,
            include_terminal=include_terminal,
            limit=limit,
        )
        return DocumentJobListResponse(
            jobs=[_to_job_response(job) for job in jobs],
            request_id=request.state.request_id,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/jobs/{job_id}", response_model=DocumentJobResponse)
def get_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentJobResponse:
    try:
        require_scope(principal, "rag:documents")
        job = service.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return _to_job_response(job, request_id=request.state.request_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/jobs/{job_id}/cancel", response_model=DocumentJobResponse)
def cancel_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentJobResponse:
    try:
        require_scope(principal, "rag:documents")
        job = service.cancel_job(job_id)
        audit_logger.log(
            event_type="document_job_cancel",
            request_id=request.state.request_id,
            principal=principal,
            metadata={"job_id": job_id, "document_id": job.document_id},
        )
        return _to_job_response(job, request_id=request.state.request_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/jobs/{job_id}/retry", response_model=DocumentJobResponse)
def retry_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentJobResponse:
    try:
        require_scope(principal, "rag:documents")
        job = service.retry_job(job_id)
        audit_logger.log(
            event_type="document_job_retry",
            request_id=request.state.request_id,
            principal=principal,
            metadata={"job_id": job_id, "document_id": job.document_id},
        )
        return _to_job_response(job, request_id=request.state.request_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/jobs/run-next", response_model=DocumentJobResponse | None)
def run_next_job(
    request: Request,
    worker_id: str = "api-worker",
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentJobResponse | None:
    try:
        require_scope(principal, "rag:documents")
        job = service.run_next_job(worker_id=worker_id)
        return (
            _to_job_response(job, request_id=request.state.request_id)
            if job is not None
            else None
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.delete("/api/documents/{document_id}", response_model=DocumentResponse)
def delete_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    try:
        require_scope(principal, "rag:documents")
        record = service.delete_document(document_id)
        audit_logger.log(
            event_type="document_delete",
            request_id=request.state.request_id,
            principal=principal,
            metadata={"document_id": document_id},
        )
        return _to_document_response(
            record,
            request_id=request.state.request_id,
            job=service.latest_job(record.document_id),
        )
    except Exception as exc:
        raise _http_error(exc) from exc


def reset_runtime_for_tests() -> None:
    global _auth_service, _document_service
    close_runtime()
    _document_service = None
    _auth_service = None


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, RuntimeUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AuthError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, LLMProviderError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Resource not found.")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _has_scope(principal: Principal, scope: str) -> bool:
    return (
        not principal.enforce_permissions
        or scope in principal.scopes
        or "rag:admin" in principal.scopes
    )


def _principal_for_login(
    request_body: LoginRequest,
    *,
    login_type: str,
    auth_service: AuthService,
) -> Principal:
    return auth_service.authenticate_login(
        request_body.username,
        request_body.password,
        login_type=login_type,
    )


def _issue_token(principal: Principal, *, request_id: str) -> TokenResponse:
    expires_in = security_settings.jwt_expire_minutes * 60
    access_token = encode_jwt(
        {
            "sub": principal.subject,
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "email": principal.email,
            "group_ids": principal.group_ids,
            "roles": principal.roles,
            "scopes": principal.scopes,
            "max_classification": principal.max_classification,
        },
        secret=security_settings.jwt_secret,
        issuer=security_settings.jwt_issuer,
        audience=security_settings.jwt_audience,
        expires_in_seconds=expires_in,
    )
    return TokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        principal=_to_principal_response(principal, request_id=request_id),
        request_id=request_id,
    )


def _to_principal_response(
    principal: Principal,
    *,
    request_id: str,
) -> PrincipalResponse:
    return PrincipalResponse(
        subject=principal.subject,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        email=principal.email,
        group_ids=principal.group_ids,
        roles=principal.roles,
        scopes=principal.scopes,
        max_classification=principal.max_classification,
        auth_mode=principal.auth_mode,
        enforce_permissions=principal.enforce_permissions,
        can_chat=_has_scope(principal, "rag:chat"),
        can_manage_documents=_has_scope(principal, "rag:documents"),
        request_id=request_id,
    )


def _to_document_response(
    record: object,
    *,
    request_id: str | None = None,
    job: object | None = None,
) -> DocumentResponse:
    data = record.model_dump() if hasattr(record, "model_dump") else record
    response = DocumentResponse.model_validate(data)
    response.request_id = request_id
    response.job = _to_job_response(job) if job is not None else None
    return response


def _to_job_response(
    job: object,
    *,
    request_id: str | None = None,
) -> DocumentJobResponse:
    data = job.model_dump() if hasattr(job, "model_dump") else job
    response = DocumentJobResponse.model_validate(data)
    response.request_id = request_id
    return response


def _split_form_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.replace(";", ",").split(","))
    return [value for value in result if value]
