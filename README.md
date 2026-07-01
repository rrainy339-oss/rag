# Enterprise RAG

This project now keeps one production-oriented ingestion and retrieval path:

```text
frontend upload -> FastAPI document service -> Docling parse -> hierarchical chunks
chunks -> BGE-M3 dense+sparse embeddings -> Qdrant hybrid collection
query -> BGE-M3 dense+sparse query embedding -> Qdrant hybrid retrieval -> rerank -> answer
```

Legacy local demo implementations and old offline indexing/retrieval scripts
have been removed from the test branch.

## Current Modules

- `frontend/`: browser UI for chat settings and document management.
- `app/api/`: FastAPI endpoints for chat, retrieval, model listing, and documents.
- `app/documents/`: upload registry, permissions, parse/chunk/index orchestration.
- `app/ingestion/`: local file parsing pipeline.
- `app/parsing/`: Docling adapter and normalized parsed document model.
- `app/chunking/`: parent/child/table chunk generation.
- `app/indexing/embeddings/bge.py`: BGE-M3 dense+sparse embedding adapter.
- `app/indexing/indexer.py`: converts chunks into Qdrant hybrid index records.
- `app/indexing/vectorstores/qdrant.py`: Qdrant dense+sparse hybrid vector store.
- `app/retrieval/qdrant_hybrid.py`: Qdrant hybrid retrieval pipeline.
- `app/retrieval/rerankers/`: optional BGE, Qwen, Cohere, and Voyage rerankers.
- `app/answering/`: context packing, prompts, and LLM provider adapters.
- `app/security/`: dev/JWT/OIDC auth, permission context, and audit logging.
- `scripts/document_worker.py`: durable document job worker.
- `scripts/parse_local.py`: optional parser debugging utility.
- `scripts/chunk_parsed.py`: optional chunking debugging utility.
- `scripts/answer.py`: optional answer-layer debugging utility.

## Install

Create and prepare the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[api,parsing,indexing,reranking,dev]"
```

The first BGE-M3 run may download model weights from Hugging Face unless you
point `RAG_API_BGE_MODEL` or `RAG_API_BGE_CACHE_DIR` at an existing local cache.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Most unit tests use fake BGE/Qdrant adapters and do not require a running Qdrant
server or real model load.

## Start Qdrant

With Docker:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## Start The API

Server-backed Qdrant:

```powershell
$env:RAG_API_QDRANT_URL="http://localhost:6333"
$env:RAG_API_QDRANT_COLLECTION="rag_chunks"
$env:RAG_API_DENSE_VECTOR_NAME="dense"
$env:RAG_API_SPARSE_VECTOR_NAME="sparse"
$env:RAG_API_LLM_PROVIDER="mock"
$env:RAG_AUTH_MODE="jwt"
$env:RAG_JWT_SECRET="replace-this-local-secret"

.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

Embedded local Qdrant storage instead of a server:

```powershell
$env:RAG_API_QDRANT_PATH="artifacts\collections\default\qdrant_hybrid"
$env:RAG_API_QDRANT_COLLECTION="rag_chunks"
```

The runtime only supports `qdrant_hybrid` and `bge-m3`; those are the defaults.

## Start The Document Worker

Uploads and reindex requests create durable rows in `document_jobs`. Run a
separate worker process to execute parsing, chunking, BGE-M3 dense+sparse
encoding, Qdrant hybrid upsert, retry, cancellation checks, and progress updates:

```powershell
.\.venv\Scripts\python.exe scripts\document_worker.py
```

For a smoke test that claims at most one runnable job:

```powershell
.\.venv\Scripts\python.exe scripts\document_worker.py --once
```

The API process watches the collection artifact timestamps and reloads its
runtime after the worker rebuilds the collection files.

## API Endpoints

- `GET /healthz`
- `POST /api/auth/admin/login`
- `POST /api/auth/chat/login`
- `GET /api/me`
- `POST /api/retrieve`
- `POST /api/chat`
- `POST /api/models`
- `GET /api/documents`
- `POST /api/documents`
- `PATCH /api/documents/{document_id}/permissions`
- `POST /api/documents/{document_id}/reindex`
- `DELETE /api/documents/{document_id}`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel`
- `POST /api/jobs/{job_id}/retry`
- `POST /api/jobs/run-next`

Document upload processing now runs through the durable `document_jobs` table:

```text
queued -> running -> parsing -> chunking -> embedding -> indexing -> ready
```

Failed jobs move to `retrying` until `max_attempts` is exhausted, then `failed`.
Queued and running jobs can be cancelled from the API or administrator frontend.

Each upload can include access fields:

- tenant
- owner
- groups
- user IDs
- classification

Those fields are copied into chunk metadata and Qdrant payloads for retrieval
permission filtering.

## Frontend Test Flow

Open the administrator login page:

```text
D:\codex project\my rag\frontend\admin-login.html
```

Default local administrator credentials:

```text
admin / admin123
```

Open the user Q&A login page:

```text
D:\codex project\my rag\frontend\chat-login.html
```

Default local user credentials:

```text
user / user123
```

The two frontends keep separate browser sessions. After login, each request sends
the issued JWT as an `Authorization: Bearer ...` header. The administrator
frontend calls document management APIs; the user frontend only calls chat,
retrieval, and model APIs.

Keep `scripts/document_worker.py` running while testing uploads. Document rows
show the latest job status and refresh through parsing/chunking/indexing until
the document status is `ready`.

Verify Qdrant:

```text
http://localhost:6333/dashboard
```

The `rag_chunks` collection should contain both a dense vector named `dense` and
a sparse vector named `sparse`.

## LLM Providers

Mock mode is the default for backend smoke tests:

```powershell
$env:RAG_API_LLM_PROVIDER="mock"
```

Ollama:

```powershell
ollama pull llama3.1
$env:RAG_API_LLM_PROVIDER="ollama"
$env:RAG_API_OLLAMA_BASE_URL="http://localhost:11434"
$env:RAG_API_OLLAMA_MODEL="llama3.1"
```

OpenAI-compatible endpoint:

```powershell
$env:RAG_API_LLM_PROVIDER="openai-compatible"
$env:RAG_API_LLM_BASE_URL="https://api.openai.com/v1"
$env:RAG_API_LLM_MODEL="gpt-4o-mini"
$env:OPENAI_API_KEY="..."
```

## Authentication And Authorization

The API supports four auth modes:

- `disabled`: local demo mode; request ACL fields are accepted.
- `dev`: trusted local simulation using environment variables or `x-rag-*` headers.
- `jwt`: local JWT login mode using `/api/auth/admin/login` and
  `/api/auth/chat/login`.
- `oidc`: JWT/OIDC mode using JWKS and claim mapping.

Local JWT auth example:

```powershell
$env:RAG_AUTH_MODE="jwt"
$env:RAG_JWT_SECRET="replace-this-local-secret"
$env:RAG_JWT_ADMIN_USERNAME="admin"
$env:RAG_JWT_ADMIN_PASSWORD="admin123"
$env:RAG_JWT_USER_USERNAME="user"
$env:RAG_JWT_USER_PASSWORD="user123"
```

Development auth example:

```powershell
$env:RAG_AUTH_MODE="dev"
$env:RAG_DEV_TENANT_ID="tenant-a"
$env:RAG_DEV_USER_ID="u1"
$env:RAG_DEV_GROUP_IDS="hr"
$env:RAG_DEV_MAX_CLASSIFICATION="confidential"
$env:RAG_DEV_SCOPES="rag:chat,rag:retrieve,rag:models,rag:documents"
```

Per-request dev overrides:

```text
x-rag-subject: alice
x-rag-tenant-id: tenant-a
x-rag-user-id: u-alice
x-rag-group-ids: hr,finance
x-rag-scopes: rag:chat,rag:retrieve
x-rag-max-classification: confidential
```

Optional audit logging:

```powershell
$env:RAG_AUDIT_LOG_PATH="artifacts\audit\api_audit.jsonl"
$env:RAG_AUDIT_LOG_QUERY="true"
```
