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
- `app/security/`: dev/OIDC auth, permission context, and audit logging.
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

.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

Embedded local Qdrant storage instead of a server:

```powershell
$env:RAG_API_QDRANT_PATH="artifacts\collections\default\qdrant_hybrid"
$env:RAG_API_QDRANT_COLLECTION="rag_chunks"
```

The runtime only supports `qdrant_hybrid` and `bge-m3`; those are the defaults.

## API Endpoints

- `GET /healthz`
- `POST /api/retrieve`
- `POST /api/chat`
- `POST /api/models`
- `GET /api/documents`
- `POST /api/documents`
- `PATCH /api/documents/{document_id}/permissions`
- `POST /api/documents/{document_id}/reindex`
- `DELETE /api/documents/{document_id}`

Document upload processing runs in a background task:

```text
uploaded -> parsing -> chunking -> indexing -> ready
```

Each upload can include access fields:

- tenant
- owner
- groups
- user IDs
- classification

Those fields are copied into chunk metadata and Qdrant payloads for retrieval
permission filtering.

## Frontend Test Flow

Open the frontend:

```text
D:\codex project\my rag\frontend\index.html
```

In settings, use:

```text
API URL: http://localhost:8000/api/chat
```

Open the document panel, choose a file, fill the access fields, and click upload.
The document list will refresh through parsing/chunking/indexing until the status
is `ready`.

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

The API supports three auth modes:

- `disabled`: local demo mode; request ACL fields are accepted.
- `dev`: trusted local simulation using environment variables or `x-rag-*` headers.
- `oidc`: JWT/OIDC mode using JWKS and claim mapping.

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
