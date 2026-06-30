# Enterprise RAG

This repository starts with the first enterprise RAG building block:

```text
local files -> Docling parser adapter -> normalized ParsedDocument output
ParsedDocument -> structure-aware parent/child/table chunks
chunks -> embedding records -> vector index + BM25/sparse index
index records + chunks -> hybrid retrieval -> expanded context output
```

The current implementation is intentionally small and modular. Later modules
such as generation, ACL policy services, and evaluation can build on the same
normalized document model.

## Current Modules

- `app/connectors/local_file.py`: discovers local files and records stable file metadata.
- `app/parsing/docling_parser.py`: lazy Docling adapter with a clean parser interface.
- `app/parsing/normalizer.py`: converts parser output into `ParsedDocument`.
- `app/chunking/pipeline.py`: converts `ParsedDocument` into retrieval-ready chunks.
- `app/chunking/hierarchical.py`: builds structure-aware parent and child chunks.
- `app/chunking/table_chunker.py`: isolates tables so they are not split into text chunks.
- `app/indexing/indexer.py`: turns retrieval chunks into dense and sparse index records.
- `app/indexing/embeddings/hashing.py`: deterministic local embeddings for tests.
- `app/indexing/embeddings/bge.py`: optional BGE-M3 dense embedding adapter.
- `app/indexing/vectorstores/memory.py`: local in-memory vector store for tests.
- `app/indexing/vectorstores/qdrant.py`: optional Qdrant vector store adapter.
- `app/indexing/sparse/memory_bm25.py`: local BM25 sparse index for tests and demos.
- `app/retrieval/pipeline.py`: hybrid dense + sparse retrieval, RRF fusion, rerank hook, and context assembly.
- `app/retrieval/query.py`: lightweight query analysis for table, exact, summary, and general queries.
- `app/retrieval/filters.py`: tenant, ACL, classification, and metadata visibility checks.
- `app/retrieval/parent_expander.py`: expands child hits to parent context while preserving table chunks.
- `app/retrieval/rerankers/bge.py`: optional local BGE cross-encoder reranker.
- `app/retrieval/rerankers/qwen.py`: optional local Qwen3 cross-encoder reranker.
- `app/retrieval/rerankers/cohere.py`: optional Cohere API reranker.
- `app/retrieval/rerankers/voyage.py`: optional Voyage AI API reranker.
- `app/retrieval/rerankers/factory.py`: creates rerankers from provider configuration.
- `app/domain/schemas.py`: shared document, element, table, citation, and metadata schemas.
- `scripts/parse_local.py`: parses one file or a folder and writes JSON/Markdown outputs.
- `scripts/chunk_parsed.py`: chunks one parsed JSON file or a folder of parsed JSON files.
- `scripts/index_chunks.py`: builds local index records and an index manifest from chunk JSON.
- `scripts/retrieve.py`: runs local hybrid retrieval against index JSON and chunk JSON.

## Run Unit Tests

The unit tests use Python's built-in `unittest` runner and do not require
Docling to be installed.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Run A Real Docling Parse

Install the optional parsing dependency first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[parsing,dev]"
```

Then parse a file or folder:

```powershell
.\.venv\Scripts\python.exe scripts\parse_local.py --input tests\fixtures\sample.md --output artifacts\parsed
```

## Chunk Parsed Documents

After parsing, generate retrieval-ready chunks:

```powershell
.\.venv\Scripts\python.exe scripts\chunk_parsed.py --input artifacts\parsed\sample.parsed.json --output artifacts\chunks
```

The chunk output includes parent chunks for context reconstruction, child chunks
for retrieval, isolated table chunks, inherited access controls, citations,
section paths, stable hashes, and contextual text for later hybrid indexing.

## Index Chunks

Create local dense/sparse index records with deterministic hashing embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\index_chunks.py --input artifacts\chunks\sample.chunks.json --output artifacts\index
```

The local indexing path writes `*.index.json` plus `index_manifest.json`. It is
intended for deterministic tests and pipeline validation. For production, install
the optional indexing adapters and configure BGE-M3 plus Qdrant:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[indexing]"
```

Install reranking dependencies for local cross-encoder rerankers:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[reranking]"
```

Run a real embedding + persistent Qdrant index locally:

```powershell
.\.venv\Scripts\python.exe scripts\index_chunks.py --input artifacts\chunks\sample.chunks.json --output artifacts\index_bge_qdrant --embedding-provider bge-m3 --backend qdrant --qdrant-path artifacts\qdrant --qdrant-collection rag_chunks
```

The first BGE-M3 run downloads the model from Hugging Face. On a compatible GPU,
add `--bge-use-fp16`; on CPU, leave fp16 disabled.

To use a running Qdrant service instead of local embedded storage:

```powershell
.\.venv\Scripts\python.exe scripts\index_chunks.py --input artifacts\chunks\sample.chunks.json --output artifacts\index_bge_qdrant --embedding-provider bge-m3 --backend qdrant --qdrant-url http://localhost:6333 --qdrant-collection rag_chunks
```

## Retrieve Contexts

Run hybrid retrieval against the local index JSON and original chunk JSON:

```powershell
.\.venv\Scripts\python.exe scripts\retrieve.py --index artifacts\index\sample.index.json --chunks artifacts\chunks\sample.chunks.json --query "health insurance" --output artifacts\retrieval\sample.retrieval.json
```

Retrieve from persistent Qdrant with the embedding provider recorded in the
index manifest:

```powershell
.\.venv\Scripts\python.exe scripts\retrieve.py --index artifacts\index_bge_qdrant\sample.index.json --chunks artifacts\chunks\sample.chunks.json --backend qdrant --qdrant-path artifacts\qdrant --qdrant-collection rag_chunks --query "health insurance" --output artifacts\retrieval\sample.qdrant.retrieval.json
```

Enable local BGE reranking after dense + sparse retrieval and RRF fusion:

```powershell
.\.venv\Scripts\python.exe scripts\retrieve.py --index artifacts\index\sample.index.json --chunks artifacts\chunks\sample.chunks.json --query "health insurance" --reranker bge --reranker-model BAAI/bge-reranker-v2-m3 --rerank-top-k 40 --top-k 5 --output artifacts\retrieval\sample.bge.reranked.json
```

Enable Qwen3 reranking:

```powershell
.\.venv\Scripts\python.exe scripts\retrieve.py --index artifacts\index\sample.index.json --chunks artifacts\chunks\sample.chunks.json --query "health insurance" --reranker qwen --reranker-model Qwen/Qwen3-Reranker-0.6B --rerank-top-k 40 --top-k 5 --output artifacts\retrieval\sample.qwen.reranked.json
```

For hosted rerankers, set the API key in the environment or pass
`--reranker-api-key`:

```powershell
$env:COHERE_API_KEY="..."
.\.venv\Scripts\python.exe scripts\retrieve.py --index artifacts\index\sample.index.json --chunks artifacts\chunks\sample.chunks.json --query "health insurance" --reranker cohere --reranker-model rerank-v4.0-pro

$env:VOYAGE_API_KEY="..."
.\.venv\Scripts\python.exe scripts\retrieve.py --index artifacts\index\sample.index.json --chunks artifacts\chunks\sample.chunks.json --query "health insurance" --reranker voyage --reranker-model rerank-2.5
```

The default retrieval path uses deterministic hashing embeddings, local BM25,
RRF fusion, a reranker abstraction with a deterministic no-op reranker, ACL
filtering, and parent context expansion. The production path can use BGE-M3 plus
Qdrant for dense retrieval, then BGE/Qwen/Cohere/Voyage reranking without
changing the retrieval response contract.

## Run the FastAPI Service

Install the API extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[api]"
```

Start the local service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

The first service layer exposes:

- `GET /healthz`
- `POST /api/retrieve`
- `POST /api/chat`
- `POST /api/models`
- `GET /api/documents`
- `POST /api/documents`
- `PATCH /api/documents/{document_id}/permissions`
- `POST /api/documents/{document_id}/reindex`
- `DELETE /api/documents/{document_id}`

By default, the API runs with the local sample index, in-memory retrieval, and
a mock LLM provider so it can be tested without external services. Configure
runtime paths and providers with environment variables:

```powershell
$env:RAG_API_INDEX_PATH="artifacts\index\sample.index.json"
$env:RAG_API_CHUNKS_PATH="artifacts\chunks\sample.chunks.json"
$env:RAG_API_BACKEND="memory"
$env:RAG_API_LLM_PROVIDER="mock"
```

Document uploads run parse -> chunk -> index in a background task. The document
indexing path defaults to BGE-M3 hybrid embeddings and Qdrant hybrid storage, so
both dense and sparse vectors are written to the configured Qdrant collection:

```powershell
$env:RAG_DOCUMENTS_BACKEND="qdrant_hybrid"
$env:RAG_DOCUMENTS_EMBEDDING_PROVIDER="bge-m3"
$env:RAG_API_BACKEND="qdrant_hybrid"
$env:RAG_API_EMBEDDING_PROVIDER="bge-m3"
$env:RAG_API_QDRANT_URL="http://localhost:6333"
$env:RAG_API_QDRANT_COLLECTION="rag_chunks"
$env:RAG_API_DENSE_VECTOR_NAME="dense"
$env:RAG_API_SPARSE_VECTOR_NAME="sparse"
```

For a local embedded Qdrant store instead of a server, set
`RAG_API_QDRANT_PATH`, for example `artifacts\collections\default\qdrant_hybrid`,
and leave `RAG_API_QDRANT_URL` unset. The frontend document panel posts access
fields with each upload: tenant, owner, groups, user IDs, and classification.

### Ollama Local LLM

Start Ollama locally, pull a model, then run the API:

```powershell
ollama pull llama3.1
$env:RAG_API_LLM_PROVIDER="ollama"
$env:RAG_API_OLLAMA_BASE_URL="http://localhost:11434"
$env:RAG_API_OLLAMA_MODEL="llama3.1"
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

In the frontend settings:

- 模式: `后端接口`
- 接口地址: `http://localhost:8000/api/chat`
- 模型接入: `Ollama 本地`
- Base URL: `http://localhost:11434`
- Click `获取模型`, then choose a model.

For an OpenAI-compatible model endpoint:

```powershell
$env:RAG_API_LLM_PROVIDER="openai-compatible"
$env:RAG_API_LLM_BASE_URL="https://api.openai.com/v1"
$env:RAG_API_LLM_MODEL="gpt-4o-mini"
$env:OPENAI_API_KEY="..."
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

In the frontend settings:

- 模式: `后端接口`
- 接口地址: `http://localhost:8000/api/chat`
- 模型接入: `OpenAI 兼容 API`
- Base URL: your `/v1` compatible endpoint
- API Key: the provider key, if required
- Click `获取模型`, then choose a model.

## Authentication And Authorization

The API supports three auth modes:

- `disabled`: local demo mode. Request ACL fields such as `tenant_id` and
  `group_ids` are still accepted for backwards-compatible local testing.
- `dev`: local enterprise simulation. The backend builds a trusted principal
  from environment variables or `x-rag-*` headers, then ignores caller-supplied
  ACL fields.
- `oidc`: production-oriented JWT/OIDC mode. The backend validates bearer
  tokens with JWKS and maps claims to the RAG permission context.

Development auth example:

```powershell
$env:RAG_AUTH_MODE="dev"
$env:RAG_DEV_TENANT_ID="tenant-a"
$env:RAG_DEV_USER_ID="u1"
$env:RAG_DEV_GROUP_IDS="hr"
$env:RAG_DEV_MAX_CLASSIFICATION="confidential"
$env:RAG_DEV_SCOPES="rag:chat,rag:retrieve,rag:models"
```

You can override the dev identity per request with headers:

```text
x-rag-subject: alice
x-rag-tenant-id: tenant-a
x-rag-user-id: u-alice
x-rag-group-ids: hr,finance
x-rag-scopes: rag:chat,rag:retrieve
x-rag-max-classification: confidential
```

OIDC example:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[api,auth]"
$env:RAG_AUTH_MODE="oidc"
$env:RAG_OIDC_ISSUER="https://login.example.com/realms/rag"
$env:RAG_OIDC_AUDIENCE="enterprise-rag"
$env:RAG_OIDC_JWKS_URL="https://login.example.com/realms/rag/protocol/openid-connect/certs"
$env:RAG_OIDC_TENANT_CLAIM="tenant_id"
$env:RAG_OIDC_GROUPS_CLAIM="groups"
$env:RAG_OIDC_SCOPES_CLAIM="scope"
```

When `RAG_AUTH_MODE` is `dev` or `oidc`, retrieval permissions are derived from
the authenticated principal. The frontend should only send the question and
model settings; tenant, user, groups, and classification come from the backend
identity context.

Optional audit logging:

```powershell
$env:RAG_AUDIT_LOG_PATH="artifacts\audit\api_audit.jsonl"
$env:RAG_AUDIT_LOG_QUERY="true"
```
