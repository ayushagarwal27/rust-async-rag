# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python RAG system that answers questions about async Rust debugging. It ingests 66 markdown files from the Rust async book, tokio docs, and tracing crate docs — chunks them, embeds them into Pinecone, and serves answers via CLI, FastAPI, Gradio UI, and MCP tools.

**Stack**: Python + uv | OpenAI text-embedding-3-large | Pinecone | LangChain + LangGraph | gpt-4o-mini | RAGAS evaluation

## Commands

```bash
# Install dependencies
uv sync

# Ingest knowledge base into Pinecone (one-time or after adding docs)
uv run python -m backend.rag.embed

# Interactive CLI query
uv run python -m backend.rag.query

# FastAPI server (run from project root)
uv run uvicorn backend.api.app:app --reload
# API docs: http://localhost:8000/docs
# UI:       http://localhost:8000  (after building frontend)

# Gradio UI
uv run python app.py

# MCP server (for Claude tool integration)
uv run python mcp_server.py

# RAGAS evaluation (20 corpus-vetted questions)
uv run python -m backend.rag.evaluate
# Writes eval_results.json with faithfulness, answer_relevancy, context_precision

# Utility scripts
uv run python -m backend.rag.chunk       # Tests chunker on a sample file
uv run python -m backend.rag.vectorstore # Deletes and recreates Pinecone index

# Frontend (React + Vite)
cd frontend && npm install   # first time only
cd frontend && npm run dev   # dev server on :5173 with /api proxy to :8000
cd frontend && npm run build # build to frontend/dist/ (served by FastAPI at /)
```

## Architecture

### Directory Structure

```
backend/
  config.py          — pydantic_settings: all env vars in one typed class
  question_bank.json — 20 QA pairs for RAGAS evaluation
  rag/
    chunk.py         — code-aware markdown chunker
    vectorstore.py   — Pinecone index creation + LangChain PineconeVectorStore
    embed.py         — ingestion pipeline (chunk → embed → upsert)
    query.py         — LangGraph RAG pipeline + public API (query/chat/chat_stream)
    session.py       — Redis + MongoDB conversation history per session_id
    evaluate.py      — RAGAS evaluation runner
  api/
    app.py           — FastAPI app, middleware, static files, route registration
    routes/
      rag.py         — POST /api/rag/query (legacy single-turn)
      chat.py        — POST /api/chat + POST /api/chat/stream (SSE)
frontend/            — React + Vite chat UI
knowledge_base/      — gitignored markdown source docs
app.py               — Gradio UI entry point
mcp_server.py        — MCP tools entry point
```

### Data Flow

**Ingestion** (run once, idempotent):

```
knowledge_base/*.md → chunk.py (500-token chunks, 50-token overlap, code-fence aware)
                    → embed.py (OpenAI text-embedding-3-large via LangChain, 3072-dim)
                    → vectorstore.py (upsert to Pinecone, stable string IDs)
```

Result: ~243 chunks in the `async-rust-docs` Pinecone index.

**Query pipeline** (`backend/rag/query.py`) — LangGraph `retrieve → rerank → generate`:

1. HyDE: generate a hypothetical doc passage to bridge vocabulary gap
2. Dense search top-10 candidates from Pinecone (`PineconeVectorStore.as_retriever`)
3. Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` CrossEncoder → top-3
4. Generate with gpt-4o-mini

### Key Design Decisions

**Code-aware chunking** (`backend/rag/chunk.py`): Pre-splits on triple-backtick fences before paragraph splitting so code blocks are never split mid-block. Code blocks under 50 tokens merge with prose; larger ones are isolated.

**Reranking**: Two-stage retrieval (dense → CrossEncoder) was added in Phase 2 after Phase 1 missed `spawn_blocking` vs `spawn` distinctions. Doubled answer relevancy from 0.48 → 0.74. Uses `ms-marco-MiniLM-L-6-v2` (k=10 → top-3) for ~10x lower latency vs `bge-reranker-base`.

**Session persistence** (`backend/rag/session.py`): conversation history is keyed by `session_id`. Redis (24h TTL) is the hot path; MongoDB is the persistent fallback on cache miss or server restart. Both are written on every turn via `asyncio.gather`.

**pydantic_settings** (`backend/config.py`): all env vars are declared as a typed `Settings` class. Loaded once at startup; every module imports `settings` from there instead of calling `os.getenv()` directly.

**Idempotent ingestion**: `embed.py` generates stable string IDs from `(source_file, chunk_index)` so re-running upserts rather than duplicates.

### Interfaces

| File | Interface | Notes |
|------|-----------|-------|
| `backend/rag/query.py` | CLI | `__main__` block runs a sample query |
| `backend/api/app.py` | FastAPI REST | CORS restricted to `https://rustler.in` |
| `app.py` | Gradio UI | Deployed to HuggingFace Spaces |
| `mcp_server.py` | MCP tools | 3 tools: `explain_stack_trace`, `search_async_patterns`, `find_tokio_examples` |

### Environment Variables (required in `.env`)

```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=async-rust-docs
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
MONGO_URL=mongodb+srv://...
MONGO_DB=rustrag
```

### Knowledge Base

`knowledge_base/` is gitignored (reproducible). Organized into:

- `async_book/` — Rust async programming book chapters
- `tokio/` — tokio.rs tutorials and topic docs
- `tracing/` — tracing crate docs

### Evaluation

`backend/question_bank.json` contains 20 corpus-vetted QA pairs. Phase 2 results:

- Faithfulness: 0.88
- Answer Relevancy: 0.74
- Context Precision: 0.87

### Cleanup

The old `src/` directory is superseded by `backend/` and can be deleted:

```bash
rm -rf src/
```
