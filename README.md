# 🦀 Async Rust AI Chatbot

A retrieval-augmented generation chatbot for debugging async Rust. Ingests 66 markdown files from the Rust async book, tokio docs, and the tracing crate : chunks them, embeds them into Pinecone, and answers questions grounded in those sources via a streaming React UI, FastAPI, Gradio, and MCP tools.

**Live demo:** [huggingface.co/spaces/Ayush027/async-rust-rag](https://huggingface.co/spaces/Ayush027/async-rust-rag)

---

## How the chatbot works

```text
User message
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Guardrail classifier  (gpt-4o-mini + Pydantic  │
│  structured output)                             │
│                                                 │
│  rag ──────────────────────────────────────►┐  │
│  direct (greeting / small talk) ──────────►┐│  │
│  off_topic ── canned refusal, no LLM call  ││  │
└────────────────────────────────────────────┼┼──┘
                                             ││
          ┌──────────────────────────────────┘│
          │  direct branch                    │
          ▼                                   │  rag branch
   gpt-4o-mini (no retrieval)                 │
          │                                   ▼
          │                        HyDE — generate a
          │                        hypothetical doc passage
          │                               │
          │                               ▼
          │                        Pinecone dense search
          │                        top-10 candidates
          │                               │
          │                               ▼
          │                        CrossEncoder rerank
          │                        (ms-marco-MiniLM-L-6-v2)
          │                        top-10 → top-3
          │                               │
          │                               ▼
          │                        gpt-4o-mini grounded
          │                        on retrieved context
          │                               │
          └───────────────────────────────┘
                                          │
                                          ▼
                               Streaming SSE response
                               + conversation history saved
                               to Redis (hot) + MongoDB (cold)
```

**Key design choices:**

- **HyDE (Hypothetical Document Embeddings)** — before searching Pinecone, the LLM writes a short passage that would answer the question. Embedding that passage instead of the raw question bridges the vocabulary gap between questions and documentation.
- **Two-stage retrieval** — dense search returns 10 candidates; the CrossEncoder reranker scores every (question, passage) pair and keeps the top 3. This doubled answer relevancy vs. dense-only (0.48 → 0.74).
- **Pydantic structured output for routing** — the classifier uses `with_structured_output` so the LLM is forced into a valid `Literal["rag", "direct", "off_topic"]`. No string parsing or normalisation needed.
- **Session persistence** — history is keyed by `session_id`. Redis (24 h TTL) is the hot path; MongoDB is the fallback on cache miss or server restart. Both are written concurrently on every turn.
- **Code-aware chunking** — pre-splits on triple-backtick fences before paragraph splitting so code blocks are never cut mid-block. Chunks are ~500 tokens with 50-token overlap.

---

## RAGAS Evaluation

Evaluated on 20 corpus-vetted question/ground-truth pairs.

| Metric            | Phase 1 (dense only) | Phase 2 (+ reranking) |
| ----------------- | -------------------- | --------------------- |
| Faithfulness      | 0.78                 | 0.88                  |
| Answer Relevancy  | 0.48                 | 0.74                  |
| Context Precision | 0.91                 | 0.87                  |

---

## Project structure

```text
async-rust-rag/
├── app.py                        # Gradio chat UI (HuggingFace Spaces entry point)
├── mcp_server.py                 # MCP server — 3 tools for Claude integration
├── pyproject.toml
├── uv.lock
│
├── backend/
│   ├── .env                      # Secret keys (gitignored)
│   ├── .env.example              # Template — copy to .env and fill in values
│   ├── config.py                 # Pydantic Settings — single source for all env vars
│   ├── question_bank.json        # 20 QA pairs used for RAGAS evaluation
│   │
│   ├── rag/
│   │   ├── chunk.py              # Code-aware markdown chunker
│   │   ├── embed.py              # Ingestion pipeline: chunk → embed → upsert to Pinecone
│   │   ├── vectorstore.py        # Pinecone index creation + LangChain PineconeVectorStore
│   │   ├── query.py              # LangGraph RAG pipeline (classify → retrieve → rerank → generate)
│   │   ├── session.py            # Redis + MongoDB conversation history per session_id
│   │   └── evaluate.py           # RAGAS evaluation runner
│   │
│   └── api/
│       ├── app.py                # FastAPI app — CORS, static files, route registration
│       └── routes/
│           ├── rag.py            # POST /api/rag/query  (legacy single-turn)
│           └── chat.py           # POST /api/chat  POST /api/chat/stream  GET /api/chat/history/{id}
│
├── frontend/
│   ├── index.html
│   ├── vite.config.js            # /api proxy → :8000 in dev
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               # React chat UI — streaming SSE, session management, history load
│       └── App.css
│
├── knowledge_base/               # Source markdown docs (gitignored, ~66 files)
│   ├── async_book/
│   ├── tokio/
│   └── tracing/
│
└── images/
    └── Screenshot1.png
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

```bash
cp backend/.env.example backend/.env
# then fill in backend/.env with your keys
```

Required variables:

| Variable                   | Description                                                                   |
| -------------------------- | ----------------------------------------------------------------------------- |
| `OPENAI_API_KEY`           | Used for embeddings (`text-embedding-3-large`) and generation (`gpt-4o-mini`) |
| `PINECONE_API_KEY`         | Pinecone serverless vector store                                              |
| `PINECONE_INDEX_NAME`      | Defaults to `async-rust-docs`                                                 |
| `UPSTASH_REDIS_REST_URL`   | Upstash Redis REST endpoint (not a socket URL)                                |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis auth token                                                      |
| `MONGO_URL`                | MongoDB connection string                                                     |
| `MONGO_DB`                 | Database name, defaults to `rustrag`                                          |

### 3. Ingest the knowledge base (one-time)

Place your markdown files under `knowledge_base/` then run:

```bash
uv run python -m backend.rag.embed
```

This chunks, embeds, and upserts ~243 chunks into Pinecone. Re-running is safe — IDs are stable so it upserts rather than duplicates.

---

## Running the app

### FastAPI + React UI (recommended)

```bash
# Build the frontend first
cd frontend && npm install && npm run build && cd ..

# Start the API server
uv run uvicorn backend.api.app:app --reload
```

Open [http://localhost:8000](http://localhost:8000). The React app is served from `frontend/dist/`.

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend dev server (hot reload)

```bash
# Terminal 1 — backend
uv run uvicorn backend.api.app:app --reload

# Terminal 2 — frontend (proxies /api to :8000)
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Gradio UI

```bash
uv run python app.py
```

### MCP server (Claude integration)

```bash
uv run python mcp_server.py
```

Exposes three tools: `explain_stack_trace`, `search_async_patterns`, `find_tokio_examples`.

### CLI query

```bash
uv run python -m backend.rag.query
```

### RAGAS evaluation

```bash
uv run python -m backend.rag.evaluate
# writes eval_results.json
```

---

## API reference

| Method | Endpoint                         | Description                                                                    |
| ------ | -------------------------------- | ------------------------------------------------------------------------------ |
| `POST` | `/api/chat`                      | Multi-turn chat, returns `{answer, sources, session_id}`                       |
| `POST` | `/api/chat/stream`               | Streaming SSE — yields `{token}` per chunk, then `{done, sources, session_id}` |
| `GET`  | `/api/chat/history/{session_id}` | Fetch conversation history for a session                                       |
| `POST` | `/api/rag/query`                 | Legacy single-turn endpoint                                                    |

---

## Tech stack

| Layer             | Technology                                 |
| ----------------- | ------------------------------------------ |
| Embeddings        | OpenAI `text-embedding-3-large` (3072-dim) |
| Vector store      | Pinecone serverless                        |
| Reranker          | `cross-encoder/ms-marco-MiniLM-L-6-v2`     |
| LLM               | `gpt-4o-mini`                              |
| RAG orchestration | LangChain + LangGraph                      |
| Session store     | Upstash Redis (hot) + MongoDB (cold)       |
| API               | FastAPI + uvicorn                          |
| Frontend          | React + Vite                               |
| Gradio UI         | HuggingFace Spaces                         |
| MCP               | `fastmcp`                                  |
| Evaluation        | RAGAS                                      |
| Package manager   | `uv`                                       |
