# 🦀 Async Rust Debugging RAG

![Prototype Screenshot](/images/Screenshot1.png)

A retrieval-augmented generation (RAG) system for debugging async Rust code.
Ingests tokio.rs docs, the tracing crate, and the Rust async book into a vector
database and answers debugging questions about deadlocks, stuck futures, task
scheduling, and async stack traces — grounded only in the retrieved docs.

Built from scratch to understand the core RAG pipeline:
chunking, embedding, retrieval, generation — then wrapped as an API.

## Tech Stack

- **Phase 1:** Python · Qdrant (hosted) · sentence-transformers · tiktoken · FastAPI
- **Phase 2:** Agentic RAG
- **Phase 3:** MCP server

---

## Architecture (Phase 1)

```
knowledge_base/ (66 markdown files)
   tokio.rs docs · tracing crate · Rust async book
        │
        ▼
   chunk.py — split into ~500-token chunks, 50-token overlap
        │
        ▼
   embed.py — embed chunks with BAAI/bge-small-en-v1.5
        │
        ▼
   Qdrant Cloud (hosted vector store)
        │
        ▼
   query.py — embed question → retrieve top-k chunks
        │
        ▼
   gpt-4o-mini — answer ONLY from retrieved context
        │
        ▼
   FastAPI endpoint (/api/rag/query) — answer + sources
```

313 chunks indexed across the knowledge base.

---

## How to run

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your-key-here
QDRANT_URL=https://xxxx.cloud.qdrant.io
QDRANT_API_KEY=your-key-here
```

### 3. Build the vector store

```bash
uv run python -c "from src.vectorstore import create_collection; create_collection()"
uv run python src/embed.py
```

This chunks every file in `knowledge_base/`, embeds them, and upserts them
into your Qdrant Cloud collection.

### 4. Ask questions via CLI

```bash
uv run python src/query.py
```

```
Async Rust Debugging RAG
Type 'exit' to quit

Question: how do I debug a deadlock in tokio?
```

### 5. Run as an API

```bash
uv run uvicorn server:app --reload
```

```bash
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "how do I debug a deadlock in tokio?"}'
```

```json
{
  "answer": "...",
  "sources": ["knowledge_base/tokio/topics/tracing-next-steps.md", "..."]
}
```

---

## Project structure

```
async-rust-rag/
├── knowledge_base/      # Source markdown docs (gitignored)
│   ├── tokio/
│   ├── tracing/
│   └── async_book/
├── src/
│   ├── __init__.py
│   ├── chunk.py
│   ├── embed.py
│   ├── query.py
│   └── vectorstore.py    # Qdrant client, search, upsert
├── server.py              # FastAPI app
├── dev_log.md             # build log : bugs found, fixes, lessons
├── pyproject.toml
└── uv.lock
```

<!-- ## Screenshots

<!-- paste screenshots below -->

---

## Dev log

The full build journey is in
[`dev_log.md`](dev_log.md).
