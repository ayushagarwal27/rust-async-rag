# 🦀 Async Rust Debugging RAG

![Prototype Screenshot](/images/Screenshot1.png)

A retrieval-augmented generation (RAG) system for debugging async Rust code.
Ingests tokio.rs docs, the tracing crate, and the Rust async book into a
vector database and answers debugging questions about deadlocks, stuck futures,
task scheduling, and async stack traces — grounded only in the retrieved docs.

Built from scratch to understand the core RAG pipeline:
chunking, embedding, retrieval, generation — then wrapped as an API.

**Live demo:** [huggingface.co/spaces/Ayush027/async-rust-rag](https://huggingface.co/spaces/Ayush027/async-rust-rag)

## Tech Stack

- **Phase 1:** Python · Qdrant · sentence-transformers · tiktoken · FastAPI
- **Phase 2:** Code-aware chunking · BGE re-ranking · RAGAS evaluation · Gradio
- **Phase 3:** Agentic RAG
- **Phase 4:** MCP server

---

## Architecture

```
knowledge_base/ (66 markdown files)
   tokio.rs docs · tracing crate · Rust async book
        │
        ▼
   chunk.py — code-aware chunking, ~500-token chunks, 50-token overlap
        │
        ▼
   embed.py — embed chunks with BAAI/bge-small-en-v1.5
        │
        ▼
   Qdrant Cloud (hosted vector store, 243 chunks)
        │
        ▼
   query.py — embed question → retrieve top-20 → rerank to top-5
              (BAAI/bge-reranker-base CrossEncoder)
        │
        ▼
   gpt-4o-mini — answer grounded in retrieved context
        │
        ▼
   FastAPI endpoint (/api/rag/query) — answer + sources
   Gradio UI (HuggingFace Spaces)
```

---

## RAGAS Evaluation (Phase 2)

Evaluated on 20 question/ground_truth pairs covering spawning, deadlocks,
tracing, futures/pinning, channels, shutdown, and Send/lifetime errors.
All ground truths vetted against the actual corpus before scoring.

| Metric            | Before (Phase 1) | After (Phase 2) |
| ----------------- | ---------------- | --------------- |
| Faithfulness      | 0.78             | 0.88            |
| Answer Relevancy  | 0.48             | 0.74            |
| Context Precision | 0.91             | 0.87            |

Key finding: re-ranking resolved retrieval gaps (spawn vs spawn_blocking);
prompt tuning drove the largest jump in answer relevancy.

---

## How to run

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

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

### 4. Ask questions via CLI

```bash
uv run python src/query.py
```

### 5. Run as an API

```bash
cd src && uv run uvicorn server:app --reload
```

```bash
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "how do I debug a deadlock in tokio?"}'
```

### 6. Run the Gradio UI locally

```bash
uv run python app.py
```

---

## Project structure

```
async-rust-rag/
├── app.py                 # Gradio chat UI
├── knowledge_base/        # Source markdown docs (gitignored)
│   ├── tokio/
│   ├── tracing/
│   └── async_book/
├── src/
│   ├── chunk.py           # Code-aware chunker
│   ├── embed.py           # Ingestion + embedding pipeline
│   ├── query.py           # Retrieval + reranking + generation
│   ├── vectorstore.py     # Qdrant client, hybrid search, upsert
│   ├── server.py          # FastAPI app
│   └── evaluate.py        # RAGAS evaluation harness
├── question_bank.json     # 20 eval questions, corpus-vetted
├── dev_log.md             # build log — bugs found, fixes, lessons
├── pyproject.toml
└── uv.lock
```

---

## Dev log

The full build journey — including bugs hit and how they were fixed — is in
[`dev_log.md`](dev_log.md).
