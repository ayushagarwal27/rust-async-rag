# 🦀 Async Rust Debugging RAG

![Prototype Screenshot](/images/Screenshot1.png)

A retrieval-augmented generation (RAG) system for debugging async Rust code.
Ingests tokio.rs docs, the tracing crate, and the Rust async book into a local
vector database (ChromaDB) and answers debugging questions about deadlocks,
stuck futures, task scheduling, and async stack traces.

Build from scratch to understand the core RAG pipeline:
chunking, embedding, retrieval, generation.

## Tech Stack:

- Phase 1 : Python · ChromaDB · sentence-transformers · tiktoken
- Phase 2 : Agentic RAG
- Phase 3 : MCP server

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
   ChromaDB (local, persisted to ./data/chroma)
        │
        ▼
   query.py — embed question → retrieve top-k chunks
        │
        ▼
   gpt-4o-mini — answer ONLY from retrieved context
        │
        ▼
   Answer + sources
```

313 chunks indexed across the knowledge base.

---

## How to run

### 1. Install dependencies

```bash
uv sync
```

### 2. Add your OpenAI API key

Create a `.env` file:

```env
OPENAI_API_KEY=your-key-here
```

### 3. Build the vector store

```bash
uv run python src/embed.py
```

This chunks every file in `knowledge_base/`, embeds them, and stores them
in ChromaDB at `data/chroma/`.

### 4. Ask questions

```bash
uv run python src/query.py
```

```
Async Rust Debugging RAG
Type 'exit' to quit

Question: how do I debug a deadlock in tokio?
```

---

## Project structure

```
async-rust-rag/
├── data/chroma/        # ChromaDB storage (gitignored)
├── knowledge_base/      # Source markdown docs (gitignored)
│   ├── tokio/
│   ├── tracing/
│   └── async_book/
├── src/
│   ├── chunk.py
│   ├── embed.py
│   └── query.py
├── dev_log.md            # build log — bugs found, fixes, lessons
├── pyproject.toml
└── uv.lock
```

## Dev log

The full build journey — including bugs hit and how they were fixed — is in
[`dev_log.md`](dev_log.md).
