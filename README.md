# 🦀 Async Rust Debugging RAG

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
