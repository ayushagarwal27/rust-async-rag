## Phase 1

### Day 1

- Initialized project and set up the project structure
- Add relevant markdown files from tokyo.rs website, tracing repo and rust async book

### Day 2

- Chunked the documentation
- Split the documents using double newline as a separator
- Used the TikToken library for encoding and decoding text
- Added results to the chunk list

### Day 3

- Embed chunks and store them in ChromaDB
- Sentenced Transformer and `BAAI/bge-small-en-v1.5` for embedding

### Day 4

- Built CLI query loop: question → embed → ChromaDB retrieval → LLM prompt → answer
- Added 'exit' to quit
- Tested with sample questions — "tokio scheduling" and "tracing async" returned good answers, but "deadlock debugging" and "spawn vs spawn_blocking" had gaps
- Found root cause: duplicate chunk IDs (chunk_0, chunk_1...) caused only the last ingested file to survive in ChromaDB : 26 chunks instead of 313
- Fixed by prefixing IDs with sanitized source filename
- Re-ingested: 313 chunks across 66 files now correctly stored
- Lesson: always verify `collection.count()` and list sources before trusting query results

### Day 5

- Migrated from local ChromaDB to hosted Qdrant Cloud
- Why: prep for deployment — Streamlit/HF Spaces can't persist a local ChromaDB folder
- Created `vectorstore.py` with `create_collection`, `upsert_chunks`, `search`
- Re-ingested all 313 chunks into Qdrant Cloud
- Verified: query "how does tokio schedule tasks" returns same relevant results as ChromaDB

### Day 6

- Decided to use FastAPI instead of HF Spaces to get REST API response that integrates more naturally in a webapp
- Wrapped `query()` in a FastAPI endpoint `/api/rag/query`
- Fixed `query()` to return `(answer, sources)` tuple — was returning only the answer string, causing `ValueError: too many values to unpack`

## Phase 2

### Day 1 : Code-aware chunking

- Rewrote `chunk_text` to handle code blocks separately from prose
- Added `split_by_code_blocks()` using `re.split(r"(```[\s\S]*?```)", text)`, splits markdown into alternating prose and code segments
- Code blocks are now atomic, never split regardless of size
- Tiny code blocks (under 50 tokens) merge with surrounding prose instead of becoming isolated chunks
- Fixed oversized single-paragraph edge case, paragraphs bigger than chunk_size get saved as their own chunk with `continue` to avoid double-appending
- Chunk count jumped from 313 → 556. Expected, because code blocks are now isolated and previously-skipped content is properly captured

**Lesson:** splitting on `\n\n` alone destroys code blocks : a markdown-aware pre-split step is necessary before any token-based chunking

### Day 2 : Re-ranking (CrossEncoder, BAAI/bge-reranker-base)

- Added reranking: retrieve top-20 via dense search, rerank down to top-5 with bge-reranker-base
- Before reranking: spawn vs spawn_blocking pulled mostly irrelevant chunks
- After reranking: io.md correctly surfaced 3 different chunks (indices 3, 4, 7) : confirmed not duplicates,genuinely the most relevant source
- Result: spawn vs spawn_blocking now answered correctly, including JoinHandle.abort() nuance

- Clarified: 556 chunk count was inflated, old chunks from pre-code-aware chunker were never deleted before re-ingesting, so they coexisted with new ones. After a clean delete + re-ingest, true count is 243.

### Day 3 — RAGAS evaluation (3-question pilot)

- Added 20 question/ground_truth pairs covering spawning, deadlocks, tracing, futures/pinning, channels, shutdown, Send/lifetime errors
- Fixed answer_relevancy nan issue: needed explicit OpenAIEmbeddings passed via LangchainEmbeddingsWrapper
- Scores: faithfulness 0.70, answer_relevancy 0.24, context_precision 0.78
- Investigated low answer_relevancy: caused by RAG correctly declining to answer even when retrieved context actually contained the answer in different phrasing
- Seem to be generation/prompt issue, not retrieval, context_precision confirms retrieval found the right chunks
