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
- Sentenced Transformer and "BAAI/bge-small-en-v1.5" for embedding

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
