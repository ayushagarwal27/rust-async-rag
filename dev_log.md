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

**What is built:**

- CLI loop: user question - embed - ChromaDB retrieval - LLM prompt - answer
- Exit with 'exit'

**What worked:**

- "how does tokio schedule tasks" : accurate answer about spawn, select!, 'static bounds
- "how does tracing work with async" : partial but correct answer

**What didn't work:**

- "how do I debug a deadlock" : no answer, gap in corpus
- "spawn vs spawn_blocking" : not enough content retrieved

**Root cause found:**

- Duplicate chunk IDs (chunk_0, chunk_1...) across files
- Only last ingested file survived in ChromaDB
- Total chunks: 26 instead of expected 313

**Fix:** prefix IDs with sanitized source filename

**Result:** 313 chunks across 66 files now correctly stored

**Lesson:**

- A RAG is only as good as what's actually in the vector DB
- Always verify collection.count() and list sources before trusting query results
