from sentence_transformers import SentenceTransformer
from vectorstore import upsert_chunks

"""
Chroma DB specific
import chromadb
client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection("async-rust-docs")

def embed_and_store(chunks:list[dict], source_file:str):
    text = [c["text"] for c in chunks]
    embeddings = model.encode(text, show_progress_bar=True)

    safe_name = source_file.replace("/", "_").replace(".", "_")

    collection.upsert(
        ids=[f"{safe_name}_{i}" for i in range(len(chunks))], 
        embeddings=embeddings.tolist(),
        documents=text,
        metadatas=[{"source": c["source_file"]} for c in chunks]
    )
"""

# load the embedding model once at module level : shared across all calls
# BAAI/bge-small-en-v1.5 produces 384-dimensional dense vectors and is
# well-suited for technical/code-adjacent documentation retrieval.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


if __name__ =="__main__":
    from pathlib import Path
    from chunk import chunk_text

    # discover all markdown files in the knowledge base
    # rglob("*.md") walks all subdirectories recursively
    all_files = list(Path("knowledge_base").rglob("*.md"))
    print(f"Found {len(all_files)} files")


    # ---  ingest in Qdrant ----
    for source_file_path in all_files:
        # read with explicit utf-8 encoding 
        text = Path(source_file_path).read_text(encoding="utf-8")

        # derive doc_type from the top-level folder under knowledge_base/
        # e.g. knowledge_base/tokio/tutorial/async.md → doc_type = "tokio"
        # stored as metadata per chunk for filtering in Qdrant queries
        doc_type = source_file_path.parts[1]

        # chunk the document using the code-aware chunker
        # returns list of dicts: {text, source_file, doc_type, chunk_index}
        chunks = chunk_text(
            text = text,
            source_file=str(source_file_path),
            doc_type=doc_type
         )
        
        # encode all chunks in one batch : more efficient than one-by-one
        embeddings = model.encode([c["text"] for c in chunks])

        # upsert into Qdrant Cloud with both dense and sparse vectors
        # upsert makes re-running safe : existing points
        # with the same ID are overwritten rather than duplicated
        upsert_chunks(chunks, embeddings, str(source_file_path))
        print(f"✓ {source_file_path} → {len(chunks)} chunks")


    # ---- ingest in chromadb ----
    # for source_file_path in all_files:
    #     try:
    #         text = Path(source_file_path).read_text()
    #         doc_type=source_file_path.parts[1]
    #         #Chunk the data
    #         chunks = chunk_text(text=text, source_file=str(source_file_path), doc_type=doc_type)
    #         embed_and_store(chunks, str(source_file_path))   # Store in ChromaDB
    #         print(f"  ingested {source_file_path} → {len(chunks)} chunks")
    #     except Exception as e:
    #          print(f"✗ ERROR on {source_file_path}: {e}")
    #          continue

    # print(f"\nTotal chunks in DB: {collection.count()}")

    # Test Query
    # query = "how does tokio schedule tasks"
    # query_embedding = model.encode([query])[0]
    # # results = collection.query(
    # #     query_embeddings=[query_embedding.tolist()],
    # #     n_results=3
    # # )
    # print(f"\n--- top 3 results for: '{query}' ---")
    # for i, doc in enumerate(results["documents"][0]):
    #     print(f"\nresult {i}:")
    #     print(doc[:200])
    #     print(f"source: {results['metadatas'][0][i]['source']}")