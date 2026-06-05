from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection("async-rust-docs")

def embed_and_store(chunks:list[dict]):
    text = [c["text"] for c in chunks]
    embeddings = model.encode(text, show_progress_bar=True)

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=text,
        metadatas=[{"source": c["source_file"]} for c in chunks]
    )

if __name__ =="__main__":
    from pathlib import Path
    from chunk import chunk_text

    # Ingest all files
    all_files = list(Path("knowledge_base").rglob("*.md"))
    print(f"Found {len(all_files)} files")

    for source_file_path in all_files:
        text = Path(source_file_path).read_text()
        doc_type=source_file_path.parts[1]
        #Chunk the data
        chunks = chunk_text(text=text, source_file=str(source_file_path), doc_type=doc_type)
        embed_and_store(chunks)   # Store in ChromaDB
        print(f"  ingested {source_file_path} → {len(chunks)} chunks")

    print(f"\nTotal chunks in DB: {collection.count()}")

    # Test Query
    query = "how does tokio schedule tasks"
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    print(f"\n--- top 3 results for: '{query}' ---")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nresult {i}:")
        print(doc[:200])
        print(f"source: {results['metadatas'][0][i]['source']}")