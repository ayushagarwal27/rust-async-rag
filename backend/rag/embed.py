"""
Ingestion pipeline: walk knowledge_base/, chunk all markdown files, and
upsert into Pinecone via LangChain's PineconeVectorStore.

IDs are derived deterministically from (source_file, chunk_index) so that
re-running ingestion is idempotent — existing vectors are overwritten.
"""

from langchain_core.documents import Document
from .vectorstore import get_vectorstore, create_index
from .chunk import chunk_text


if __name__ == "__main__":
    # usage: uv run python -m backend.rag.embed
    from pathlib import Path

    create_index()
    vectorstore = get_vectorstore()

    all_files = list(Path("knowledge_base").rglob("*.md"))
    print(f"Found {len(all_files)} files")

    for source_file_path in all_files:
        text = source_file_path.read_text(encoding="utf-8")
        doc_type = source_file_path.parts[1]

        chunks = chunk_text(
            text=text,
            source_file=str(source_file_path),
            doc_type=doc_type,
        )

        docs = [
            Document(
                page_content=c["text"],
                metadata={
                    "source": c["source_file"],
                    "doc_type": c["doc_type"],
                    "chunk_index": c["chunk_index"],
                },
            )
            for c in chunks
        ]

        safe_name = str(source_file_path).replace("/", "_").replace(".", "_")
        ids = [f"{safe_name}_{i}" for i in range(len(chunks))]

        vectorstore.add_documents(docs, ids=ids)
        print(f"✓ {source_file_path} → {len(chunks)} chunks")
