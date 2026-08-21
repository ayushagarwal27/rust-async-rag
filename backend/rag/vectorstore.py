from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from backend.config import settings

VECTOR_SIZE = 3072  # text-embedding-3-large output dimension

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=settings.openai_api_key,
)

_pc = Pinecone(api_key=settings.pinecone_api_key)


def create_index():
    """
    Create the Pinecone serverless index if it does not already exist.
    Safe to call repeatedly — skips creation if already present.
    """
    existing = [idx.name for idx in _pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        _pc.create_index(
            name=settings.pinecone_index_name,
            dimension=VECTOR_SIZE,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {settings.pinecone_index_name}")
    else:
        print(f"Pinecone index already exists: {settings.pinecone_index_name}")


def get_vectorstore() -> PineconeVectorStore:
    """
    Return a LangChain PineconeVectorStore bound to text-embedding-3-large.

    Used for both ingestion (add_documents) and retrieval (as_retriever).
    """
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=_embeddings,
    )


if __name__ == "__main__":
    # usage: uv run python -m backend.rag.vectorstore
    existing = [idx.name for idx in _pc.list_indexes()]
    if settings.pinecone_index_name in existing:
        _pc.delete_index(settings.pinecone_index_name)
        print(f"Deleted Pinecone index: {settings.pinecone_index_name}")
    create_index()
