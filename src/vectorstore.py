from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os

from dotenv import load_dotenv

load_dotenv(override=True)

COLLECTION_NAME="async-rust-docs"
VECTOR_SIZE=384

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def create_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME, 
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    
def upsert_chunks(chunks: list[dict], embeddings, source_file:str):
     safe_name = source_file.replace("/", "_").replace(".", "_")

     points = [
          PointStruct(
               id=abs(hash(f"{safe_name}_{i}")) % (2**63),
               vector=embeddings[i].tolist(),
               payload={
                "text": chunks[i]["text"],
                "source": chunks[i]["source_file"],
                "doc_type": chunks[i]["doc_type"],
                "chunk_index": chunks[i]["chunk_index"]
               }
          )

          for i in range(len(chunks))
     ]

     client.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query_embedding, n_results:int = 5):
     results = client.query_points(
          collection_name=COLLECTION_NAME,
          query=query_embedding.tolist(),
          limit=n_results,
          with_payload=True
    )
     
     return results.points

if __name__ =="__main__":
     client.delete_collection(collection_name=COLLECTION_NAME)
     create_collection()