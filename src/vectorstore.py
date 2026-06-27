from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os

from dotenv import load_dotenv

load_dotenv(override=True)

# collection name in Qdrant Cloud : used across all operations
COLLECTION_NAME="async-rust-docs"

# must match the output dimension of the embedding model (BAAI/bge-small-en-v1.5)
# if we switch embedding models, this value must be updated and the collection
# recreated, stored vectors and new vectors must have the same dimensionality
VECTOR_SIZE=384

# single shared client, initialized once at module level
# credentials are read from environment variables
client = QdrantClient(
     url=os.getenv("QDRANT_URL"), 
     api_key=os.getenv("QDRANT_API_KEY")
)

def create_collection():
    """
    Create the Qdrant collection with cosine similarity for dense vectors.
    Only needs to be run once, calling it again on an existing collection
    will raise an error, so check existence before calling in scripts.
    """
    client.create_collection(
        collection_name=COLLECTION_NAME, 
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    
def upsert_chunks(chunks: list[dict], embeddings, source_file:str):
     """
          Upsert a list of chunks (with their embeddings) into Qdrant.
          
          Uses upsert (not insert) so re-running ingestion is idempotent,
          existing points with the same ID are overwritten, not duplicated.
          
          Point IDs are derived deterministically from source_file + chunk index
          using SHA-256 to avoid collisions across files. Python's built-in hash()
          is not used because it is randomized per process (PYTHONHASHSEED) and
          would produce different IDs across runs, causing phantom duplicates.
          
          Args:
               chunks:      list of chunk dicts from chunk_text()
               embeddings:  numpy array of shape (num_chunks, VECTOR_SIZE)
               source_file: original file path, used to generate unique point IDs
     """
     safe_name = source_file.replace("/", "_").replace(".", "_")

     points = [
          PointStruct(
               # SHA-256 based ID — stable across runs, no collisions
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
     """
          Search for the most similar chunks to a query embedding.
          
          Uses dense cosine similarity, returns the top-n most similar points
          with their full payload (text, source, doc_type, chunk_index).
          
          In query.py this is called with n_results=20 to cast a wide net,
          then the CrossEncoder reranker narrows results down to top-5.
          
          Args:
               query_embedding: 1D numpy array of shape (VECTOR_SIZE,)
               n_results:       number of candidates to return
          
          Returns:
               list of ScoredPoint objects, access text via point.payload["text"]
     """
     results = client.query_points(
          collection_name=COLLECTION_NAME,
          query=query_embedding.tolist(),
          limit=n_results,
          with_payload=True
    )
     
     return results.points

if __name__ =="__main__":
     # reset the collection, deletes all existing points and recreates it
     # run this before re-ingesting to avoid stale data
     client.delete_collection(collection_name=COLLECTION_NAME)
     create_collection()