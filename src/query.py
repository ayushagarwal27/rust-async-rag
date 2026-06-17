import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, CrossEncoder
from vectorstore import search
import chromadb
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_collection("async-rust-docs")
openai = OpenAI(api_key=api_key)

def query(question:str, n_results:int = 5)->str:
    question_embedding = model.encode([question])[0]

    # ---- Search in ChromaDB ----
    # results = collection.query(
    #     query_embeddings=[question_embedding.tolist()],
    #     n_results=n_results
    # )
    # context = "\n\n---\n\n".join(results["documents"][0])

    # --- Search in Qdrant ----

    # retrieve top-20 candidates instead of top-5 directly
    candidates = search(question_embedding, n_results=20)

    # re-rank down to top-5
    pairs = [[question, c.payload["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_results = [item for item, score in ranked[:n_results]]

    for r in top_results:
        print(r.payload["source"], r.payload["chunk_index"])

    context = "\n\n---\n\n".join([r.payload["text"] for r in top_results])
    sources = [r.payload["source"] for r in top_results]


    prompt = f"""You are an expert in async Rust debugging.
    Use ONLY the following documentation excerpts to answer the question.
    If the answer isn't in the excerpts, say so.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content": prompt}])
    answer = response.choices[0].message.content
    # print(answer)
    return answer, sources

if __name__ == "__main__":
    question = "what is the difference between spawn and spawn_blocking?"

    question_embedding = model.encode([question])[0]

    raw_candidates = search(question_embedding, n_results=5)

    print("--- BEFORE reranking (raw top-5) ---")

    for c in raw_candidates:
        print(c.payload["source"])
    
    # after reranking
    answer, sources = query(question)
    print("\n--- AFTER reranking ---")
    print(answer)
    print(sources)