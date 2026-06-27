import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.vectorstore import search
from openai import OpenAI

"""
ChromaDB specific
import chromadb
client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_collection("async-rust-docs")
"""

# load environment variables from .env file
# override=True ensures .env values take precedence over any existing
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# load the same embedding model used during ingestion,
# a different model produces incompatible vector spaces
# and similarity search returns garbage results
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# cross-encoder reranker : scores each (question, chunk) pair directly
# more accurate than cosine similarity but slower, so only used on top-20
# candidates retrieved by the fast dense vector search, not the full corpus
reranker = CrossEncoder("BAAI/bge-reranker-base")

openai = OpenAI(api_key=api_key)

def query(question:str, n_results:int = 5)->str:
    """
    Full RAG pipeline: embed question → retrieve candidates → rerank → generate.
 
    Args:
        question:  the user's debugging question
        n_results: number of top chunks to pass to the LLM after reranking
 
    Returns:
        tuple of (answer: str, sources: list[str], context_texts: list[str])
        - answer:        LLM-generated response grounded in retrieved docs
        - sources:       file paths of the top-k retrieved chunks
        - context_texts: raw text of the top-k retrieved chunks
                         (used by RAGAS evaluation for faithfulness scoring)
    """

    # embed the question using the same model as ingestion
    question_embedding = model.encode([question])[0]

    # retrieve top-20 candidates via dense vector similarity
    candidates = search(question_embedding, n_results=20)

    # re-rank down to top-5
    pairs = [[question, c.payload["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    # sort by reranker score descending, take top-k
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_results = [item for item, score in ranked[:n_results]]

    # build context from reranked chunks and construct the prompt
    for r in top_results:
        print(r.payload["source"], r.payload["chunk_index"])

    context_texts = [r.payload["text"] for r in top_results]
    context = "\n\n---\n\n".join(context_texts)
    sources = [r.payload["source"] for r in top_results]

    # the prompt instructs the LLM to synthesize across excerpts rather than
    # pattern-matching exact phrasing, this reduced unnecessary refusals
    # from 9/20 to ~2/20 in RAGAS evaluation (answer relevancy 0.48 → 0.74)
    prompt = f"""You are an expert in async Rust debugging.
                Answer the question using the information in the excerpts below. You may 
                synthesize and connect information across multiple excerpts, even if no 
                single excerpt uses the exact same wording as the question. Only say the 
                excerpts don't cover this if the underlying concept is genuinely absent, 
                not just differently phrased.

                Context:
                {context}

                Question: {question}
                Answer:"""
    
    # generate the answer, grounded in the retrieved context
    response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content": prompt}])
    answer = response.choices[0].message.content

    return answer, sources, context_texts

if __name__ == "__main__":
     # before/after reranking comparison,  useful for debugging retrieval quality
    question = "what is the difference between spawn and spawn_blocking?"

    question_embedding = model.encode([question])[0]

    raw_candidates = search(question_embedding, n_results=5)

    print("--- BEFORE reranking (raw top-5) ---")

    for c in raw_candidates:
        print(c.payload["source"])
    
    answer, sources = query(question)
    print("\n--- AFTER reranking ---")
    print(answer)
    print(sources)


    """
        ---- Search in ChromaDB ----
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_results
        )
        context = "\n\n---\n\n".join(results["documents"][0])
    """