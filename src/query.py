import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_collection("async-rust-docs")
openai = OpenAI(api_key=api_key)

def query(question:str, n_results:int = 5)->str:
    question_embedding = model.encode([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=n_results
    )

    context = "\n\n---\n\n".join(results["documents"][0])
    prompt = f"""You are an expert in async Rust debugging.
    Use ONLY the following documentation excerpts to answer the question.
    If the answer isn't in the excerpts, say so.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    response = openai.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user", "content": prompt}])
    answer = response.choices[0].message.content
    print(answer)
    return answer

if __name__ == "__main__":
    print("Async Rust Debugging RAG")
    print("Type 'exit' to quit\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        query(question=question)
        print()