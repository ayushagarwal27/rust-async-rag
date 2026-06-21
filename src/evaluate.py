import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from query import query

def load_eval_data(path:str = "question_bank.json") -> list[dict]:
    data = json.load(open("question_bank.json"))

    if isinstance(data, dict):
        data = data.get("eval_data", data)
    
    return data


def run_rag_on_eval_set(eval_data:list[dict])-> list[dict]:
    results = []

    for i, item in enumerate(eval_data):
        question = item['question']
        ground_truth = item['ground_truth']

        print(f"[{i+1}/{len(eval_data)}] {question[:60]}...")

        answer, sources, contexts = query(question)

        results.append({
            "question":question,
            "answer":answer,
            "ground_truth":ground_truth,
            "contexts":contexts
        })
    
    return results

def build_ragas_dataset(results:list[dict])-> Dataset:
    return Dataset.from_list(results)

ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", max_tokens=2000)  # increase from default
)

def run_evaluations(dataset:Dataset):
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    scores = evaluate(
        dataset, 
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return scores

if __name__ == "__main__":
    eval_data = load_eval_data()
    # eval_data = eval_data[:3]

    print(f"Running RAG on {len(eval_data)} evaluation questions...\n")
    results = run_rag_on_eval_set(eval_data)

    print("\nBuilding RAGAS dataset...")
    dataset = build_ragas_dataset(results)

    print("Running evaluation (this may take a few minutes)...\n")
    scores = run_evaluations(dataset)

    print("\n=== RAGAS Scores ===")
    print(scores)

    # save results
    output_path = Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {"scores": scores.to_pandas().to_dict(orient="records")[0] if hasattr(scores, "to_pandas") else str(scores),
            "per_question": results},
            f,
            indent=2,
        )
    print(f"\nSaved detailed results to {output_path}")