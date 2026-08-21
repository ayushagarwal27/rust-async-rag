import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from backend.config import settings
from .query import query

# question_bank.json lives in backend/ alongside the rag package
_QUESTION_BANK = Path(__file__).parent.parent / "question_bank.json"
_OUTPUT_PATH   = Path(__file__).parent.parent.parent / "eval_results.json"

ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, max_tokens=2000)
)


def load_eval_data(path: Path = _QUESTION_BANK) -> list[dict]:
    """Load QA pairs from question_bank.json (bare list or {"eval_data": [...]} format)."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("eval_data", data)
    return data


def run_rag_on_eval_set(eval_data: list[dict]) -> list[dict]:
    """Run the full RAG pipeline on every question and collect answers + contexts."""
    results = []
    for i, item in enumerate(eval_data):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(eval_data)}] {question[:60]}...")
        answer, _, contexts = query(question)
        results.append({
            "question":     question,
            "answer":       answer,
            "ground_truth": ground_truth,
            "contexts":     contexts,
        })
    return results


def build_ragas_dataset(results: list[dict]) -> Dataset:
    return Dataset.from_list(results)


def run_evaluations(dataset: Dataset):
    """Score with RAGAS: faithfulness, answer_relevancy, context_precision."""
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    )
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )


if __name__ == "__main__":
    # usage: uv run python -m backend.rag.evaluate
    eval_data = load_eval_data()
    print(f"Running RAG on {len(eval_data)} evaluation questions...\n")
    results = run_rag_on_eval_set(eval_data)

    print("\nBuilding RAGAS dataset...")
    dataset = build_ragas_dataset(results)

    print("Running evaluation (this may take a few minutes)...\n")
    scores = run_evaluations(dataset)
    print("\n=== RAGAS Scores ===")
    print(scores)

    with open(_OUTPUT_PATH, "w") as f:
        if hasattr(scores, "to_pandas"):
            df = scores.to_pandas()
            aggregate = {
                col: round(df[col].mean(), 4)
                for col in ["faithfulness", "answer_relevancy", "context_precision"]
                if col in df.columns
            }
            per_question_scores = df.to_dict(orient="records")
        else:
            aggregate = str(scores)
            per_question_scores = []

        json.dump(
            {"aggregate": aggregate, "per_question_scores": per_question_scores, "per_question": results},
            f,
            indent=2,
        )
    print(f"\nSaved detailed results to {_OUTPUT_PATH}")
