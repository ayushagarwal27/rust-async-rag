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
    """
        Load the evaluation dataset from a JSON file.
    
        Handles both bare list format ([{...}, {...}]) and wrapped dict format
        ({"eval_data": [{...}, {...}]}) in case the JSON was exported differently.
    
        The question_bank.json contains 20 question/ground_truth pairs covering:
        spawning, deadlocks, tracing, futures/pinning, channels, shutdown,
        and Send/lifetime errors.
    """
    data = json.load(open("question_bank.json"))

    # unwrap if exported as a dict with a named key
    if isinstance(data, dict):
        data = data.get("eval_data", data)
    
    return data


def run_rag_on_eval_set(eval_data:list[dict])-> list[dict]:
    """
        Run the full RAG pipeline on every question in the eval set.
    
        Calls query() for each item, which does: embed -> retrieve top-20 ->
        rerank to top-5 -> generate answer. The returned context_texts (raw
        retrieved chunk text, not just source filenames) are critical for
        RAGAS : faithfulness and context_precision need the actual text to
        verify claims against, not just file paths.
    
        Args:
            eval_data: list of {question, ground_truth} dicts
    
        Returns:
            list of {question, answer, ground_truth, contexts} dicts,
            ready to be converted to a RAGAS Dataset
    """
    results = []

    for i, item in enumerate(eval_data):
        question = item['question']
        ground_truth = item['ground_truth']

        print(f"[{i+1}/{len(eval_data)}] {question[:60]}...")

         # query() returns (answer, sources, context_texts)
        # sources = file paths (for display/debugging)
        # context_texts = raw chunk text (required by RAGAS metrics)
        answer, _, contexts = query(question)

        results.append({
            "question":question,
            "answer":answer,
            "ground_truth":ground_truth,
            "contexts":contexts # list of strings, not filenames
        })
    
    return results

def build_ragas_dataset(results:list[dict])-> Dataset:
    """
        Convert the list of RAG results into a HuggingFace Dataset object,
        which is the format RAGAS's evaluate() function expects.
    """
    return Dataset.from_list(results)

# initialize the judge LLM once at module level : reused across all metric
# evaluations to avoid re-instantiating on every call.
# max_tokens=2000 is set higher than RAGAS's default because faithfulness
# evaluation breaks long answers into individual claims and verifies each,
# this process can exceed the default token limit for detailed answers,
# causing IncompleteOutputException and silent NaN scores
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", max_tokens=2000)
)

def run_evaluations(dataset:Dataset):
    """
    Score the RAG results using three RAGAS metrics:
 
    - faithfulness: are the claims in the answer supported by retrieved context?
    - answer_relevancy: does the answer actually address the question?
      (penalizes off-topic answers and unnecessary refusals)
    - context_precision: are the retrieved chunks actually relevant to the question?
 
    Note: answer_relevancy requires an embedding model to compare the answer's
    semantic content to the question. Without explicitly passing embeddings here,
    RAGAS defaults to an OpenAIEmbeddings wrapper that has a version mismatch
    with newer langchain releases, always pass embeddings explicitly.
 
    Returns:
        RAGAS EvaluationResult object : call .to_pandas() for per-question scores
    """
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
    # eval_data = eval_data[:3]  # uncomment to test on a subset first

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