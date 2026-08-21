import asyncio
from typing import AsyncIterator, Literal, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from backend.config import settings
from .vectorstore import get_vectorstore

_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)

_reranker = CrossEncoder("BAAI/bge-reranker-base")

_vectorstore = get_vectorstore()

_HYDE_SYSTEM = SystemMessage(
    content=(
        "You are a technical documentation writer for async Rust. "
        "Given a question, write a short passage (2-3 sentences) from a documentation page "
        "that would directly answer it. Write only the passage, no preamble or explanation."
    )
)

_SYSTEM = SystemMessage(
    content=(
        "You are an expert in async Rust debugging. "
        "Answer questions using the provided documentation excerpts. You may "
        "synthesize and connect information across multiple excerpts, even if no "
        "single excerpt uses the exact same wording as the question. Only say the "
        "excerpts don't cover this if the underlying concept is genuinely absent, "
        "not just differently phrased."
    )
)

_DIRECT_SYSTEM = SystemMessage(
    content=(
        "You are a helpful assistant specialising in async Rust. "
        "Answer conversationally and concisely."
    )
)

class _RouteDecision(BaseModel):
    route: Literal["rag", "direct", "off_topic"]

_router_llm = _llm.with_structured_output(_RouteDecision)

_ROUTER_SYSTEM = SystemMessage(
    content=(
        "You are a topic classifier. Classify the user message into exactly one route.\n\n"
        "rag       — questions about async Rust, tokio, tracing, futures, pinning, channels, "
        "            deadlocks, executors, task scheduling, Rust syntax, ownership, lifetimes, "
        "            or any Rust / systems-programming concept.\n"
        "direct    — greetings (hi, hello, hey), thanks, small talk, or meta questions about you.\n"
        "off_topic — everything else: geography, history, cooking, other programming languages, "
        "            math, general knowledge, science unrelated to Rust, etc."
    )
)

_OFF_TOPIC_RESPONSE = (
    "I'm specialised in async Rust — tokio, tracing, futures, and related topics. "
    "I can't help with that, but feel free to ask anything about async Rust!"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _classify(question: str) -> Literal["rag", "direct", "off_topic"]:
    """Classify the question via structured output — guaranteed to be a valid route."""
    decision: _RouteDecision = _router_llm.invoke(
        [_ROUTER_SYSTEM, HumanMessage(content=question)]
    )
    return decision.route


def _generate_hypothesis(question: str) -> str:
    """HyDE: generate a hypothetical documentation passage to improve retrieval."""
    return _llm.invoke([_HYDE_SYSTEM, HumanMessage(content=question)]).content


def _history_messages(history: list[dict]) -> list:
    """Convert history dicts to LangChain message objects."""
    msgs = []
    for turn in history:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        else:
            msgs.append(AIMessage(content=turn["content"]))
    return msgs


def _build_rag_messages(question: str, context: str, history: list[dict]) -> list:
    """system → history → RAG-augmented question."""
    return [
        _SYSTEM,
        *_history_messages(history),
        HumanMessage(content=f"Context from documentation:\n{context}\n\nQuestion: {question}"),
    ]


def _build_direct_messages(question: str, history: list[dict]) -> list:
    """system → history → bare question (no retrieval context)."""
    return [_DIRECT_SYSTEM, *_history_messages(history), HumanMessage(content=question)]


def _get_context(question: str) -> tuple[list[Document], list[str], list[str]]:
    """
    Synchronous HyDE → retrieve (top-10) → rerank (top-3) pipeline.
    Used by chat_stream via asyncio.to_thread.
    """
    hypothesis = _generate_hypothesis(question)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 10})
    candidates = retriever.invoke(hypothesis)

    pairs = [[question, doc.page_content] for doc in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[:5]]

    sources = [doc.metadata.get("source", "") for doc in top_docs]
    context_texts = [doc.page_content for doc in top_docs]
    return top_docs, sources, context_texts


# ── LangGraph state ────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    history: list[dict]
    candidates: NotRequired[list[Document]]   # top-10 from dense retrieval
    top_docs: NotRequired[list[Document]]     # top-3 after reranking
    answer: str
    sources: list[str]
    context_texts: list[str]


# ── Graph nodes ────────────────────────────────────────────────────────────────

def _route(state: RAGState) -> Literal["retrieve", "direct_generate", "off_topic_generate"]:
    """
    Router node: classify the question and return the next node name.
    Greetings skip retrieval; off-topic messages are deflected.
    """
    decision = _classify(state["question"])
    if decision == "direct":
        return "direct_generate"
    if decision == "off_topic":
        return "off_topic_generate"
    return "retrieve"


def _retrieve(state: RAGState) -> dict:
    """HyDE + dense retrieval: fetch top-10 candidates from Pinecone."""
    hypothesis = _generate_hypothesis(state["question"])
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 10})
    return {"candidates": retriever.invoke(hypothesis)}


def _rerank(state: RAGState) -> dict:
    """Cross-encoder reranking: keep top-3 from the candidates."""
    pairs = [[state["question"], doc.page_content] for doc in state["candidates"]]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(state["candidates"], scores), key=lambda x: x[1], reverse=True)
    return {"top_docs": [doc for doc, _ in ranked[:5]]}


def _generate(state: RAGState) -> dict:
    """Grounded generation using retrieved context."""
    context = "\n\n---\n\n".join(doc.page_content for doc in state["top_docs"])
    messages = _build_rag_messages(state["question"], context, state.get("history", []))
    answer = _llm.invoke(messages).content
    sources = [doc.metadata.get("source", "") for doc in state["top_docs"]]
    context_texts = [doc.page_content for doc in state["top_docs"]]
    return {"answer": answer, "sources": sources, "context_texts": context_texts}


def _direct_generate(state: RAGState) -> dict:
    """Direct LLM response — no retrieval, used for greetings and small talk."""
    messages = _build_direct_messages(state["question"], state.get("history", []))
    answer = _llm.invoke(messages).content
    return {"answer": answer, "sources": [], "context_texts": []}


def _off_topic_generate(state: RAGState) -> dict:
    """Guardrail: deflect questions outside async Rust without hitting the LLM."""
    return {"answer": _OFF_TOPIC_RESPONSE, "sources": [], "context_texts": []}


# ── Build the LangGraph ────────────────────────────────────────────────────────

_builder = StateGraph(RAGState)
_builder.add_node("retrieve",          _retrieve)
_builder.add_node("rerank",            _rerank)
_builder.add_node("generate",          _generate)
_builder.add_node("direct_generate",   _direct_generate)
_builder.add_node("off_topic_generate", _off_topic_generate)

_builder.add_conditional_edges(START, _route, {
    "retrieve":          "retrieve",
    "direct_generate":   "direct_generate",
    "off_topic_generate": "off_topic_generate",
})
_builder.add_edge("retrieve",          "rerank")
_builder.add_edge("rerank",            "generate")
_builder.add_edge("generate",          END)
_builder.add_edge("direct_generate",   END)
_builder.add_edge("off_topic_generate", END)

rag_graph = _builder.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

def query(question: str, n_results: int = 3) -> tuple[str, list[str], list[str]]:
    """Single-turn entry point (backward-compatible)."""
    return chat(question, history=[])


def chat(message: str, history: list[dict] | None = None) -> tuple[str, list[str], list[str]]:
    """Multi-turn agentic RAG: routes to retrieval or direct reply as needed."""
    result = rag_graph.invoke({"question": message, "history": history or []})
    return result["answer"], result["sources"], result["context_texts"]


async def chat_stream(
    message: str, history: list[dict] | None = None
) -> AsyncIterator[tuple[str | None, dict | None]]:
    """
    Streaming agentic RAG pipeline.

    1. Classify the message in a thread pool (one LLM call).
    2a. If 'direct': stream a bare LLM reply — no retrieval.
    2b. If 'rag':    retrieve + rerank in a thread pool, then stream the answer.

    Yields (token, None) per token, then (None, meta) as the final event.
    """
    route = await asyncio.to_thread(_classify, message)

    if route == "off_topic":
        yield _OFF_TOPIC_RESPONSE, None
        updated_history = (history or []) + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": _OFF_TOPIC_RESPONSE},
        ]
        yield None, {"sources": [], "history": updated_history}
        return

    if route == "direct":
        messages = _build_direct_messages(message, history or [])
        sources: list[str] = []
    else:
        top_docs, sources, _ = await asyncio.to_thread(_get_context, message)
        messages = _build_rag_messages(
            message,
            "\n\n---\n\n".join(doc.page_content for doc in top_docs),
            history or [],
        )

    full_answer = ""
    async for chunk in _llm.astream(messages):
        token = chunk.content
        if token:
            full_answer += token
            yield token, None

    updated_history = (history or []) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": full_answer},
    ]
    yield None, {"sources": list(set(sources)), "history": updated_history}


if __name__ == "__main__":
    # usage: uv run python -m backend.rag.query
    for q in ["hello!", "what is the difference between spawn and spawn_blocking?"]:
        print(f"\nQ: {q}")
        answer, sources, _ = query(q)
        print(f"A: {answer}")
        if sources:
            print(f"Sources: {sources}")
