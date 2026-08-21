from fastapi import APIRouter
from pydantic import BaseModel
from backend.rag.query import query

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/rag/query")
def ask(req: QueryRequest):
    """Single-turn RAG query (legacy endpoint, kept for backward compatibility)."""
    answer, sources, _ = query(req.question)
    return {"answer": answer, "sources": list(set(sources))}
