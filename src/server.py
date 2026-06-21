from fastapi import FastAPI
from pydantic import BaseModel
from query import query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rustler.in"],
    allow_methods=["POST"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question:str

@app.post("/api/rag/query")
def ask(req:QueryRequest):
    answer, sources, _ = query(req.question)
    return {"answer":answer, "sources":list(set(sources))}
