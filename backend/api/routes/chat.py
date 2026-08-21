import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.rag.query import chat, chat_stream
from backend.rag.session import load_history, save_history, SESSION_TTL

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("/chat")
async def chatbot(req: ChatRequest):
    """
    Multi-turn chatbot endpoint.

    History is loaded from Redis (fast) or MongoDB (fallback) by session_id,
    so clients only send the current message — no history management needed.

    Response: {"answer": "...", "sources": [...], "session_id": "..."}
    """
    history = await load_history(req.session_id)
    answer, sources, _ = chat(req.message, history)

    updated_history = history + [
        {"role": "user",      "content": req.message},
        {"role": "assistant", "content": answer},
    ]
    await save_history(req.session_id, updated_history)

    return {"answer": answer, "sources": list(set(sources)), "session_id": req.session_id}


@router.post("/chat/stream")
async def chatbot_stream(req: ChatRequest):
    """
    Streaming multi-turn chatbot (Server-Sent Events).

    SSE events:
        data: {"token": "..."}                                       — per LLM token
        data: {"done": true, "sources": [...], "session_id": "..."}  — final
    """
    history = await load_history(req.session_id)

    async def event_stream():
        updated_history = None
        async for token, meta in chat_stream(req.message, history):
            if token is not None:
                yield f"data: {json.dumps({'token': token})}\n\n"
            else:
                updated_history = meta["history"]
                yield f"data: {json.dumps({'done': True, 'sources': meta['sources'], 'session_id': req.session_id})}\n\n"

        if updated_history is not None:
            await save_history(req.session_id, updated_history)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    """
    Return the conversation history for a session (Redis → MongoDB fallback).

    Response: {"session_id": "...", "messages": [...], "ttl_seconds": 86400}
    """
    history = await load_history(session_id)
    return {"session_id": session_id, "messages": history, "ttl_seconds": SESSION_TTL}
