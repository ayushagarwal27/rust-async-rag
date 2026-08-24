from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import rag, chat

app = FastAPI(title="Async Rust RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.rustler.in"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API routes ─────────────────────────────────────────────────────────────────

app.include_router(rag.router,  prefix="/api")
app.include_router(chat.router, prefix="/api")

# ── React SPA (served from frontend/dist/ after `npm run build`) ───────────────
# backend/api/app.py → parent.parent.parent = project root

_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str = ""):
    """Serve index.html for all non-API routes so the React router works."""
    index = _DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"detail": "Frontend not built. Run: cd frontend && npm run build"}
