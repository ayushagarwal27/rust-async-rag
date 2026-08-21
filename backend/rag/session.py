import asyncio
import json
from datetime import datetime, timezone

from upstash_redis.asyncio import Redis
import motor.motor_asyncio
from backend.config import settings

SESSION_TTL = 86_400  # 24 hours

# ── Lazy singletons ────────────────────────────────────────────────────────────

_redis: Redis | None = None
_mongo_db = None


def _get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token,
        )
    return _redis


def _get_mongo():
    global _mongo_db
    if _mongo_db is None:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.mongo_url,
            tlsAllowInvalidCertificates=True,
        )
        _mongo_db = client[settings.mongo_db]
    return _mongo_db


# ── Public helpers ─────────────────────────────────────────────────────────────

async def load_history(session_id: str) -> list[dict]:
    """
    Return conversation history for a session.

    Cache hierarchy:
      1. Upstash Redis (fast, TTL-bounded) — hit on active sessions
      2. MongoDB (persistent) — hit on cache miss / server restart
      3. [] — new session
    """
    r = _get_redis()
    cached = await r.get(f"session:{session_id}")
    if cached:
        return json.loads(cached)

    doc = await _get_mongo().sessions.find_one(
        {"session_id": session_id}, {"_id": 0, "history": 1}
    )
    if doc:
        history = doc.get("history", [])
        await r.set(f"session:{session_id}", json.dumps(history), ex=SESSION_TTL)
        return history

    return []


async def save_history(session_id: str, history: list[dict]) -> None:
    """Persist updated history to both Upstash Redis and MongoDB concurrently."""
    r = _get_redis()
    payload = json.dumps(history)

    await asyncio.gather(
        r.set(f"session:{session_id}", payload, ex=SESSION_TTL),
        _get_mongo().sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "history": history,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        ),
    )
