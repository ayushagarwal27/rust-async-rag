from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve .env relative to this file (backend/.env),
# regardless of where uvicorn / the CLI is invoked from.
_ENV_FILE = Path(__file__).parent / ".env"

# Populate os.environ so third-party libraries (langchain_pinecone, openai SDK,
# etc.) that read env vars directly — rather than from our settings object — also
# pick up the values.
load_dotenv(str(_ENV_FILE), override=True)


class Settings(BaseSettings):
    """
    Central configuration loaded from environment variables / .env file.

    All modules import `settings` from here instead of calling os.getenv()
    directly. pydantic_settings validates types at startup and raises a clear
    error if a required variable is missing.
    """

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    # OpenAI
    openai_api_key: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "async-rust-docs"

    # Upstash Redis (REST API — not a socket URL)
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str

    # MongoDB
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db: str = "rustrag"


settings = Settings()
