import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel


def _maybe_load_dotenv() -> None:
    # Local developer convenience. If python-dotenv is not installed, skip.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


class Settings(BaseModel):
    # Backend
    app_name: str = "AI Adaptive Onboarding Engine"
    environment: str = "dev"
    frontend_origin: str = "http://localhost:5173"

    # Persistence (demo uses in-memory store)
    analysis_ttl_seconds: int = 60 * 60  # 1 hour

    # Embeddings
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.6
    top_k_resume_matches: int = 3

    # Optional LLM extraction
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(v)
    except Exception:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _maybe_load_dotenv()

    return Settings(
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        analysis_ttl_seconds=_env_int("ANALYSIS_TTL_SECONDS", 60 * 60),
        embeddings_model_name=os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        similarity_threshold=_env_float("SIMILARITY_THRESHOLD", 0.6),
        top_k_resume_matches=_env_int("TOP_K_RESUME_MATCHES", 3),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

