"""Central configuration; secrets are supplied at runtime by the UI or environment."""
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.35"))
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    knowledge_base_path: str = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()


def env_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()

def validate_settings() -> None:
    """Fail fast on invalid non-secret configuration values."""
    if settings.temperature < 0 or settings.temperature > 2:
        raise ValueError("TEMPERATURE must be between 0 and 2.")
    if settings.chunk_size < 100:
        raise ValueError("CHUNK_SIZE must be at least 100 characters.")
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
    if settings.retrieval_k < 1:
        raise ValueError("RETRIEVAL_K must be at least 1.")
    if not 0 <= settings.score_threshold <= 1:
        raise ValueError("SCORE_THRESHOLD must be between 0 and 1 for relevance-score retrieval.")
    if not settings.model.strip() or not settings.embedding_model.strip():
        raise ValueError("OPENAI_MODEL and OPENAI_EMBEDDING_MODEL must not be empty.")

if __name__ == "__main__":
    validate_settings()
    print("SupportPearlz configuration loaded (secrets masked):")
    print(f"model={settings.model}")
    print(f"embedding_model={settings.embedding_model}")
    print(f"temperature={settings.temperature}")
    print(f"chunk_size={settings.chunk_size}")
    print(f"chunk_overlap={settings.chunk_overlap}")
    print(f"retrieval_k={settings.retrieval_k}")
    print(f"score_threshold={settings.score_threshold}")
    print(f"vector_store_path={settings.vector_store_path}")
    print(f"knowledge_base_path={settings.knowledge_base_path}")
    print("OPENAI_API_KEY=*** (value not displayed)")
