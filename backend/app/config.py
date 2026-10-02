"""Single source of environment-driven settings for the LegalGPT backend.

Values come from environment variables or the repo-root `.env` file. Every
variable read here must be documented in `.env.example`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `CHROMA_PERSIST_DIR=` in .env means "use the default"
        extra="ignore",
    )

    # Secrets — empty by default; features that need them fail loudly at use time.
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # Storage paths — defaults resolve against the repo root so they are the same
    # whether the backend is launched from the repo root or from backend/.
    chroma_persist_dir: Path = Field(default=REPO_ROOT / "data" / "chroma_db", alias="CHROMA_PERSIST_DIR")
    annotation_store_dir: Path = Field(
        default=REPO_ROOT / "data" / "annotation_store", alias="ANNOTATION_STORE_DIR"
    )
    prolog_rules_dir: Path = Field(
        default=BACKEND_DIR / "app" / "verification" / "rules", alias="PROLOG_RULES_DIR"
    )

    # Module 0 — retrieval & generation
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chroma_collection: str = Field(default="ipc_bare_act", alias="CHROMA_COLLECTION")
    # Cosine-similarity floor below which retrieved chunks are discarded (all discarded -> refusal).
    retrieval_min_similarity: float = Field(default=0.55, alias="RETRIEVAL_MIN_SIMILARITY")

    # Comma-separated list of origins allowed by CORS (Vite dev server by default).
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
