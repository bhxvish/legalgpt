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
    raw_judgments_dir: Path = Field(default=REPO_ROOT / "data" / "raw_judgments", alias="RAW_JUDGMENTS_DIR")
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
    # Parsed IPC chunks (committed); embedded into an empty index at startup.
    corpus_chunks_path: Path = Field(default=REPO_ROOT / "data" / "corpus" / "ipc_chunks.jsonl", alias="CORPUS_CHUNKS_PATH")
    # Cosine-similarity floor below which retrieved chunks are discarded (all discarded -> refusal).
    retrieval_min_similarity: float = Field(default=0.55, alias="RETRIEVAL_MIN_SIMILARITY")
    # floor for questions below RETRIEVAL_MIN_SIMILARITY that the scope check says are about an
    # IPC offence (everyday fact patterns; app/core/scope.py)
    retrieval_scope_floor: float = Field(default=0.40, alias="RETRIEVAL_SCOPE_FLOOR")
    # Everyday fact patterns: the chat's LLM names the likely IPC offences, and their sections are
    # retrieved (app/core/query_rewriter.py). One extra short LLM call per new question.
    query_rewrite: bool = Field(default=True, alias="QUERY_REWRITE")
    # scope-check probability a question needs before a rewrite may steer retrieval
    rewrite_gate: float = Field(default=0.5, alias="REWRITE_GATE")

    # Module 2 — BERT-assisted annotation
    rrl_model_dir: Path = Field(default=REPO_ROOT / "data" / "models" / "rrl", alias="RRL_MODEL_DIR")
    review_tau_conf: float = Field(default=0.7, alias="REVIEW_TAU_CONF")
    review_audit_rate: float = Field(default=0.1, alias="REVIEW_AUDIT_RATE")
    review_max_audit_error: float = Field(default=0.2, alias="REVIEW_MAX_AUDIT_ERROR")

    # Module 3 — LoRA fine-tuning
    # Which LLMClient answers chat requests: "groq" (hosted, GROQ_MODEL) or "adapter" (local
    # base model + LoRA adapter; run the backend from .venv-gpu for GPU inference).
    llm_backend: str = Field(default="groq", alias="LLM_BACKEND", pattern="^(groq|adapter)$")
    lora_dir: Path = Field(default=REPO_ROOT / "data" / "models" / "lora", alias="LORA_DIR")
    adapter_path: Path | None = Field(default=None, alias="ADAPTER_PATH")  # default: newest under LORA_DIR
    # More chat models for the model switch, comma-separated <provider>:<model> (app/core/openai_compat.py),
    # e.g. "gemini:gemini-3.5-flash-lite,openai:gpt-4o-mini". Each needs its key (GEMINI_API_KEY, OPENAI_API_KEY ...).
    extra_models: str = Field(default="", alias="EXTRA_MODELS")
    # A larger local model for harder questions ("Local heavy" in the model switch), run through
    # llama.cpp: the base model as GGUF (e.g. Qwen2.5-7B-Instruct Q4_K_M) and the LoRA adapter
    # converted to GGUF (backend/scripts/colab/README.md). Layers that fit go on the GPU.
    heavy_model_gguf: Path | None = Field(default=None, alias="HEAVY_MODEL_GGUF")
    heavy_lora_gguf: Path | None = Field(default=None, alias="HEAVY_LORA_GGUF")
    heavy_gpu_layers: int = Field(default=16, alias="HEAVY_GPU_LAYERS")
    # Local models also do their own query rewrite, so nothing is sent to a hosted model.
    offline_mode: bool = Field(default=False, alias="OFFLINE_MODE")
    # How the local (LoRA) model is loaded: auto | float32 | bfloat16 | 4bit. auto = bfloat16 on a GPU
    # (4-bit if it does not fit), float32 on a CPU with >= 12 GB RAM, else bfloat16.
    local_llm_dtype: str = Field(default="auto", alias="LOCAL_LLM_DTYPE", pattern="^(auto|float32|bfloat16|float16|4bit)$")

    # Module 4 — explainability. Bands describe how well the evidence matched the question,
    # NOT whether the answer is correct. Defaults calibrated on this corpus (PLANNING.md, Phase 5).
    explain_band_high: float = Field(default=0.70, alias="EXPLAIN_BAND_HIGH")
    explain_band_medium: float = Field(default=0.60, alias="EXPLAIN_BAND_MEDIUM")
    explain_support_threshold: float = Field(default=0.55, alias="EXPLAIN_SUPPORT_THRESHOLD")

    # Comma-separated list of origins allowed by CORS (Vite dev server by default).
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    # ZKML (app/zk/): circuits + proving keys + signing key (git-ignored), and the published
    # manifest / verification keys (committed). Build them with scripts/zk_setup.py.
    zk_dir: Path = Field(default=REPO_ROOT / "data" / "zk", alias="ZK_DIR")
    zk_published_dir: Path = Field(default=REPO_ROOT / "docs" / "zk", alias="ZK_PUBLISHED_DIR")
    attest_answers: bool = Field(default=True, alias="ATTEST_ANSWERS")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
