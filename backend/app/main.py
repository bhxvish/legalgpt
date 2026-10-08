"""FastAPI app: wires Modules 0-5 together. Every service is lazy, so building the app does not load
models, open ChromaDB or call Groq; the embedding model warms up in the background at startup."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.annotation.collector import JudgmentCollector
from app.annotation.router import AnnotationRouter
from app.annotation.store import AnnotationStore
from app.config import Settings, get_settings
from app.core.chat_router import ChatRouter
from app.core.corpus import ensure_index
from app.core.embeddings import EmbeddingService
from app.core.llm_client import AdapterClient, GroqClient, LLMClient, LLMClientError, latest_adapter
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.explainability.attribution_analyzer import AttributionAnalyzer
from app.explainability.confidence_scorer import ConfidenceScorer
from app.explainability.explanation_builder import ExplanationBuilder
from app.labeling_assistant.intake import JudgmentIntake
from app.labeling_assistant.review_queue import HumanReviewQueue
from app.labeling_assistant.router import ReviewRouter
from app.verification.router import VerifyRouter

if TYPE_CHECKING:
    from app.verification.verification_service import VerificationService

logger = logging.getLogger(__name__)

# uvicorn only configures its own loggers; without this, app INFO messages (e.g. "index ready" after
# the first-start index build) are dropped. Scoped to `app` so library request logs stay quiet.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    _app_logger.addHandler(_handler)
    _app_logger.setLevel(logging.INFO)
    _app_logger.propagate = False


class HealthResponse(BaseModel):
    status: str


def build_llm_client(settings: Settings) -> LLMClient:
    """The default chat model: LLM_BACKEND=groq (hosted) or adapter (local base model + LoRA)."""
    if settings.llm_backend == "adapter":
        path = settings.adapter_path or latest_adapter(settings.lora_dir)
        if path is None:
            raise LLMClientError(f"LLM_BACKEND=adapter but no adapter in {settings.lora_dir}; train one first")
        return AdapterClient(path)
    return GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)


def build_models(settings: Settings) -> tuple[dict[str, LLMClient], dict[str, str], str]:
    """Both chat models the base-vs-tuned toggle offers: (clients, notes, default id)."""
    models: dict[str, LLMClient] = {"groq": GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)}
    notes = {"groq": "hosted base model"}
    adapter = settings.adapter_path or latest_adapter(settings.lora_dir)
    if adapter is not None:
        models["adapter"] = AdapterClient(adapter)
        notes["adapter"] = "LoRA-tuned Qwen2.5-1.5B, runs on this machine (GPU if available, otherwise slow on CPU)"
    else:
        notes["adapter"] = f"no LoRA adapter found in {settings.lora_dir} (train one with backend/scripts/train_lora.py)"
    default = settings.llm_backend if settings.llm_backend in models else "groq"
    if settings.llm_backend == "adapter" and "adapter" not in models:
        raise LLMClientError(notes["adapter"])
    return models, notes, default


def build_verification_service(settings: Settings) -> "VerificationService | None":
    """Fact extraction always uses the hosted model: it must return reliable JSON for every
    predicate, which the small local adapter cannot. None if SWI-Prolog is unavailable."""
    try:
        from app.verification.fact_extractor import FactExtractor
        from app.verification.prolog_engine import PrologEngine
        from app.verification.verification_service import VerificationService

        extractor = FactExtractor(GroqClient(settings.groq_api_key, settings.groq_model, max_tokens=4096, temperature=0.0))
        return VerificationService(extractor, PrologEngine(settings.prolog_rules_dir))
    except Exception:  # pyswip missing, SWI-Prolog not installed, or broken rules
        logger.exception("verification disabled: could not start the Prolog engine")
        return None


def build_chat_router(settings: Settings, verifier: "VerificationService | None" = None) -> ChatRouter:
    embedder = EmbeddingService(settings.embedding_model)
    retriever = Retriever(
        embedder=embedder,
        store=VectorStore(settings.chroma_persist_dir, settings.chroma_collection),
        min_similarity=settings.retrieval_min_similarity,
    )
    explainer = ExplanationBuilder(
        AttributionAnalyzer(embedder, settings.explain_support_threshold),
        ConfidenceScorer(high=settings.explain_band_high, medium=settings.explain_band_medium),
    )
    models, notes, default = build_models(settings)
    return ChatRouter(retriever, models[default], explainer=explainer, models=models, default_model=default,
                      model_notes=notes, verifier=verifier)


def build_annotation_router(settings: Settings) -> AnnotationRouter:
    return AnnotationRouter(JudgmentCollector(settings.raw_judgments_dir), AnnotationStore(settings.annotation_store_dir))


def build_review_router(settings: Settings) -> ReviewRouter:
    store = AnnotationStore(settings.annotation_store_dir)
    queue = HumanReviewQueue(store, settings.review_max_audit_error)
    collector = JudgmentCollector(settings.raw_judgments_dir)
    # Uploaded judgments are labelled by the newest InLegalBERT checkpoint (loaded on first upload).
    intake = JudgmentIntake(collector, store, queue, settings.rrl_model_dir,
                            tau_conf=settings.review_tau_conf, audit_rate=settings.review_audit_rate)
    return ReviewRouter(queue, collector, intake)


def create_app(chat_router: ChatRouter | None = None, annotation_router: AnnotationRouter | None = None) -> FastAPI:
    settings = get_settings()
    verifier = build_verification_service(settings)
    chat_router = chat_router or build_chat_router(settings, verifier)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the embedding model in the background so the first question doesn't pay for it, and on
        # a fresh checkout (empty index) build the index from the committed chunks. /health stays
        # responsive meanwhile; a chat request arriving early simply waits for the model.
        embedder = getattr(chat_router.retriever, "embedder", None)
        store = getattr(chat_router.retriever, "store", None)
        if isinstance(embedder, EmbeddingService):

            def warmup() -> None:
                _ = embedder.model
                if isinstance(store, VectorStore):
                    try:
                        ensure_index(store, embedder, settings.corpus_chunks_path)
                    except Exception:
                        logger.exception("could not build the vector index")

            threading.Thread(target=warmup, name="embedding-warmup", daemon=True).start()
        yield

    app = FastAPI(title="LegalGPT", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    app.include_router(chat_router.router)
    app.include_router((annotation_router or build_annotation_router(settings)).router)
    app.include_router(build_review_router(settings).router)
    if verifier is not None:
        app.include_router(VerifyRouter(verifier).router)
    return app


app = create_app()
