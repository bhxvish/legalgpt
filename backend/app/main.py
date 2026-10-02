import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.annotation.collector import JudgmentCollector
from app.annotation.router import AnnotationRouter
from app.annotation.store import AnnotationStore
from app.config import Settings, get_settings
from app.labeling_assistant.review_queue import HumanReviewQueue
from app.labeling_assistant.router import ReviewRouter
from app.core.chat_router import ChatRouter
from app.core.embeddings import EmbeddingService
from app.core.llm_client import AdapterClient, GroqClient, LLMClient, LLMClientError, latest_adapter
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore


class HealthResponse(BaseModel):
    status: str


def build_chat_router(settings: Settings) -> ChatRouter:
    """Default wiring. Every service initializes lazily, so building this is cheap and
    does not load the embedding model, open ChromaDB, or contact Groq."""
    retriever = Retriever(
        embedder=EmbeddingService(settings.embedding_model),
        store=VectorStore(settings.chroma_persist_dir, settings.chroma_collection),
        min_similarity=settings.retrieval_min_similarity,
    )
    return ChatRouter(retriever, build_llm_client(settings))


def build_llm_client(settings: Settings) -> LLMClient:
    """LLM_BACKEND switches the chat model without code changes (both implement LLMClient)."""
    if settings.llm_backend == "adapter":
        path = settings.adapter_path or latest_adapter(settings.lora_dir)
        if path is None:
            raise LLMClientError(f"LLM_BACKEND=adapter but no adapter in {settings.lora_dir}; train one first")
        return AdapterClient(path)
    return GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)


def build_annotation_router(settings: Settings) -> AnnotationRouter:
    return AnnotationRouter(JudgmentCollector(settings.raw_judgments_dir), AnnotationStore(settings.annotation_store_dir))


def build_review_router(settings: Settings) -> ReviewRouter:
    queue = HumanReviewQueue(AnnotationStore(settings.annotation_store_dir), settings.review_max_audit_error)
    return ReviewRouter(queue, JudgmentCollector(settings.raw_judgments_dir))


def create_app(chat_router: ChatRouter | None = None, annotation_router: AnnotationRouter | None = None) -> FastAPI:
    settings = get_settings()
    chat_router = chat_router or build_chat_router(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the embedding model in the background so the first question doesn't pay for it.
        # /health stays responsive meanwhile; a chat request arriving early simply waits.
        embedder = getattr(chat_router.retriever, "embedder", None)
        if isinstance(embedder, EmbeddingService):
            threading.Thread(target=lambda: embedder.model, name="embedding-warmup", daemon=True).start()
        yield

    app = FastAPI(title="LegalGPT", version="0.1.0", lifespan=lifespan)

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
    return app


app = create_app()
