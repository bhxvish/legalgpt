"""EmbeddingService: sentence-transformers wrapper (all-MiniLM-L6-v2 by default)."""

import threading
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class EmbeddingService:
    """Produces L2-normalized embeddings. The model loads lazily on first use so that
    importing the app (e.g. for /health or unit tests) never downloads or loads it."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: "SentenceTransformer | None" = None
        self._lock = threading.Lock()

    @property
    def model(self) -> "SentenceTransformer":
        with self._lock:  # startup warm-up and the first request may race
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
