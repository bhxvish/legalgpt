"""The parsed IPC corpus as a committed JSONL file, so a clean checkout can build the index without
the source PDF (bare-act text is public domain in India: Copyright Act s.52(1)(q))."""

import json
import logging
from pathlib import Path

from app.core.embeddings import EmbeddingService
from app.core.models import Chunk
from app.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, **c.metadata(), "text": c.text}, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def ensure_index(store: VectorStore, embedder: EmbeddingService, chunks_path: Path) -> int:
    """Embed and upsert the committed chunks if the collection is empty. Returns chunks added."""
    if store.count() > 0:
        return 0
    if not chunks_path.is_file():
        logger.warning("vector index is empty and %s is missing; legal mode will refuse every question", chunks_path)
        return 0
    chunks = load_chunks(chunks_path)
    logger.info("vector index is empty: embedding %d chunks from %s (one-off, a minute or two on CPU)",
                len(chunks), chunks_path)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    logger.info("index ready: %d chunks", store.count())
    return len(chunks)
