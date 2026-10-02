"""VectorStore: thin ChromaDB wrapper using cosine distance."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from app.core.models import Chunk

T = TypeVar("T")


@dataclass
class QueryResult:
    chunk: Chunk
    distance: float  # cosine distance in [0, 2]; similarity = 1 - distance

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


class VectorStore:
    """Persists chunks + embeddings in a ChromaDB collection.

    Pass `client` (e.g. `chromadb.EphemeralClient()`) in tests; otherwise a persistent
    client is opened lazily at `persist_dir` on first use.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "ipc_bare_act",
        client: Any | None = None,
    ) -> None:
        if client is None and persist_dir is None:
            raise ValueError("VectorStore needs either persist_dir or client")
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.collection_name = collection_name
        self._client = client
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            if self._client is None:
                import chromadb

                assert self.persist_dir is not None
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,  # we always supply our own embeddings
            )
        return self._collection

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 1000) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=[e.tolist() for e in embeddings[i : i + batch_size]],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
            )

    def query(self, embedding: np.ndarray, k: int, where: dict[str, Any] | None = None) -> list[QueryResult]:
        if self.count() == 0:
            return []
        res = self._call(
            lambda col: col.query(
                query_embeddings=[np.asarray(embedding).tolist()],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        )
        return [
            QueryResult(self._to_chunk(cid, doc, meta), float(dist))
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_by_path(self, citation_path: str) -> list[Chunk]:
        res = self._call(lambda col: col.get(where={"citation_path": citation_path}, include=["documents", "metadatas"]))
        return [self._to_chunk(cid, doc, meta) for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

    def count(self) -> int:
        return int(self._call(lambda col: col.count()))

    def _call(self, fn: Callable[[Any], T]) -> T:
        """Run `fn` on the collection; if it was dropped and recreated meanwhile (e.g. the ingest
        script ran with --reset while the server was up), reopen it once and retry."""
        from chromadb.errors import NotFoundError

        try:
            return fn(self.collection)
        except NotFoundError:
            self._collection = None
            return fn(self.collection)

    def reset(self) -> None:
        """Drop and recreate the collection (used by the ingest script's --reset)."""
        _ = self.collection  # ensure client exists
        self._client.delete_collection(self.collection_name)
        self._collection = None

    @staticmethod
    def _to_chunk(chunk_id: str, document: str, meta: dict[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            text=document,
            citation_path=str(meta.get("citation_path", "")),
            chapter=str(meta.get("chapter", "")),
            section=str(meta.get("section", "")),
            title=str(meta.get("title", "")),
            role=str(meta.get("role", "Law")),
        )
