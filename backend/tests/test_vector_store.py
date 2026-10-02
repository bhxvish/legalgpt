import uuid

import chromadb

from app.core.models import Chunk
from app.core.vector_store import VectorStore
from tests.conftest import FakeEmbedder


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(chunk_id=f"c{i}", doc_id="IPC", text=t, citation_path=f"IPC > Section {i}") for i, t in enumerate(texts)]


def test_upsert_query_and_get_by_path(fake_embedder: FakeEmbedder, memory_store: VectorStore) -> None:
    chunks = _chunks("theft of movable property", "murder punishment death")
    memory_store.upsert(chunks, fake_embedder.embed([c.text for c in chunks]))
    res = memory_store.query(fake_embedder.embed(["movable property theft"])[0], k=2)
    assert res[0].chunk.chunk_id == "c0" and res[0].similarity > res[1].similarity
    assert [c.chunk_id for c in memory_store.get_by_path("IPC > Section 1")] == ["c1"]


def test_query_on_empty_store_returns_empty(fake_embedder: FakeEmbedder, memory_store: VectorStore) -> None:
    assert memory_store.query(fake_embedder.embed(["anything"])[0], k=5) == []


def test_survives_collection_reset_by_another_process(fake_embedder: FakeEmbedder) -> None:
    """The server's store keeps working after the ingest script drops and rebuilds the collection."""
    client, name = chromadb.EphemeralClient(), f"test-{uuid.uuid4().hex[:12]}"
    server, ingest = VectorStore(client=client, collection_name=name), VectorStore(client=client, collection_name=name)
    chunks = _chunks("theft of movable property")
    server.upsert(chunks, fake_embedder.embed([chunks[0].text]))
    assert server.count() == 1
    ingest.reset()
    ingest.upsert(chunks, fake_embedder.embed([chunks[0].text]))
    assert server.count() == 1
    assert server.query(fake_embedder.embed(["theft"])[0], k=1)[0].chunk.chunk_id == "c0"
