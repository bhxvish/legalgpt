from pathlib import Path

import chromadb

from app.config import REPO_ROOT
from app.core.corpus import ensure_index, load_chunks, write_chunks
from app.core.models import Chunk
from app.core.vector_store import VectorStore
from tests.conftest import FakeEmbedder

CHUNKS = [
    Chunk("ipc-304a-0", "IPC", "Whoever causes the death of any person by doing any rash or negligent act...",
          "IPC > Chapter XVI > Section 304A", "XVI", "304A", "Causing death by negligence"),
    Chunk("ipc-379-0", "IPC", "Whoever commits theft shall be punished...", "IPC > Chapter XVII > Section 379",
          "XVII", "379", "Punishment for theft"),
]


def test_chunks_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "corpus" / "chunks.jsonl"
    write_chunks(CHUNKS, path)
    assert load_chunks(path) == CHUNKS


def test_ensure_index_fills_an_empty_store_once(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_chunks(CHUNKS, path)
    store, embedder = VectorStore(client=chromadb.EphemeralClient(), collection_name="corpus-test"), FakeEmbedder()
    assert ensure_index(store, embedder, path) == 2 and store.count() == 2
    assert ensure_index(store, embedder, path) == 0 and embedder.calls == 1  # already built: no re-embedding
    assert store.get_by_path("IPC > Chapter XVI > Section 304A")[0].title == "Causing death by negligence"


def test_ensure_index_without_chunks_file_leaves_store_empty(tmp_path: Path) -> None:
    store = VectorStore(client=chromadb.EphemeralClient(), collection_name="corpus-missing")
    assert ensure_index(store, FakeEmbedder(), tmp_path / "missing.jsonl") == 0 and store.count() == 0


def test_committed_corpus_is_the_full_ipc() -> None:
    chunks = load_chunks(REPO_ROOT / "data" / "corpus" / "ipc_chunks.jsonl")
    sections = {c.section for c in chunks}
    assert len(chunks) == 685 and {"302", "304A", "304B", "379", "498A"} <= sections


def test_concurrent_first_open_of_a_persistent_store(tmp_path: Path) -> None:
    """The startup index build and the first chat request open the store at the same time."""
    from concurrent.futures import ThreadPoolExecutor

    store = VectorStore(tmp_path / "chroma", "concurrent_open")
    with ThreadPoolExecutor(8) as pool:
        counts = list(pool.map(lambda _: store.count(), range(16)))
    assert counts == [0] * 16
