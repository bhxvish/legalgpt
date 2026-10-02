import hashlib
import re
import uuid
from collections.abc import Iterator

import chromadb
import numpy as np
import pytest

from app.core.llm_client import LLMClient
from app.core.models import Message
from app.core.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: texts sharing words get positive
    cosine similarity, texts with disjoint vocabularies get ~0. No model download."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]{3,}", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class StubLLM(LLMClient):
    """Records every call; streams a canned reply in a few fragments."""

    model_id = "stub-llm"

    def __init__(self, reply: str = "Death by negligence is covered by Section 304A [1].", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("stub failure")
        return self.reply

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("stub failure")
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def memory_store() -> VectorStore:
    """In-memory ChromaDB collection, unique per test."""
    return VectorStore(client=chromadb.EphemeralClient(), collection_name=f"test-{uuid.uuid4().hex[:12]}")
