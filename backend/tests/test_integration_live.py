"""Live checks against Groq and the real ingested index.

Skipped unless GROQ_API_KEY is set. Run only these with:  pytest -m integration
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.llm_client import GroqClient
from app.core.models import Message

pytestmark = pytest.mark.integration

settings = get_settings()
needs_groq = pytest.mark.skipif(not settings.groq_api_key, reason="GROQ_API_KEY not set")


def _events(body: str) -> list[tuple[str, object]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@needs_groq
def test_groq_streams_a_reply() -> None:
    llm = GroqClient(settings.groq_api_key, settings.groq_model)
    text = "".join(llm.stream([Message("user", "Reply with exactly the word: pong")]))
    assert "pong" in text.lower()


@pytest.fixture(scope="module")
def live_client() -> TestClient:
    from app.core.vector_store import VectorStore

    if VectorStore(settings.chroma_persist_dir, settings.chroma_collection).count() == 0:
        pytest.skip("no ingested corpus; run backend/scripts/ingest_corpus.py first")
    from app.main import create_app

    return TestClient(create_app())


@needs_groq
def test_live_legal_answer_cites_sources(live_client: TestClient) -> None:
    resp = live_client.post("/api/chat", json={"question": "What is the punishment for causing death by negligence?"})
    events = _events(resp.text)
    meta = events[0][1]
    assert meta["refused"] is False  # type: ignore[index]
    answer = "".join(d["text"] for e, d in events if e == "token")  # type: ignore[index]
    assert "[1]" in answer or "[2]" in answer, answer


@needs_groq
def test_live_out_of_corpus_question_is_refused(live_client: TestClient) -> None:
    resp = live_client.post("/api/chat", json={"question": "What is the best recipe for chocolate cake?"})
    assert _events(resp.text)[0][1]["refused"] is True  # type: ignore[index]
