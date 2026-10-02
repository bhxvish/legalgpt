"""ChatRouter over SSE, with a stub LLMClient (no Groq calls)."""

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.chat_router import GENERAL_CAUTION, SCOPE_REFUSAL, ChatRouter
from app.core.models import Chunk
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from tests.conftest import FakeEmbedder, StubLLM


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def answer_text(events: list[tuple[str, object]]) -> str:
    return "".join(d["text"] for e, d in events if e == "token")  # type: ignore[index]


class ExplodingRetriever:
    def retrieve(self, question: str) -> list:
        raise AssertionError("retrieval must not run in general mode")


@pytest.fixture
def retriever(fake_embedder: FakeEmbedder, memory_store: VectorStore) -> Retriever:
    chunk = Chunk(
        chunk_id="IPC:304A:x", doc_id="IPC", section="304A", title="Causing death by negligence",
        citation_path="IPC > Chapter XVI > Section 304A",
        text="Section 304A. Causing death by negligence\nWhoever causes the death of any person by doing any rash or negligent act.",
    )
    memory_store.upsert([chunk], fake_embedder.embed([chunk.text]))
    return Retriever(fake_embedder, memory_store, min_similarity=0.2)


def client_for(retriever: object, llm: StubLLM) -> TestClient:
    app = FastAPI()
    app.include_router(ChatRouter(retriever, llm).router)  # type: ignore[arg-type]
    return TestClient(app)


def test_out_of_scope_question_streams_refusal_without_calling_llm(retriever: Retriever) -> None:
    llm = StubLLM()
    resp = client_for(retriever, llm).post("/api/chat", json={"question": "Best mutual funds for tax saving?", "mode": "legal"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert events[0] == ("meta", {"mode": "legal", "model": "stub-llm", "refused": True})
    assert answer_text(events) == SCOPE_REFUSAL
    assert "sources" not in [e for e, _ in events]
    assert events[-1][0] == "done"
    assert llm.calls == []  # refusal must not reach the model


def test_in_scope_question_streams_sources_then_cited_answer(retriever: Retriever) -> None:
    llm = StubLLM()
    resp = client_for(retriever, llm).post(
        "/api/chat", json={"question": "Punishment for death caused by a negligent act?", "mode": "legal"}
    )
    events = parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
    sources = events[1][1]
    assert sources[0]["marker"] == 1 and sources[0]["citation_path"] == "IPC > Chapter XVI > Section 304A"  # type: ignore[index]
    assert "[1]" in answer_text(events)
    # the model saw the numbered source
    assert "[1] (IPC > Chapter XVI > Section 304A)" in llm.calls[0][-1].content


def test_general_mode_bypasses_retrieval_and_appends_caution() -> None:
    llm = StubLLM(reply="Hello there.")
    resp = client_for(ExplodingRetriever(), llm).post("/api/chat", json={"question": "hi", "mode": "general"})
    events = parse_sse(resp.text)
    assert events[0][1]["mode"] == "general"  # type: ignore[index]
    assert answer_text(events) == "Hello there." + GENERAL_CAUTION
    assert len(llm.calls) == 1


def test_llm_failure_becomes_error_event(retriever: Retriever) -> None:
    resp = client_for(retriever, StubLLM(fail=True)).post("/api/chat", json={"question": "hi", "mode": "general"})
    events = parse_sse(resp.text)
    assert ("error", {"message": "stub failure"}) in events
    assert events[-1][0] == "done"


def test_history_is_forwarded_to_the_model(retriever: Retriever) -> None:
    llm = StubLLM()
    client_for(retriever, llm).post(
        "/api/chat",
        json={
            "question": "And for a negligent act causing death?",
            "mode": "legal",
            "history": [{"role": "user", "content": "earlier q"}, {"role": "assistant", "content": "earlier a"}],
        },
    )
    assert [m.content for m in llm.calls[0][1:3]] == ["earlier q", "earlier a"]


def test_rejects_invalid_mode(retriever: Retriever) -> None:
    resp = client_for(retriever, StubLLM()).post("/api/chat", json={"question": "x", "mode": "other"})
    assert resp.status_code == 422


def test_fullwidth_citation_brackets_are_normalized(retriever: Retriever) -> None:
    llm = StubLLM(reply="Two years, or fine, or both 【1】.")
    resp = client_for(retriever, llm).post("/api/chat", json={"question": "Punishment for a negligent act causing death?"})
    assert answer_text(parse_sse(resp.text)) == "Two years, or fine, or both [1]."


class FragmentLLM(StubLLM):
    """Streams exactly the given fragments."""

    def __init__(self, fragments: list[str]) -> None:
        super().__init__()
        self.fragments = fragments

    def stream(self, messages: list) -> Iterator[str]:  # type: ignore[override]
        self.calls.append(messages)
        yield from self.fragments


def test_citation_markers_split_across_fragments_are_canonicalized(retriever: Retriever) -> None:
    dagger, lb, rb = chr(0x2020), chr(0x3010), chr(0x3011)
    llm = FragmentLLM(["Within seven years [", f"1{dagger}L2", "-L4]", f" of marriage {lb}1{rb}.", " Unclosed [note"])
    resp = client_for(retriever, llm).post("/api/chat", json={"question": "Punishment for a negligent act causing death?"})
    assert answer_text(parse_sse(resp.text)) == "Within seven years [1] of marriage [1]. Unclosed [note"
