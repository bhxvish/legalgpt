"""End to end: a raw question -> retrieval -> generation -> explanation -> (on request) verification.

CI path (always runs): the real parser, chunker, retriever, prompt builder, explanation layer and
verification service, over real IPC text (fixtures/ipc_excerpt.txt). Only the outside world is
stubbed: a bag-of-words embedder, scripted LLMs, and — when SWI-Prolog is not installed — a
Python engine with the same verdict semantics (the real PrologEngine is used when available).

Live path: set LEGALGPT_LIVE_E2E=1 to run the same chain through create_app() with Groq, MiniLM,
the ingested ChromaDB index and SWI-Prolog.
"""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import chromadb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import BACKEND_DIR
from app.core.chat_router import ChatRouter
from app.core.parser import LegalDocumentParser
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.explainability.attribution_analyzer import AttributionAnalyzer
from app.explainability.explanation_builder import ExplanationBuilder
from app.verification.fact_extractor import FactExtractor
from app.verification.prolog_engine import ElementDef, SectionDef
from app.verification.verification_service import VerificationService
from tests.conftest import FakeEmbedder, StubLLM
from tests.test_verification import ScriptedLLM

FIXTURE = Path(__file__).parent / "fixtures" / "ipc_excerpt.txt"
QUESTION = ("A truck driver drove rashly on the highway and caused the death of a pedestrian; he did not intend "
            "or know that death was likely. What is the punishment under Section 304A for causing death by negligence?")
ANSWER = ("Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable "
          "homicide shall be punished with imprisonment of either description for a term which may extend to two "
          "years, or with fine, or with both [1].")
FACTS = json.dumps({
    "caused_death": {"value": "true", "evidence": "caused the death of a pedestrian"},
    "act_type": {"value": "rash", "evidence": "drove rashly on the highway"},
    "intent_to_kill": {"value": "false", "evidence": "he did not intend"},
    "knowledge_likely_death": {"value": "false", "evidence": "or know that death was likely"},
    "drove_vehicle": {"value": "true", "evidence": "A truck driver drove"},
    "on_public_way": {"value": "true", "evidence": "on the highway"},
})


class PyEngine:
    """Stand-in for PrologEngine in CI without SWI-Prolog: same interface and the same verdict
    rule as rules/core.pl (violated > missing > consistent), for s.304A and s.279."""

    SECTIONS = [
        SectionDef("304A", "Causing death by negligence", (
            ElementDef("caused_death", ("true",), "The act caused the death of a person"),
            ElementDef("act_type", ("rash", "negligent"), "The act was rash or negligent"),
            ElementDef("intent_to_kill", ("false",), "No intention to cause death or bodily injury likely to cause death"),
            ElementDef("knowledge_likely_death", ("false",), "No knowledge that the act was likely to cause death"),
        )),
        SectionDef("279", "Rash driving or riding on a public way", (
            ElementDef("drove_vehicle", ("true",), "The accused drove a vehicle or rode"),
            ElementDef("on_public_way", ("true",), "This happened on a public way"),
            ElementDef("act_type", ("rash", "negligent"), "The driving or riding was rash or negligent"),
            ElementDef("endangered_safety", ("true",), "It endangered human life or was likely to cause hurt or injury"),
        )),
    ]

    def __init__(self) -> None:
        self.facts: dict[str, dict[str, str]] = {}

    def sections(self) -> list[SectionDef]:
        return self.SECTIONS

    @contextmanager
    def session(self, case_id: str, clauses: list[str]) -> Iterator["PyEngine"]:
        try:
            for clause in clauses:  # fact('id', pred, value)
                _, pred, value = clause[len("fact("):-1].split(", ")
                self.facts.setdefault(case_id, {})[pred] = value
            yield self
        finally:
            self.facts.pop(case_id, None)

    def element_statuses(self, case_id: str, section: str) -> dict[str, str]:
        facts = self.facts.get(case_id, {})
        sec = next(s for s in self.SECTIONS if s.section == section)
        return {e.predicate: ("missing" if facts.get(e.predicate, "unknown") == "unknown"
                              else "satisfied" if facts[e.predicate] in e.required else "violated") for e in sec.elements}

    def verdict(self, case_id: str, section: str) -> str:
        statuses = self.element_statuses(case_id, section).values()
        return "INCONSISTENT" if "violated" in statuses else "INSUFFICIENT" if "missing" in statuses else "CONSISTENT"


def _engine():
    try:
        from app.verification.prolog_engine import PrologEngine

        return PrologEngine(BACKEND_DIR / "app" / "verification" / "rules")
    except Exception:
        return PyEngine()


ENGINES = ["python"] + (["prolog"] if not isinstance(_engine(), PyEngine) else [])


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _build(engine_kind: str, verifier_available: bool = True):
    """The real chain over a real IPC excerpt, with the outside world stubbed."""
    embedder = FakeEmbedder()
    parser = LegalDocumentParser(doc_id="IPC")
    text = "\n".join(l for l in FIXTURE.read_text(encoding="utf-8").splitlines() if not l.startswith("#"))
    chunks = parser.chunk(parser.parse_text(text))
    store = VectorStore(client=chromadb.EphemeralClient(), collection_name=f"e2e-{engine_kind}-{os.getpid()}-{id(text)}")
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = Retriever(embedder, store, min_similarity=0.2)
    base, tuned = StubLLM(ANSWER), StubLLM("Section 304A applies [1].")
    tuned.model_id = "stub-tuned"
    extractor_llm = ScriptedLLM([FACTS, FACTS])  # two runs: values must agree
    verifier = (VerificationService(FactExtractor(extractor_llm), PyEngine() if engine_kind == "python" else _engine())
                if verifier_available else None)
    router = ChatRouter(retriever, base, explainer=ExplanationBuilder(AttributionAnalyzer(embedder, 0.3)),
                        models={"groq": base, "adapter": tuned}, default_model="groq", verifier=verifier)
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app), base, tuned, extractor_llm


@pytest.mark.parametrize("engine_kind", ENGINES)
def test_full_chain_question_to_answer_explanation_and_verification(engine_kind: str) -> None:
    client, base, _, extractor_llm = _build(engine_kind)
    events = _events(client.post("/api/chat", json={"question": QUESTION, "mode": "legal", "verify_section": "304A"}).text)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
    assert names.index("explanation") < names.index("verification") < names.index("done")
    data = {e: d for e, d in events if e != "token"}

    # retrieval over the real IPC excerpt found s.304A (the stub embedder may rank its state amendment higher)
    assert "IPC > Chapter XVI > Section 304A" in [s["citation_path"] for s in data["sources"]]
    # generation: the answer cites the retrieved source
    answer = "".join(d["text"] for e, d in events if e == "token")
    assert "[1]" in answer and len(base.calls) == 1
    # explanation: relevance, band, resolved citation, supported sentence
    ex = data["explanation"]
    assert ex["relevance"] is not None and ex["band"] in ("High", "Medium", "Low")
    assert ex["citation_check"]["resolved"] == [1] and ex["citation_check"]["unresolved"] == []
    assert ex["support_ratio"] == 1.0
    # verification: facts from the question checked against s.304A
    ver = data["verification"]
    assert ver["cited_section"] == "304A" and ver["status"] == "CONSISTENT"
    assert {e["predicate"]: e["status"] for e in ver["elements"]} == {
        "caused_death": "satisfied", "act_type": "satisfied", "intent_to_kill": "satisfied", "knowledge_likely_death": "satisfied"}
    assert len(extractor_llm.calls) == 2  # two-run agreement


def test_model_toggle_switches_the_answering_client() -> None:
    client, base, tuned, _ = _build("python")
    meta = dict(_events(client.post("/api/chat", json={"question": QUESTION, "model": "adapter"}).text))["meta"]
    assert meta["model"] == "stub-tuned" and len(tuned.calls) == 1 and base.calls == []
    meta = dict(_events(client.post("/api/chat", json={"question": QUESTION}).text))["meta"]
    assert meta["model"] == "stub-llm" and len(base.calls) == 1  # default
    models = client.get("/api/models").json()
    assert [(m["id"], m["default"]) for m in models] == [("groq", True), ("adapter", False)]
    err = dict(_events(client.post("/api/chat", json={"question": QUESTION, "model": "gpt-9"}).text))
    assert "not available" in err["error"]["message"]


def test_no_verification_unless_asked_and_graceful_when_unavailable() -> None:
    client, *_ = _build("python")
    names = [e for e, _ in _events(client.post("/api/chat", json={"question": QUESTION}).text)]
    assert "verification" not in names and "explanation" in names
    client, *_ = _build("python", verifier_available=False)
    events = dict(_events(client.post("/api/chat", json={"question": QUESTION, "verify_section": "304A"}).text))
    assert "not available" in events["verification"]["error"] and "explanation" in events and "error" not in events


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("LEGALGPT_LIVE_E2E") != "1", reason="set LEGALGPT_LIVE_E2E=1 for the live chain")
def test_live_full_chain() -> None:
    from app.main import create_app

    events = _events(TestClient(create_app()).post(
        "/api/chat", json={"question": QUESTION, "mode": "legal", "verify_section": "304A"}).text)
    data = {e: d for e, d in events if e != "token"}
    assert not data["meta"]["refused"] and data["sources"]
    assert "[" in "".join(d["text"] for e, d in events if e == "token")
    assert data["explanation"]["relevance"] is not None
    assert data["verification"].get("status") in ("CONSISTENT", "INSUFFICIENT", "INCONSISTENT")
