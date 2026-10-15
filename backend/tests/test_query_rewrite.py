"""LLM query rewrite for everyday fact patterns: parsing, the scope gate, and the chat wiring."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.chat_router import ChatRouter
from app.core.models import Chunk, Message
from app.core.query_rewriter import QueryRewriter, Rewrite, parse
from app.core.retriever import Retriever
from tests.conftest import StubLLM

CHUNKS = [
    Chunk(chunk_id="IPC:419:a", doc_id="IPC", section="419", title="Punishment for cheating by personation",
          citation_path="IPC > Chapter XVII > Section 419",
          text="Section 419. Punishment for cheating by personation\nWhoever cheats by personation shall be punished."),
    Chunk(chunk_id="IPC:323:a", doc_id="IPC", section="323", title="Punishment for voluntarily causing hurt",
          citation_path="IPC > Chapter XVI > Section 323",
          text="Section 323. Punishment for voluntarily causing hurt\nWhoever voluntarily causes hurt shall be punished."),
]
QUESTION = "A caller pretending to be from my bank got my OTP and emptied my account."
REPLY = json.dumps({"criminal": True, "offences": [{"name": "Cheating by personation", "section": "419"}]})


class _Scope:
    threshold = 0.5

    def __init__(self, p: float) -> None:
        self.p = p

    def probability(self, _embedding) -> float:
        return self.p


@pytest.fixture
def retriever(fake_embedder, memory_store) -> Retriever:
    memory_store.upsert(CHUNKS, fake_embedder.embed([c.text for c in CHUNKS]))
    return Retriever(fake_embedder, memory_store, min_similarity=0.9, scope=_Scope(0.9), scope_floor=0.9)


# ------------------------------------------------------------------ parsing
def test_parse_reads_the_json_in_a_reply() -> None:
    rw = parse("Sure:\n```json\n" + REPLY + "\n```")
    assert rw.criminal and rw.offences == [("Cheating by personation", "419")]
    assert rw.text == "Cheating by personation (section 419)"


@pytest.mark.parametrize("reply", ["no json here", '{"offences": []}', '{"criminal": "yes"}', "{broken"])
def test_unusable_replies_give_none(reply: str) -> None:
    assert parse(reply) is None


def test_parse_cleans_sections_and_names() -> None:
    rw = parse(json.dumps({"criminal": True, "offences": [
        {"name": "Cheating (dishonest; inducing)", "section": "s. 420"}, {"name": "dup", "section": "420"},
        {"name": "x", "section": "not a number"}, {"name": "a", "section": "304a"}, {"name": "b", "section": "1"},
        {"name": "c", "section": "2"}]}))
    assert [s for _, s in rw.offences] == ["420", "304A", "1"]  # deduplicated, invalid dropped, at most 3
    assert Rewrite.from_text(rw.text).offences == rw.offences  # the disclosed text round-trips


def test_civil_matters_carry_no_offences() -> None:
    rw = parse('{"criminal": false, "offences": [{"name": "theft", "section": "379"}]}')
    assert rw.criminal is False and rw.offences == []


def test_rewrites_are_cached_per_model() -> None:
    llm, rewriter = StubLLM(reply=REPLY), QueryRewriter()
    assert rewriter.rewrite(QUESTION, llm) == rewriter.rewrite(QUESTION, llm)
    assert len(llm.calls) == 1


def test_a_failing_model_gives_no_rewrite() -> None:
    assert QueryRewriter().rewrite(QUESTION, StubLLM(fail=True)) is None


# ------------------------------------------------------------------ retrieval
def test_a_rewrite_fetches_its_sections(retriever: Retriever) -> None:
    plain = retriever.retrieve_with_diagnostics(QUESTION)
    assert not plain.sources  # the lay wording alone does not reach the floor
    res = retriever.retrieve_with_diagnostics(QUESTION, rewrite=parse(REPLY))
    assert [s.section for s in res.sources] == ["419"]
    assert res.rewrite["used"] and res.query.endswith("Cheating by personation (section 419)")


@pytest.mark.parametrize("p, criminal, used", [(0.9, True, True), (0.3, True, False), (0.9, False, False)])
def test_the_scope_gate(retriever: Retriever, p: float, criminal: bool, used: bool) -> None:
    """A rewrite steers retrieval only when the LLM says criminal *and* the scope check agrees."""
    retriever.scope = _Scope(p)
    rw = Rewrite(criminal, [("Cheating by personation", "419")] if criminal else [])
    res = retriever.retrieve_with_diagnostics(QUESTION, rewrite=rw)
    assert res.rewrite["used"] is used
    assert bool(res.sources) is used


# ------------------------------------------------------------------ chat
def _chat(retriever: Retriever, llm: StubLLM) -> list[tuple[str, dict]]:
    app = FastAPI()
    app.include_router(ChatRouter(retriever, llm, rewriter=QueryRewriter()).router)
    body = TestClient(app).post("/api/chat", json={"question": QUESTION}).text
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class _TwoReplies(StubLLM):
    """Answers the rewrite prompt with JSON and the question with a cited answer."""

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return REPLY if "Reply with JSON only" in messages[0].content else self.reply


def test_chat_uses_the_rewrite_and_reports_it(retriever: Retriever) -> None:
    llm = _TwoReplies(reply="This is cheating by personation under Section 419 [1].")
    events = dict(_chat(retriever, llm))
    assert [s["section"] for s in events["sources"]] == ["419"]
    assert events["retrieval"]["rewrite"]["used"] is True
    assert events["retrieval"]["rewrite"]["offences"][0]["section"] == "419"


def test_a_question_naming_its_section_is_not_rewritten(retriever: Retriever) -> None:
    llm = _TwoReplies()
    app = FastAPI()
    app.include_router(ChatRouter(retriever, llm, rewriter=QueryRewriter()).router)
    TestClient(app).post("/api/chat", json={"question": "What does section 323 say?"})
    assert not any("Reply with JSON only" in m[0].content for m in llm.calls)


def test_visible_reasoning_is_skipped() -> None:
    rw = parse('<think>maybe {"criminal": false} ... no</think>\n' + REPLY)
    assert rw.criminal and rw.offences == [("Cheating by personation", "419")]


def test_a_rate_limited_rewrite_does_not_fail_the_question() -> None:
    from app.core.llm_client import LLMRateLimitError

    class _Limited(StubLLM):
        def generate(self, messages):
            raise LLMRateLimitError("The hosted model's daily usage limit has been reached.")

    assert QueryRewriter().rewrite(QUESTION, _Limited()) is None


def test_strict_rewrites_raise_instead_of_falling_back() -> None:
    """Benchmarks use strict=True: a rate-limited rewrite must not silently become a no-rewrite answer."""
    from app.core.llm_client import LLMRateLimitError

    class _Limited(StubLLM):
        def generate(self, messages):
            raise LLMRateLimitError("too many requests")

    with pytest.raises(LLMRateLimitError):
        QueryRewriter().rewrite(QUESTION, _Limited(), strict=True)
