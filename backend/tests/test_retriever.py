import pytest

from app.core.models import Chunk
from app.core.retriever import Candidate, QueryExpander, Retriever
from app.core.vector_store import VectorStore
from tests.conftest import FakeEmbedder


def _chunk(cid: str, text: str, role: str = "Law", section: str = "") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="IPC", text=text, citation_path=f"IPC > Section {section or cid}", section=section, role=role)


@pytest.fixture
def retriever(fake_embedder: FakeEmbedder, memory_store: VectorStore) -> Retriever:
    chunks = [
        _chunk("304A", "Section 304A. Causing death by negligence. Whoever causes death by rash or negligent act.", section="304A"),
        _chunk("378", "Section 378. Theft. Whoever intending to take dishonestly any movable property commits theft.", section="378"),
        _chunk("420", "Section 420. Cheating and dishonestly inducing delivery of property.", section="420"),
    ]
    memory_store.upsert(chunks, fake_embedder.embed([c.text for c in chunks]))
    return Retriever(fake_embedder, memory_store, min_similarity=0.2)


# ------------------------------------------------------------- QueryExpander


def test_query_expander_expands_abbreviations() -> None:
    assert QueryExpander().expand("Punishment u/s 302 IPC r/w CrPC?") == (
        "Punishment under section 302 Indian Penal Code read with Code of Criminal Procedure?"
    )


def test_query_expander_extracts_section_refs() -> None:
    q = "Is it covered u/s 304a IPC, or Section 299, or 300 of the Indian Penal Code?"
    assert QueryExpander.section_refs(q) == ["304A", "299", "300"]
    assert QueryExpander.section_refs("How many years in prison?") == []


# --------------------------------------------------------- assemble_context


def test_assemble_context_empty_candidates_returns_empty(retriever: Retriever) -> None:
    assert retriever.assemble_context([]) == []


def test_assemble_context_returns_empty_when_nothing_clears_floor(retriever: Retriever) -> None:
    weak = [Candidate(_chunk("a", "text"), similarity=0.1, score=0.12), Candidate(_chunk("b", "text"), similarity=0.19)]
    assert retriever.assemble_context(weak) == []


def test_assemble_context_respects_limits_and_numbers_markers(retriever: Retriever) -> None:
    cands = [Candidate(_chunk(str(i), "x" * 1000), similarity=0.9 - i / 100) for i in range(10)]
    ev = retriever.assemble_context(cands, max_chunks=6, max_context_chars=3500)
    assert [e.marker for e in ev] == [1, 2, 3]
    assert [e.chunk_id for e in ev] == ["0", "1", "2"]
    assert retriever.assemble_context(cands, max_chunks=2, max_context_chars=100_000)[-1].marker == 2


def test_explicit_section_match_bypasses_floor(retriever: Retriever) -> None:
    ev = retriever.assemble_context([Candidate(_chunk("420", "text"), similarity=0.05, explicit=True)])
    assert [e.chunk_id for e in ev] == ["420"]


# ------------------------------------------------------------------ rerank


def test_rerank_applies_role_preference_and_dedupes(retriever: Retriever) -> None:
    law = Candidate(_chunk("law", "t", role="Law"), similarity=0.50)
    facts = Candidate(_chunk("facts", "t", role="Facts"), similarity=0.55)
    ranked = retriever.rerank([facts, law, Candidate(_chunk("law", "t", role="Law"), similarity=0.50, explicit=False)])
    assert [c.chunk.chunk_id for c in ranked] == ["law", "facts"]  # 0.50 * 1.2 > 0.55 * 1.0
    assert ranked[0].score == pytest.approx(0.60)


# --------------------------------------------------------------- retrieve


def test_retrieve_in_corpus_question(retriever: Retriever) -> None:
    ev = retriever.retrieve("What is the punishment for causing death by negligent driving?")
    assert ev and ev[0].section == "304A"


def test_retrieve_out_of_corpus_question_is_empty(retriever: Retriever) -> None:
    assert retriever.retrieve("What is the capital gains tax rate on mutual funds?") == []


def test_retrieve_named_section_is_included(retriever: Retriever) -> None:
    ev = retriever.retrieve("Explain s. 420 please")
    assert "420" in [e.section for e in ev]


def test_follow_up_uses_previous_question_as_context(retriever: Retriever) -> None:
    follow_up = "How long can sentence be?"
    assert retriever.retrieve(follow_up) == []  # alone, it matches nothing
    ev = retriever.retrieve(follow_up, context="Whoever causes death by a rash or negligent act")
    assert ev and ev[0].section == "304A"


def test_context_does_not_hide_a_change_of_topic(retriever: Retriever) -> None:
    ev = retriever.retrieve("Whoever intending to take dishonestly movable property commits theft", context="negligent death")
    assert ev[0].section == "378"


@pytest.mark.parametrize(
    "question, refs",
    [
        ("Appeal under Section 378(1) of the Code of Criminal Procedure against acquittal", []),
        ("convicted u/s 302 IPC and Section 25 of the Arms Act, 1959", ["302"]),
        ("statement u/s 313 Cr.P.C. was recorded; charge under Section 304A", ["304A"]),
        ("Section 6 of the POCSO Act and Section 377 of the Indian Penal Code", ["377"]),
        ("What does Section 279 of the Indian Penal Code provide?", ["279"]),
        ("Sections 302 and 34 apply", ["302"]),
    ],
)
def test_section_refs_ignore_other_enactments(question: str, refs: list[str]) -> None:
    assert QueryExpander.section_refs(question) == refs
