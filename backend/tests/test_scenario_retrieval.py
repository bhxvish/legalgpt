"""Everyday fact-pattern retrieval: lay-term expansion, the scope check and the floor rules."""

import numpy as np
import pytest

from app.core.lay_terms import LAY_TERMS, statutory_terms
from app.core.models import Chunk
from app.core.retriever import QueryExpander, Retriever
from app.core.scope import ScopeCheck, train
from app.finetuning.ipc_qa import offence_parts, plain_punishment


# ------------------------------------------------------------------ lexicon
@pytest.mark.parametrize(
    "question, contains",
    [
        ("Someone slapped me at the bus stop", "voluntarily causing hurt"),
        ("The seller took my money and never delivered the laptop", "cheating"),
        ("My employer kept the money I gave him", "criminal breach of trust"),
        ("Two men snatched her purse at knifepoint", "robbery"),
        ("He locked me in a room and would not let me leave", "wrongful confinement"),
        ("A man followed her home and keeps messaging her", "stalking"),
    ],
)
def test_lay_descriptions_get_statutory_wording(question: str, contains: str) -> None:
    assert any(contains in t for t in statutory_terms(question))


@pytest.mark.parametrize(
    "question",
    ["What is the punishment for murder?", "Define theft", "What is cheating?", "Explain criminal trespass",
     "What is the punishment for robbery?", "What does Section 498A provide?", "What is stalking under the IPC?"],
)
def test_questions_in_statutory_language_are_not_expanded(question: str) -> None:
    """Triggers are lay words only: expanding a question that already uses the IPC's terms once sent
    "punishment for murder" to the definition (s.300) instead of s.302."""
    assert statutory_terms(question) == []


def test_lexicon_rules_never_trigger_on_ipc_terms() -> None:
    for word in ("murder", "robbery", "cheating", "forgery", "trespass", "assault", "kidnapping", "extortion",
                 "defamation", "stalking", "dowry", "theft"):
        assert statutory_terms(f"What is {word}?") == [], word
    assert len(LAY_TERMS) >= 15


def test_search_query_appends_but_expand_does_not() -> None:
    e = QueryExpander()
    q = "Someone slapped me u/s 323?"
    assert "voluntarily causing hurt" in e.search_query(q)
    assert "voluntarily causing hurt" not in e.expand(q)  # the scope check reads the plain question


# ------------------------------------------------------------------ scope check
def test_logistic_regression_separates_a_toy_problem() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(1, 0.3, (30, 4)), rng.normal(-1, 0.3, (30, 4))])
    y = np.array([1.0] * 30 + [0.0] * 30)
    w, b = train(X, y)
    check = ScopeCheck(w, b)
    assert check.in_scope(np.ones(4)) and not check.in_scope(-np.ones(4))


def test_committed_scope_model_loads() -> None:
    check = ScopeCheck.load()
    assert check is not None and len(check.weights) == 384 and check.embedding_model


# ------------------------------------------------------------------ floor rules
class _Scope:
    threshold = 0.5

    def __init__(self, p: float) -> None:
        self.p = p

    def probability(self, _embedding) -> float:
        return self.p


def _retriever(p: float | None) -> Retriever:
    return Retriever(None, None, min_similarity=0.55, scope=None if p is None else _Scope(p), scope_floor=0.40)


def test_floor_rules() -> None:
    q = np.zeros(4)
    assert _retriever(None).floor_for(0.30, q) == (0.55, None)  # no scope check: the plain floor
    assert _retriever(0.9).floor_for(0.60, q) == (0.55, None)  # clears the floor: not consulted
    assert _retriever(0.9).floor_for(0.45, q)[0] == 0.40  # about a crime: lower floor
    assert _retriever(0.1).floor_for(0.45, q)[0] == 0.55  # not about a crime: unchanged
    # lexicon wording raised the similarity: answered only if the scope check agrees
    assert _retriever(0.1).floor_for(0.70, q, lexicon=True)[0] > 1.0
    assert _retriever(0.9).floor_for(0.70, q, lexicon=True)[0] == 0.40


def test_boilerplate_does_not_count_as_a_substantive_match() -> None:
    r = _retriever(0.9)
    from app.core.retriever import Candidate

    amend = Chunk("IPC:1:a", "IPC", "t", "IPC > Chapter I > Section 2", chapter="I", section="2")
    real = Chunk("IPC:304A:a", "IPC", "t", "IPC > Chapter XVI > Section 304A", chapter="XVI", section="304A")
    assert r.best_substantive([Candidate(amend, 0.6), Candidate(real, 0.45)]) == 0.45
    assert r.best_substantive([Candidate(amend, 0.6, explicit=True)]) == 0.6  # named by the user


# ------------------------------------------------------------------ plain-language LoRA targets
def test_plain_punishment_wording() -> None:
    assert plain_punishment("imprisonment of either description for a term which may extend to two years, or with fine, "
                            "or with both") == "imprisonment (simple or rigorous) of up to two years, or a fine, or both"
    assert plain_punishment("death, or imprisonment for life, and shall also be liable to fine") == \
        "death, or imprisonment for life, and a fine"
    assert "of at least one year and up to five years" in plain_punishment(
        "imprisonment of either description for a term which shall not be less than one year but which may extend to five years")


def test_offence_parts_only_for_whoever_sections() -> None:
    off = Chunk("IPC:304A:x", "IPC", "Section 304A. Causing death by negligence\nWhoever causes the death of any person by "
                "doing any rash or negligent act not amounting to culpable homicide, shall be punished with imprisonment of "
                "either description for a term which may extend to two years, or with fine, or with both.", "IPC > Section 304A",
                section="304A", title="Causing death by negligence")
    conduct, punishment = offence_parts(off)
    assert conduct.strip().startswith("causes the death") and punishment.startswith("imprisonment (simple or rigorous)")
    definition = Chunk("IPC:378:x", "IPC", "Section 378. Theft\nWhoever, intending to take dishonestly any movable property "
                       "out of the possession of any person without that person's consent, moves that property, is said to "
                       "commit theft.", "IPC > Section 378", section="378", title="Theft")
    assert offence_parts(definition) is None


# ------------------------------------------------------------------ definition questions
def test_defined_term_extraction() -> None:
    from app.core.retriever import defined_term

    assert defined_term("What is robbery under the IPC?") == "robbery"
    assert defined_term("How does the Indian Penal Code define making a false document?") == "making a false document"
    assert defined_term("Which IPC section defines abetment of a thing, and what does it say?") == "abetment of a thing"
    assert defined_term("What is lurking house-trespass under the IPC?") == "lurking house trespass"
    assert defined_term("Someone slapped me at the bus stop") is None


def test_title_index_uses_each_sections_own_chunk() -> None:
    from app.core.retriever import defined_sections, title_index

    chunks = [Chunk("IPC:390:a", "IPC", "Section 390. Robbery ...", "IPC > Chapter XVII > Section 390", section="390", title="Robbery"),
              Chunk("IPC:392:a", "IPC", "Section 392. Punishment for robbery ...", "IPC > Chapter XVII > Section 392",
                    section="392", title="Punishment for robbery"),
              Chunk("IPC:390:sa", "IPC", "amendment", "IPC > Chapter XVII > Section 390 > State Amendments", section="390",
                    title="Robbery (State Amendment)")]
    idx = title_index(chunks)
    assert idx["robbery"] == ["390"] and idx["punishment for robbery"] == ["392"]
    assert defined_sections("What is robbery under the IPC?", idx) == ["390"]
    assert defined_sections("What is the punishment for robbery?", idx) == ["392"]
    assert defined_sections("What is a succession certificate?", idx) == []
