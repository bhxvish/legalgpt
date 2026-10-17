from types import SimpleNamespace

import numpy as np

from app.core.models import Chunk, SourceEvidence
from app.core.prompt_builder import LEGAL_SYSTEM_PROMPT, PromptBuilder
from app.judgments.library import (JudgmentDoc, JudgmentSearch, chunk_meta, passages, short_citation,
                                   tidy_title)


def doc(roles, title="Ram v. State", court="Supreme Court of India", year=2023):
    return JudgmentDoc("c1", title, court, year, [f"Sentence {i}." for i in range(len(roles))], roles, "library")


def test_passages_group_consecutive_roles_and_skip_unlabelled():
    ps = passages(doc(["Facts", "Facts", None, "Ruling", "Ruling", "Ruling", "Ruling", "Ruling"]))
    assert [p.role for p in ps] == ["Facts", "Ruling", "Ruling"]  # at most 4 sentences per passage
    assert ps[0].text == "Sentence 0. Sentence 1."
    assert ps[1].citation_path == "Ram v. State (SC, 2023), sentences 4-7"
    assert chunk_meta(ps[1]) == ("Supreme Court of India", 2023, "library")


def test_titles_and_citations_read_cleanly():
    assert tidy_title("STATE OF U.P. versus RAM KUMAR @ RAMU") == "State of U.P. v. Ram Kumar @ Ramu"
    assert tidy_title("Narender vs State of Delhi (12 Oct, 2021)") == "Narender v. State of Delhi"
    assert short_citation("A v. B", "", 2021) == "A v. B (2021)"
    assert short_citation("A v. B", "", None) == "A v. B"


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def count(self):
        return len(self.hits)

    def query(self, embedding, k, where=None):
        w = where["$and"][0] if "$and" in where else where
        roles = w["role"]["$in"] if "$in" in w["role"] else [w["role"]]
        return [h for h in self.hits if h.chunk.role in roles]


class FakeEmbedder:
    def embed(self, texts):
        return np.zeros((len(texts), 4))


def hit(case, role, sim, i=0):
    c = Chunk(chunk_id=f"jdg::{case}::{i}", doc_id=case, text=f"{case} text", citation_path=f"{case} (SC, 2023), sentence 1",
              chapter="Supreme Court of India|2023|library", title=case, role=role)
    return SimpleNamespace(chunk=c, similarity=sim)


def test_search_keeps_the_floor_roles_and_one_passage_per_case():
    store = FakeStore([hit("A", "Ruling", 0.8), hit("A", "Ruling", 0.79, 1), hit("B", "Argument", 0.9),
                       hit("C", "Law Applied", 0.6), hit("D", "Ruling", 0.4)])
    js = JudgmentSearch(store, FakeEmbedder(), min_similarity=0.55, k=3)
    got = js.search("q")
    assert [(c.doc_id, round(s, 2)) for c, s in got] == [("A", 0.8), ("C", 0.6)]  # no Argument, no 2nd A, D below floor
    sources = js.to_sources(got, first_marker=4)
    assert [s.marker for s in sources] == [4, 5]
    assert sources[0].kind == "judgment" and sources[0].court == "Supreme Court of India" and sources[0].year == 2023


def test_prompt_marks_judgments_and_states_the_rule():
    s = [SourceEvidence(marker=1, chunk_id="ipc", doc_id="ipc", citation_path="IPC > Section 304A", text="Whoever...",
                        similarity=0.8, score=0.8, role="Law", section="304A"),
         SourceEvidence(marker=2, chunk_id="j", doc_id="A", citation_path="A v. B (SC, 2023), sentence 3", text="We hold...",
                        similarity=0.7, score=0.7, role="Ruling", kind="judgment", title="A v. B", year=2023)]
    text = PromptBuilder.format_sources(s)
    assert "[1] (IPC > Section 304A)" in text
    assert "[2] (JUDGMENT: A v. B (SC, 2023), sentence 3; court not recorded; Ruling passage)" in text
    assert "Sources marked JUDGMENT" in LEGAL_SYSTEM_PROMPT


def test_chat_adds_judgments_after_statute_sources_only_when_statute_found():
    from app.core.chat_router import ChatRouter
    from app.core.retriever import RetrievalResult

    router = ChatRouter.__new__(ChatRouter)
    router.judgments = JudgmentSearch(FakeStore([hit("A", "Ruling", 0.8)]), FakeEmbedder())
    res = RetrievalResult(sources=[], best_similarity=0.9, min_similarity=0.3, candidates=3, query="theft (section 379)")
    assert router._judgment_sources("q", res, 0) == []  # nothing from the IPC: no case law either
    got = router._judgment_sources("q", res, 3)
    assert [s.marker for s in got] == [4] and got[0].kind == "judgment"
    router.judgments = None
    assert router._judgment_sources("q", res, 3) == []
