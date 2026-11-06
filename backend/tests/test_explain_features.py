import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.models import SourceEvidence
from app.explainability import bias
from app.explainability.perturbation import fact_influence, source_influence, split_facts, word_importance
from app.explainability.reasoning import build_trace
from app.explainability.router import parse_sse


def src(marker, section, title="", kind="statute"):
    return SourceEvidence(marker=marker, chunk_id=f"ipc-{section}", doc_id="ipc", citation_path=f"IPC > Section {section}",
                          text=f"Section {section} text", similarity=0.7, score=0.7, role="Law", section=section,
                          title=title, kind=kind)


# --- the swap -------------------------------------------------------------------------------------------

def test_gender_swap_keeps_case_and_handles_her():
    text = "Her husband Ramesh beat her and took her jewellery. She told him and his mother."
    out, swaps = bias.swap(text)
    assert out == "His wife Rekha beat him and took his jewellery. He told her and her father."
    assert {"from": "Ramesh", "to": "Rekha"} in swaps and len(swaps) == 9


def test_swap_is_symmetric_on_plain_words_and_leaves_other_words():
    out, _ = bias.swap("The woman and the man; mother-in-law; Mr. Rao")
    assert out == "The man and the woman; father-in-law; Mrs. Rao"
    assert bias.swap("The truck hit a pedestrian.")[1] == []


def test_religion_swap_uses_names_and_places():
    out, swaps = bias.swap("Mohan Lal went to the temple.", "religion")
    assert out == "Mohammed Khan went to the mosque." and len(swaps) == 3


# --- the law and the comparison ---------------------------------------------------------------------------

def test_gender_specific_sections_cover_dowry_and_map_to_bns():
    ipc = bias.gender_specific_sections("ipc")
    assert "304B" in ipc and "498A" in ipc and "379" not in ipc
    bns = bias.gender_specific_sections("bns")
    assert "80" in bns and "85" in bns  # dowry death, cruelty by husband (official BPRD table)


def test_difference_in_a_gender_specific_section_is_law_not_bias():
    o = bias.Run("q", answer="Dowry death under section 304B [1].", cited=["304B", "498A"])
    s = bias.Run("q2", answer="Cruelty may not apply.", cited=["306"])
    r = bias.compare("gender", o, s, [{"from": "wife", "to": "husband"}], control=bias.Run("q", cited=["304B", "498A", "306"]))
    kinds = {d["section"]: d["kind"] for d in r.differences}
    assert kinds == {"304B": "law", "498A": "law", "306": "variation"}
    assert r.verdict == "law_required"


def test_unexplained_difference_or_stereotype_is_possible_bias():
    o = bias.Run("q", answer="Theft under section 379.", cited=["379"])
    s = bias.Run("q2", answer="No offence; women often exaggerate such complaints.", refused=False, cited=[])
    r = bias.compare("gender", o, s, [{"from": "he", "to": "she"}])
    assert r.verdict == "possible_bias"
    assert r.differences[0]["kind"] == "unexplained" and r.stereotypes[0]["why"] == "generalises about women"
    same = bias.compare("gender", o, bias.Run("q2", answer="Theft under section 379.", cited=["379"]), [])
    assert same.verdict == "no_difference"


def test_case_element_status_change_is_compared_per_section():
    o = bias.Run("t", cited=["380"], elements={"s.380": "CONSISTENT"})
    s = bias.Run("t2", cited=["380"], elements={"s.380": "INSUFFICIENT"})
    r = bias.compare("gender", o, s, [{"from": "he", "to": "she"}])
    assert r.verdict == "possible_bias" and r.differences[0]["side"] == "CONSISTENT → INSUFFICIENT"


def test_sensitivity_and_cited_sections():
    assert "dowry" in bias.sensitivity("Her in-laws demanded dowry")
    assert "304B" in bias.sensitivity("x", ["304B"])
    assert bias.sensitivity("Someone stole my bicycle") is None
    sources = [{"marker": 1, "section": "379"}, {"marker": 2, "section": "380"}, {"marker": 3, "section": "", "kind": "judgment"}]
    assert bias.cited_sections("Theft [1]; see also section 457 and [3].", sources) == ["379", "457"]


# --- reasoning trace -------------------------------------------------------------------------------------

class JsonLLM:
    model_id = "fake"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, messages):
        return self.reply


def test_trace_keeps_only_sections_and_sources_that_exist():
    reply = json.dumps({
        "issue": "Is taking a bicycle from a shed theft?",
        "rules": [{"section": "379", "applies": True, "why": "movable property taken", "sources": [1, 9]},
                  {"section": "420", "applies": False, "why": "made up", "sources": [1]}],
        "application": [{"fact": "took my bicycle", "element": "movable property", "section": "379", "holds": "yes", "sources": [1]}],
        "conclusion": "Theft under s.379.",
        "key_facts": ["stole my bicycle", "a stolen car"]})
    t = build_trace("Someone stole my bicycle from the shed.", "Theft [1].", [src(1, "379", "Theft")], JsonLLM("```json\n" + reply + "\n```"))
    assert [r["section"] for r in t["rules"]] == ["379"] and t["rules"][0]["sources"] == [1]
    assert t["removed"] == 2  # source 9 and section 420
    assert t["key_facts"] == ["stole my bicycle"] and t["unquoted_key_facts"] == ["a stolen car"]


def test_trace_reports_a_reply_without_json():
    assert "error" in build_trace("q", "a", [src(1, "379")], JsonLLM("I cannot"))


# --- perturbation -------------------------------------------------------------------------------------------

class WordEmbedder:
    """Embeds by counting a few words, so removing a word changes the vector predictably."""
    vocab = ["stole", "bicycle", "theft", "night"]

    def embed(self, texts):
        return np.array([[t.lower().count(w) + 0.01 for w in self.vocab] for t in texts], dtype=np.float32)


class Expander:
    def expand(self, q):
        return q

    def search_query(self, q):
        return q + ("\ntheft" if "stole" in q.lower() else "")


def test_word_importance_finds_the_words_that_tie_the_question_to_the_section():
    store = SimpleNamespace(get_embeddings=lambda ids: {"ipc-379": np.array([0, 0, 1, 0], dtype=np.float32)})
    scope = SimpleNamespace(probability=lambda e: float(min(1.0, 0.2 + 0.6 * e[0])))
    retriever = SimpleNamespace(store=store, embedder=WordEmbedder(), expander=Expander(), scope=scope)
    out = word_importance("Someone stole my bicycle at night", [src(1, "379")], retriever)
    by = {w["word"]: w for w in out["words"]}
    assert by["stole"]["match_drop"]["1"] > 0.3 and by["bicycle"]["match_drop"]["1"] <= 0.05
    assert by["stole"]["scope_drop"] > 0.3 and "Someone" in by  # stopwords ("my", "at") are skipped
    assert "my" not in by


class SectionLLM:
    """Cites [n] for every source it is given; 'refuses' without s.379."""
    model_id = "fake"

    def generate(self, messages):
        text = messages[-1].content
        if "Section 379 text" not in text:
            return "I don't know — the retrieved sources do not cover this question."
        return " ".join(f"[{m}]" for m in range(1, 4) if f"[{m}]" in text) or "x"


class Builder:
    def build(self, question, sources, mode, history, law="ipc"):
        from app.core.models import Message
        body = "\n".join(f"[{s.marker}] ({s.citation_path})\n{s.text}" for s in sources)
        return [Message("system", "rules"), Message("user", f"{body}\n\nQ: {question}")]


def test_source_influence_marks_the_decisive_source():
    sources = [src(1, "379"), src(2, "380")]
    out = source_influence("q", "Theft [1] [2].", sources, SectionLLM(), Builder())
    rows = {r["section"]: r for r in out["sources"]}
    assert rows["379"]["decisive"] and rows["379"]["refusal_changed"]
    assert rows["380"]["changed"] and rows["380"]["lost"] == ["380"]
    assert out["control"]["changed"] is False


def test_fact_influence_and_split():
    assert split_facts("Someone broke into my shed at night and stole my bicycle. Which offence is this?") == [
        "Someone broke into my shed at night and stole my bicycle"]

    class FactLLM(SectionLLM):
        def generate(self, messages):
            return "Theft [1]." if "stole" in messages[-1].content else "I don't know — the retrieved sources do not cover this question."

    q = "My neighbour stole my bicycle. It was raining that day."
    out = fact_influence(q, [src(1, "379")], FactLLM(), Builder(), key_facts=["stole my bicycle"], answer="Theft [1].")
    rows = {r["fact"]: r for r in out["facts"]}
    assert rows["stole my bicycle"]["influential"] and rows["stole my bicycle"]["named_by_trace"]
    assert not rows["It was raining that day"]["influential"]
    assert out["faithfulness"]["agreement"] == 1.0


def test_parse_sse_joins_tokens():
    events = ['event: meta\ndata: {"refused": false, "model": "m"}\n\n', 'event: token\ndata: {"text": "Hi "}\n\n',
              'event: token\ndata: {"text": "there"}\n\n', 'event: sources\ndata: [{"marker": 1}]\n\n']
    got = parse_sse(events)
    assert got["answer"] == "Hi there" and got["meta"]["model"] == "m" and got["sources"] == [{"marker": 1}]


def test_gender_specific_section_in_both_versions_gets_a_check_note():
    o = bias.Run("q", cited=["304B", "498A"])
    s = bias.Run("q2", cited=["304B", "498A"])
    r = bias.compare("gender", o, s, [{"from": "wife", "to": "husband"}])
    assert r.verdict == "no_difference" and [n["section"] for n in r.law_notes] == ["304B", "498A"]


def test_ruled_out_sections_and_consequences_of_a_gender_specific_offence():
    ans = "Section 304B applies only to the death of a woman, so it does not apply here. Culpable homicide under section 304 may apply [2]."
    assert bias.ruled_out(ans) == {"304B"}
    assert bias.relied_sections(ans, [{"marker": 1, "section": "304B"}, {"marker": 2, "section": "304"}]) == ["304"]
    o = bias.Run("q", cited=["304B", "498A"])
    s = bias.Run("q2", cited=["304", "498A"])
    r = bias.compare("gender", o, s, [{"from": "wife", "to": "husband"}])
    assert r.verdict == "law_required" and {d["section"]: d["kind"] for d in r.differences} == {"304B": "law", "304": "consequence"}


def test_caste_income_and_region_swap_words_both_ways():
    assert bias.swap("Ravi Sharma, a Brahmin, was attacked.", "caste")[0] == "Ravi Jatav, a Dalit, was attacked."
    assert bias.swap("A Dalit man and his upper-caste neighbour.", "caste")[0] == "A Brahmin man and his lower-caste neighbour."
    out, swaps = bias.swap("A wealthy businessman from a bungalow took Rs. 5,000.", "income")
    assert out == "A poor daily-wage labourer from a slum took Rs. 5,000." and len(swaps) == 3  # amounts stay
    assert bias.swap("The Kannadiga shopkeeper", "region")[0] == "The Bihari shopkeeper"


def test_a_difference_must_hold_in_both_runs_of_a_version():
    first, second = bias.Run("a", cited=["379"]), bias.Run("b", cited=["379", "420"])
    assert bias.compare("income", first, second, [], control=first).verdict == "possible_bias"
    assert bias.compare("income", first, second, [], control=first, repeat=bias.Run("b", cited=["379"])).verdict == "variation"
    assert bias.compare("income", first, second, [], control=first, repeat=second).verdict == "possible_bias"


def test_a_definition_and_its_punishment_are_one_offence():
    titles = {"441": "Criminal trespass", "447": "Punishment for criminal trespass", "143": "Punishment", "379": "Punishment for theft"}
    offences = bias.same_offence(titles)
    assert offences == {"447": "441"}
    first, second = bias.Run("a", cited=["441"]), bias.Run("b", cited=["447"])
    assert bias.compare("income", first, second, []).verdict == "possible_bias"
    assert bias.compare("income", first, second, [], offences=offences).verdict == "no_difference"


def test_only_an_unexplained_difference_needs_second_runs():
    """A second run can only clear a difference, so skipping it when there is none changes no verdict."""
    first = bias.Run("a", cited=["379"])
    for control, repeat in ((None, None), (bias.Run("a", cited=["420"]), bias.Run("b", cited=["406"]))):
        assert bias.compare("income", first, bias.Run("b", cited=["379"]), [], control=control, repeat=repeat).verdict != "possible_bias"


def test_both_runs_together_and_reports_failures():
    assert bias.both(lambda: 1, lambda: 2) == (1, 2)
    with pytest.raises(ValueError):
        bias.both(lambda: 1, lambda: int("x"))


def test_an_empty_reading_of_a_case_file_is_tried_again():
    from app.case_analysis.router import CaseRouter

    router = CaseRouter.__new__(CaseRouter)
    readings = iter([bias.Run("a"), bias.Run("a", cited=["379"])])
    router._assess_once = lambda *_: next(readings)
    assert router._assess("a", None, False).cited == ["379"]


def test_swap_keeps_a_and_an_correct():
    assert bias.swap("He cheated an old woman.", "age")[0] == "He cheated a young woman."
    assert bias.swap("A wealthy man and an upper-caste family.", "income")[0].startswith("A poor man")


def test_age_swap_changes_ages_not_every_man():
    assert bias.swap("A 15-year-old boy hit a man aged 70.", "age")[0] == "A 40-year-old man hit a man aged 40."
    assert bias.swap("A man stole a phone six years ago.", "age")[1] == []  # not an age


def test_counterfactual_adds_a_sentence_when_there_is_nothing_to_swap():
    cf = bias.counterfactual("Someone stole my bicycle.", "caste")
    assert cf.mode == "added" and cf.first.endswith("upper-caste family.") and cf.second.endswith("Dalit family.")
    assert bias.counterfactual("A Brahmin stole my bicycle.", "caste").mode == "swap"
    assert bias.counterfactual("Someone stole my bicycle.", "gender").mode == "none"


def test_age_sections_count_as_law_and_caste_gets_the_scst_note():
    o = bias.Run("q", cited=["361", "379"])
    s = bias.Run("q2", cited=["379"])
    r = bias.compare("age", o, s, [{"from": "15", "to": "40"}])
    assert r.verdict == "law_required" and r.differences[0]["kind"] == "law"
    r = bias.compare("caste", bias.Run("q", cited=["379"]), bias.Run("q2", cited=["380"]), [{"from": "a", "to": "b"}], mode="added")
    assert r.verdict == "possible_bias" and r.mode == "added"
    assert "SC/ST" in r.law_notes[-1]["why"] and r.law_notes[-1]["plain"]
