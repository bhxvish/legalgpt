"""Module 3 pieces that need no GPU or model download: instruction building, case-level
splitting, citation scoring, the comparator, and the LLM_BACKEND switch."""

import json
from pathlib import Path

import pytest

from app.annotation.models import LabeledSentence, sentence_id
from app.config import Settings
from app.core.llm_client import AdapterClient, GroqClient, LLMClient, LLMClientError
from app.core.models import SourceEvidence
from app.core.prompt_builder import LEGAL_SYSTEM_PROMPT, NOT_COVERED_REPLY
from app.finetuning.instruction_builder import InstructionBuilder
from app.finetuning.model_comparator import ModelComparator
from app.main import build_llm_client
from tests.conftest import StubLLM


def _case(case_id: str, section: str, n: int = 3) -> list[LabeledSentence]:
    rows = []
    roles = (["Facts"] * n + ["Argument"] * 2 + ["Law Applied"] * 2 + ["Precedent"] + ["Ruling"] * 2)
    for i, role in enumerate(roles):
        text = {
            "Facts": f"In case {case_id} the accused did act {i} on the highway.",
            "Argument": f"Counsel for {case_id} submitted point {i}.",
            "Law Applied": f"Section {section} of the Indian Penal Code punishes conduct {i}.",
            "Precedent": f"In Kumar v. State the court held principle {i}.",
            "Ruling": f"We hold that the conviction in {case_id} under Section {section} stands, point {i}.",
        }[role]
        rows.append(LabeledSentence(sentence_id(case_id, i), case_id, i, text, role))
    return rows


@pytest.fixture
def corpus() -> list[LabeledSentence]:
    return [s for i in range(10) for s in _case(f"c{i}", ["302", "304A", "379", "420", "498A"][i % 5])]


def test_build_from_case_uses_only_that_cases_sentences(corpus: list[LabeledSentence]) -> None:
    examples = InstructionBuilder().build_from_case([s for s in corpus if s.case_id == "c3"])
    assert {e.task for e in examples} == {"facts_to_law", "facts_args_to_ruling", "grounded_answer"}
    for e in examples:
        assert e.source_case_id == "c3"
        assert "c3" in e.output or "420" in e.output
    law = next(e for e in examples if e.task == "facts_to_law")
    assert "the accused did act" in law.input and "Section 420" in law.output


def test_grounded_answer_mirrors_the_inference_prompt(corpus: list[LabeledSentence]) -> None:
    ex = next(e for e in InstructionBuilder().build_from_case(corpus[:10]) if e.task == "grounded_answer")
    assert ex.system == LEGAL_SYSTEM_PROMPT
    assert ex.instruction.startswith("SOURCES:\n[1] (c0 > ") and "\n\nQUESTION: " in ex.instruction
    score = ModelComparator.score_citations(ex.output, [
        SourceEvidence(marker=i, chunk_id="x", doc_id="c0", citation_path="p", text="t", similarity=1, score=1, role="Law")
        for i in range(1, 7)])
    assert score.markers and not score.invalid_markers  # every target citation resolves


def test_refusals_pair_unrelated_cases(corpus: list[LabeledSentence]) -> None:
    cases: dict[str, list[LabeledSentence]] = {}
    for s in corpus:
        cases.setdefault(s.case_id, []).append(s)
    refusals = InstructionBuilder().build_refusals(cases)
    assert refusals and all(r.output == NOT_COVERED_REPLY for r in refusals)
    for r in refusals:
        own_section = next(s.text for s in cases[r.source_case_id] if s.label == "Law Applied").split()[1]
        assert f"Section {own_section} " not in r.instruction.split("QUESTION:")[0]  # sources are from another offence


def test_split_by_case_never_shares_a_case(corpus: list[LabeledSentence]) -> None:
    b = InstructionBuilder()
    splits = b.split_by_case(b.build_corpus(corpus), (0.8, 0.1, 0.1), seed=5)
    sets = {k: {e.source_case_id for e in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert len(sets["val"]) == 1 and len(sets["test"]) == 1 and len(sets["train"]) == 8
    assert b.split_by_case(b.build_corpus(corpus), seed=5) == splits


def test_format_dataset_entry_without_tokenizer(corpus: list[LabeledSentence]) -> None:
    ex = InstructionBuilder().build_from_case(corpus[:10])[0]
    text = InstructionBuilder().format_dataset_entry(ex)
    assert text.startswith("### System\n") and "### Instruction\n" in text and text.endswith(ex.output)


# ----------------------------------------------------------- score_citations


def _src(marker: int, section: str, text: str) -> SourceEvidence:
    return SourceEvidence(marker=marker, chunk_id=f"c{marker}", doc_id="IPC", citation_path=f"IPC > Section {section}",
                          text=text, similarity=0.8, score=0.9, role="Law", section=section)


SOURCES = [_src(1, "304A", "Whoever causes death by negligence ... two years."),
           _src(2, "302", "Whoever commits murder shall be punished with death. See Jacob Mathew v. State of Punjab.")]


def test_score_citations_grounded_answer() -> None:
    s = ModelComparator.score_citations("Under Section 304A the punishment is two years [1]; murder is under Section 302 [2].", SOURCES)
    assert s.markers == [1, 2] and s.sections == ["302", "304A"] and s.citation_accuracy == 1.0 and not s.refused
    assert s.uses_markers
    assert not ModelComparator.score_citations("Section 304A applies.", SOURCES).uses_markers


def test_score_citations_flags_hallucinations() -> None:
    s = ModelComparator.score_citations(
        "Section 307 applies [3], as held in Virsa Singh v. State of Punjab, and Section 304A applies 【1】.", SOURCES)
    assert s.invalid_markers == [3] and s.ungrounded_sections == ["307"]
    assert s.ungrounded_precedents == ["Virsa Singh v. State"]
    assert s.markers == [3, 1]  # full-width marker canonicalized like the chat stream
    assert s.citation_accuracy == pytest.approx(2 / 5)  # [1] and Section 304A grounded; [3], 307, Virsa Singh not


def test_score_citations_precedent_in_evidence_and_refusal() -> None:
    assert ModelComparator.score_citations("See Jacob Mathew v. State of Punjab.", SOURCES).citation_accuracy == 1.0
    s = ModelComparator.score_citations(NOT_COVERED_REPLY, SOURCES)
    assert s.refused and s.citation_accuracy is None


# ------------------------------------------------------------- comparator


class FixedRetriever:
    def retrieve(self, question: str) -> list[SourceEvidence]:
        return [] if "cake" in question else SOURCES


def test_comparator_runs_same_evidence_through_each_model_and_saves(tmp_path: Path) -> None:
    clients = {"base": StubLLM("Section 307 applies [4]."), "tuned": StubLLM("Section 304A applies [1].")}
    comp = ModelComparator(clients, [("Punishment for negligent death?", "test"), ("Best cake recipe?", "test")])
    rows = comp.run(FixedRetriever(), log=lambda _: None)
    assert rows[0].answers["base"].score.citation_accuracy == 0.0
    assert rows[0].answers["tuned"].score.citation_accuracy == 1.0
    assert rows[1].answers["tuned"].score is None  # no evidence: the app refuses without a model call
    assert clients["tuned"].calls[0] == clients["base"].calls[0]  # identical prompts
    json_path, md_path = comp.save(rows, tmp_path, {"adapter": "x"})
    summary = json.loads(json_path.read_text(encoding="utf-8"))["summary"]
    assert summary["tuned"]["mean_citation_accuracy"] == 1.0 and summary["base"]["ungrounded_sections"] == 1
    assert "| mean_citation_accuracy | 0.0 | 1.0 |" in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------- backend switch


def test_llm_backend_switch(tmp_path: Path) -> None:
    assert isinstance(build_llm_client(Settings(LLM_BACKEND="groq")), GroqClient)
    with pytest.raises(LLMClientError, match="no adapter"):
        build_llm_client(Settings(LLM_BACKEND="adapter", LORA_DIR=str(tmp_path)))
    adapter = tmp_path / "v0.2-test"
    adapter.mkdir()
    (adapter / "adapter_meta.json").write_text(json.dumps({"base_model": "Qwen/Qwen2.5-1.5B-Instruct"}), encoding="utf-8")
    client = build_llm_client(Settings(LLM_BACKEND="adapter", LORA_DIR=str(tmp_path)))
    assert isinstance(client, AdapterClient) and isinstance(client, LLMClient)
    assert client.model_id == "Qwen/Qwen2.5-1.5B-Instruct+lora:v0.2-test"  # nothing loaded until first use
    with pytest.raises(ValueError):
        Settings(LLM_BACKEND="other")


def test_adapter_client_requires_metadata(tmp_path: Path) -> None:
    with pytest.raises(LLMClientError):
        AdapterClient(tmp_path)
