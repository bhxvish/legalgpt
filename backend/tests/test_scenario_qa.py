"""Scenario training examples: built from the matching section's text, cited, never refusals."""

from app.core.models import Chunk, SourceEvidence
from app.finetuning.scenario_qa import CAVEAT, ScenarioQaBuilder

C379 = Chunk("IPC:379:a", "IPC", "Section 379. Punishment for theft\nWhoever commits theft shall be punished with "
             "imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
             "IPC > Chapter XVII > Section 379", section="379", title="Punishment for theft")
C378 = Chunk("IPC:378:a", "IPC", "Section 378. Theft\nWhoever, intending to take dishonestly any movable property out of "
             "the possession of any person without that person's consent, moves that property, is said to commit theft.",
             "IPC > Chapter XVII > Section 378", section="378", title="Theft")
C323 = Chunk("IPC:323:a", "IPC", "Section 323. Punishment for voluntarily causing hurt\nWhoever voluntarily causes hurt "
             "shall be punished with imprisonment.", "IPC > Chapter XVI > Section 323", section="323", title="Punishment for voluntarily causing hurt")


def _src(c: Chunk, marker: int) -> SourceEvidence:
    return SourceEvidence(marker=marker, chunk_id=c.chunk_id, doc_id="IPC", citation_path=c.citation_path, text=c.text,
                          similarity=0.6, score=0.6, role="Law", section=c.section, title=c.title)


class _Retriever:
    def __init__(self, sources):
        self.sources = sources

    def retrieve(self, question, rewrite=None):
        return self.sources


def test_answer_cites_the_first_labelled_section_and_points_to_a_second() -> None:
    b = ScenarioQaBuilder(_Retriever([]), [C379, C378, C323])
    text = b.answer([_src(C323, 1), _src(C379, 2), _src(C378, 3)], ["378", "379"])
    assert text.startswith("The facts appear to fit punishment for theft under Section 379 of the IPC [2]")
    assert "three years" in text and CAVEAT in text and "See also Section 378 (Theft) [3]" in text


def test_no_labelled_section_among_sources_gives_no_example() -> None:
    b = ScenarioQaBuilder(_Retriever([_src(C323, 1)]), [C323])
    assert b.answer([_src(C323, 1)], ["379"]) is None
    assert b.build([{"id": "x", "question": "My phone was taken", "acceptable": ["379"]}]) == []


def test_build_uses_the_inference_prompt_and_tags_the_case() -> None:
    b = ScenarioQaBuilder(_Retriever([_src(C379, 1)]), [C379])
    [ex] = b.build([{"id": "sc-9", "question": "My phone was taken from my bag", "acceptable": ["379"]}],
                   rewrite_for=lambda q: object())  # same sources with the rewrite: one example, not two
    assert ex.source_case_id == "scenario:sc-9" and ex.task == "scenario_answer"
    assert "QUESTION: My phone was taken from my bag" in ex.instruction and "[1]" in ex.output
