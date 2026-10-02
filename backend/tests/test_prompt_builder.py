from app.core.models import Message, SourceEvidence
from app.core.prompt_builder import NOT_COVERED_REPLY, PromptBuilder


def _ev(marker: int, path: str, text: str) -> SourceEvidence:
    return SourceEvidence(
        marker=marker, chunk_id=f"c{marker}", doc_id="IPC", citation_path=path, text=text,
        similarity=0.8, score=0.9, role="Law",
    )


def test_legal_prompt_numbers_sources_in_order_and_sets_rules() -> None:
    sources = [_ev(1, "IPC > Section 304A", "Causing death by negligence."), _ev(2, "IPC > Section 299", "Culpable homicide.")]
    msgs = PromptBuilder().build("What is 304A?", sources, "legal")
    system, user = msgs[0], msgs[-1]
    assert system.role == "system"
    assert "ONLY" in system.content and "[1]" in system.content and NOT_COVERED_REPLY in system.content
    assert user.content.index("[1] (IPC > Section 304A)") < user.content.index("[2] (IPC > Section 299)")
    assert user.content.endswith("QUESTION: What is 304A?")


def test_general_prompt_has_no_sources_and_keeps_recent_history() -> None:
    history = [Message("user", f"q{i}") for i in range(10)]
    msgs = PromptBuilder(history_turns=4).build("hello", [], "general", history)
    assert "SOURCES" not in msgs[-1].content
    assert [m.content for m in msgs[1:-1]] == ["q6", "q7", "q8", "q9"]
