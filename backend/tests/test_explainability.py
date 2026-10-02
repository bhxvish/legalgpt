"""Module 4: confidence bands, citation validation, attribution, explanation records, and the
explanation SSE event on /api/chat."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.chat_router import ChatRouter
from app.core.models import Chunk, SourceEvidence
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore
from app.explainability import confidence_scorer
from app.explainability.attribution_analyzer import AttributionAnalyzer
from app.explainability.citation_validator import CitationValidator
from app.explainability.confidence_scorer import BAND_NOTE, ConfidenceScorer
from app.explainability.explanation_builder import ExplanationBuilder
from tests.conftest import FakeEmbedder, StubLLM


def src(marker: int, text: str, section: str = "", similarity: float = 0.8) -> SourceEvidence:
    return SourceEvidence(marker=marker, chunk_id=f"c{marker}", doc_id="IPC", citation_path=f"IPC > Section {section or marker}",
                          text=text, similarity=similarity, score=similarity, role="Law", section=section)


NEGLIGENCE = src(1, "Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable "
                    "homicide shall be punished with imprisonment for two years or with fine or with both.", "304A", 0.81)
RASH_DRIVING = src(2, "Whoever drives any vehicle on a public way so rashly or negligently as to endanger human life "
                      "shall be punished with imprisonment for six months.", "279", 0.66)
SOURCES = [NEGLIGENCE, RASH_DRIVING]

GROUNDED = ("Causing the death of any person by a rash or negligent act is punished with imprisonment for two years or "
            "with fine or with both [1]. Driving a vehicle rashly on a public way so as to endanger human life is "
            "punished with imprisonment for six months [2].")
OFF_TOPIC = ("The appellant was granted anticipatory bail by the Sessions Judge [1]. Property disputes between "
             "brothers are settled by partition decrees in civil courts [2].")


# ----------------------------------------------------------- ConfidenceScorer


def test_similarity_normalizes_each_distance_space() -> None:
    s = ConfidenceScorer.similarity
    assert s(0.25, "cosine") == pytest.approx(0.75) and s(0.25, "ip") == pytest.approx(0.75)
    assert s(0.5, "l2") == pytest.approx(0.75)  # squared L2 of unit vectors = 2 - 2cos
    assert s(1.6, "cosine") == 0.0 and s(-0.1, "cosine") == 1.0  # clamped to [0, 1]
    with pytest.raises(ValueError):
        s(0.1, "manhattan")


def test_bands_and_thresholds() -> None:
    sc = ConfidenceScorer()
    assert [sc.band(x) for x in (0.85, 0.70, 0.69, 0.60, 0.59, 0.1)] == ["High", "High", "Medium", "Medium", "Low", "Low"]
    assert ConfidenceScorer(high=0.75, medium=0.5).band(0.6) == "Medium"
    with pytest.raises(ValueError):
        ConfidenceScorer(high=0.5, medium=0.7)


def test_confidence_is_documented_as_not_answer_correctness() -> None:
    doc = (confidence_scorer.__doc__ or "") + (ConfidenceScorer.band.__doc__ or "")
    assert "does NOT measure whether the *answer* is" in doc and "Not an accuracy guarantee" in doc
    assert "not a measure of whether the answer is correct" in BAND_NOTE


# ---------------------------------------------------------- CitationValidator


def test_validator_flags_marker_without_a_retrieved_source() -> None:
    malformed = "Death by negligence is punished with two years [1]. The court may also order compensation [7][1]."
    check = CitationValidator().validate(malformed, SOURCES)
    assert check.markers == [1, 7, 1]
    assert check.resolved == [1] and check.unresolved == [7] and check.uncited_sources == [2]
    assert not check.ok


def test_validator_canonicalizes_model_marker_variants() -> None:
    assert CitationValidator.extract_markers("two years 【1】 and six months [2†L4-L6]") == [1, 2]
    assert CitationValidator().validate(GROUNDED, SOURCES).ok
    assert not CitationValidator().validate("No citations here.", SOURCES).ok


# -------------------------------------------------------- AttributionAnalyzer


def test_support_ratio_lower_for_off_topic_answer() -> None:
    analyzer = AttributionAnalyzer(FakeEmbedder(), support_threshold=0.3)
    grounded = analyzer.attribute(GROUNDED, SOURCES)
    off = analyzer.attribute(OFF_TOPIC, SOURCES)
    assert analyzer.support_ratio(grounded) == 1.0
    assert analyzer.support_ratio(off) < analyzer.support_ratio(grounded)
    assert analyzer.support_ratio(off) == 0.0
    assert [a.source_marker for a in grounded] == [1, 2]  # each sentence matched to the source it restates
    assert "rash or negligent act" in grounded[0].evidence


def test_attribution_checks_the_cited_source_itself() -> None:
    analyzer = AttributionAnalyzer(FakeEmbedder(), support_threshold=0.45)
    swapped = ("Driving a vehicle rashly on a public way so as to endanger human life is punished with "
               "imprisonment for six months [1].")
    a = analyzer.attribute(swapped, SOURCES)[0]
    assert a.supported and a.source_marker == 2 and a.cited_markers == [1] and a.cited_supported is False


def test_split_sentences_strips_markup_and_fragments() -> None:
    text = "**Section 279** applies [1].\n\n* Rash driving endangering life is punishable with imprisonment [1].\n"
    assert AttributionAnalyzer.split_sentences(text) == ["Rash driving endangering life is punishable with imprisonment [1]."]


def _minilm_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _minilm_cached(), reason="MiniLM not in the local Hugging Face cache")
def test_support_ratio_with_the_real_embedder() -> None:
    from app.core.embeddings import EmbeddingService

    analyzer = AttributionAnalyzer(EmbeddingService())  # default threshold 0.55
    assert analyzer.support_ratio(analyzer.attribute(GROUNDED, SOURCES)) == 1.0
    assert analyzer.support_ratio(analyzer.attribute(OFF_TOPIC, SOURCES)) == 0.0


# --------------------------------------------------------- ExplanationBuilder


@pytest.fixture
def builder() -> ExplanationBuilder:
    return ExplanationBuilder(AttributionAnalyzer(FakeEmbedder(), support_threshold=0.3))


def test_record_for_grounded_answer(builder: ExplanationBuilder) -> None:
    rec = builder.build("Punishment for negligent death?", SOURCES, GROUNDED, "stub")
    assert rec.relevance == 0.81 and rec.band == "High" and rec.band_note == BAND_NOTE
    assert rec.support_ratio == 1.0 and rec.warnings == []
    assert [(s.marker, s.cited, s.supports) for s in rec.sources] == [(1, True, [0]), (2, True, [1])]


def test_record_flags_weak_answer(builder: ExplanationBuilder) -> None:
    rec = builder.build("Punishment for negligent death?", SOURCES, OFF_TOPIC + " See also [9].", "stub")
    assert any("[9]" in w and "not among the retrieved sources" in w for w in rec.warnings)
    assert any("not supported by any retrieved source" in w for w in rec.warnings)
    assert rec.citation_check.unresolved == [9]


def test_record_for_refusal_reports_best_candidate(builder: ExplanationBuilder) -> None:
    rec = builder.build("Best cake recipe?", [], "refused", "stub", best_similarity=0.31, retrieval_floor=0.55, refused=True)
    assert rec.refused and rec.relevance == 0.31 and rec.band == "Low" and rec.sources == []
    assert "0.31" in rec.warnings[0] and "0.55" in rec.warnings[0]


def test_sse_event_format(builder: ExplanationBuilder) -> None:
    event = builder.to_sse_event(builder.build("q", SOURCES, GROUNDED, "stub"))
    name, data = event.strip().split("\n", 1)
    assert name == "event: explanation" and json.loads(data.removeprefix("data: "))["band"] == "High"


# --------------------------------------------------- /api/chat end to end


@pytest.fixture
def client(fake_embedder: FakeEmbedder, memory_store: VectorStore) -> TestClient:
    chunk = Chunk(chunk_id="IPC:304A:x", doc_id="IPC", section="304A", title="Causing death by negligence",
                  citation_path="IPC > Chapter XVI > Section 304A", text=NEGLIGENCE.text)
    memory_store.upsert([chunk], fake_embedder.embed([chunk.text]))
    retriever = Retriever(fake_embedder, memory_store, min_similarity=0.2)
    llm = StubLLM(reply="Causing the death of any person by a rash or negligent act is punished with imprisonment "
                        "for two years or with fine or with both [1].")
    app = FastAPI()
    app.include_router(ChatRouter(retriever, llm, explainer=ExplanationBuilder(AttributionAnalyzer(fake_embedder, 0.3))).router)
    return TestClient(app)


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_legal_answer_carries_explanation(client: TestClient) -> None:
    events = _events(client.post("/api/chat", json={"question": "punishment for death by rash or negligent act"}).text)
    names = [e for e, _ in events]
    assert names.index("explanation") > max(i for i, n in enumerate(names) if n == "token")  # after the answer
    rec = dict(events)["explanation"]
    assert rec["relevance"] is not None and rec["band"] in ("High", "Medium", "Low")
    assert rec["support_ratio"] == 1.0 and rec["citation_check"]["unresolved"] == []


def test_refusal_also_carries_explanation(client: TestClient) -> None:
    rec = dict(_events(client.post("/api/chat", json={"question": "chocolate cake recipe"}).text))["explanation"]
    assert rec["refused"] is True and rec["relevance"] is not None and rec["band"] == "Low"


def test_general_mode_has_no_explanation(client: TestClient) -> None:
    names = [e for e, _ in _events(client.post("/api/chat", json={"question": "hi", "mode": "general"}).text)]
    assert "explanation" not in names


def test_cited_source_is_credited_when_it_supports_the_sentence() -> None:
    # sentence restates source 2 but both sources share vocabulary; it cites [2]
    near = src(1, "Whoever drives any vehicle on a public way so rashly as to endanger human life commits an offence.", "279A", 0.7)
    analyzer = AttributionAnalyzer(FakeEmbedder(), support_threshold=0.3)
    a = analyzer.attribute("Whoever drives any vehicle on a public way so rashly or negligently as to endanger human "
                           "life shall be punished with imprisonment for six months [2].", [near, RASH_DRIVING])[0]
    assert a.cited_supported and a.source_marker == 2


def test_not_covered_sentence_is_not_flagged_as_unsupported(builder: ExplanationBuilder) -> None:
    from app.core.prompt_builder import NOT_COVERED_REPLY

    answer = f"{NOT_COVERED_REPLY} " + GROUNDED.split(" [1].")[0] + " [1]."
    rec = builder.build("q", SOURCES, answer, "stub")
    assert rec.refused and len(rec.attributions) == 1 and rec.support_ratio == 1.0
    assert not any("not supported" in w for w in rec.warnings)


def test_refusal_despite_good_evidence_is_flagged(builder: ExplanationBuilder) -> None:
    from app.core.prompt_builder import NOT_COVERED_REPLY

    rec = builder.build("Punishment under section 304A?", SOURCES, NOT_COVERED_REPLY, "stub")
    assert rec.refused and rec.support_ratio is None and rec.attributions == []
    assert any("said the sources do not cover this" in w and "high evidence match" in w for w in rec.warnings)
