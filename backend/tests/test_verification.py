"""Module 5: Prolog rules (hand-verified per section), schema, mapper, extraction repair,
fact clean-up, the verification service and its API. Prolog tests need SWI-Prolog + pyswip."""

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import BACKEND_DIR
from app.core.llm_client import LLMClient
from app.core.models import Message
from app.verification.fact_extractor import ExtractionError, FactExtractor
from app.verification.fact_schema import BY_NAME, PREDICATES, FactSchema
from app.verification.predicate_mapper import PredicateMapper

RULES = BACKEND_DIR / "app" / "verification" / "rules"


def _prolog_available() -> bool:
    try:
        from pyswip import Prolog

        Prolog()
        return True
    except Exception:
        return False


needs_prolog = pytest.mark.skipif(not _prolog_available(), reason="SWI-Prolog / pyswip not available")


@pytest.fixture(scope="module")
def engine():
    if not _prolog_available():
        pytest.skip("SWI-Prolog / pyswip not available")
    from app.verification.prolog_engine import PrologEngine

    return PrologEngine(RULES)


def run(engine, case: str, section: str, **facts: str) -> tuple[str, dict[str, str]]:
    """Hand-assert facts (unspecified predicates stay absent = unknown), query, clean up."""
    clauses = [f"fact('{case}', {k}, {v})" for k, v in facts.items()]
    with engine.session(case, clauses):
        return engine.verdict(case, section), engine.element_statuses(case, section)


# ---------------------------------------------------------- rules: one block per section
# Each scenario states the legal analysis it encodes; the verdict must match it.


@needs_prolog
def test_304a_death_by_negligence(engine) -> None:
    # Negligent driver kills a pedestrian, no intention, no knowledge death was likely: s.304A made out.
    base = dict(caused_death="true", act_type="negligent", intent_to_kill="false", knowledge_likely_death="false")
    assert run(engine, "a1", "304A", **base)[0] == "CONSISTENT"
    # Intention to kill makes it culpable homicide, which s.304A expressly excludes.
    assert run(engine, "a2", "304A", **{**base, "intent_to_kill": "true"})[0] == "INCONSISTENT"
    # An intentional act is not a rash or negligent one.
    assert run(engine, "a3", "304A", **{**base, "act_type": "intentional"})[0] == "INCONSISTENT"
    # Facts silent on knowledge: the "not culpable homicide" element cannot be confirmed.
    verdict, statuses = run(engine, "a4", "304A", **{**base, "knowledge_likely_death": "unknown"})
    assert verdict == "INSUFFICIENT" and statuses["knowledge_likely_death"] == "missing"
    # Negation-as-failure trap: with intent simply not asserted, the verdict must NOT be consistent.
    assert run(engine, "a5", "304A", caused_death="true", act_type="rash", knowledge_likely_death="false")[0] == "INSUFFICIENT"


@needs_prolog
def test_279_rash_driving(engine) -> None:
    base = dict(drove_vehicle="true", on_public_way="true", act_type="rash", endangered_safety="true")
    assert run(engine, "b1", "279", **base)[0] == "CONSISTENT"
    # On a private farm road, not a public way: s.279 does not apply.
    assert run(engine, "b2", "279", **{**base, "on_public_way": "false"})[0] == "INCONSISTENT"
    assert run(engine, "b3", "279", **{**base, "endangered_safety": "unknown"})[0] == "INSUFFICIENT"


@needs_prolog
def test_337_and_338_hurt_by_rash_act(engine) -> None:
    hurt = dict(caused_hurt="true", act_type="negligent", endangered_safety="true")
    assert run(engine, "c1", "337", **hurt)[0] == "CONSISTENT"
    # Simple hurt only: s.338 needs grievous hurt.
    assert run(engine, "c2", "338", **hurt, caused_grievous_hurt="false")[0] == "INCONSISTENT"
    assert run(engine, "c3", "338", caused_grievous_hurt="true", act_type="rash", endangered_safety="true")[0] == "CONSISTENT"
    # A pure accident is neither rash nor negligent.
    assert run(engine, "c4", "337", **{**hurt, "act_type": "accidental"})[0] == "INCONSISTENT"


@needs_prolog
def test_323_voluntary_hurt(engine) -> None:
    assert run(engine, "d1", "323", caused_hurt="true", act_type="intentional")[0] == "CONSISTENT"
    # Hurt caused negligently is s.337 territory, not voluntary hurt.
    assert run(engine, "d2", "323", caused_hurt="true", act_type="negligent")[0] == "INCONSISTENT"


@needs_prolog
def test_379_theft(engine) -> None:
    base = dict(movable_property="true", taken_from_possession="true", without_consent="true",
                dishonest_intention="true", property_moved="true")
    assert run(engine, "e1", "379", **base)[0] == "CONSISTENT"
    # The owner consented: no theft (s.378 requires taking without consent).
    assert run(engine, "e2", "379", **{**base, "without_consent": "false"})[0] == "INCONSISTENT"
    # Taken under a bona fide claim of right: no dishonest intention.
    assert run(engine, "e3", "379", **{**base, "dishonest_intention": "false"})[0] == "INCONSISTENT"
    # Land is not movable property (s.378 Explanation 1).
    assert run(engine, "e4", "379", **{**base, "movable_property": "false"})[0] == "INCONSISTENT"


@needs_prolog
def test_304b_dowry_death(engine) -> None:
    base = dict(woman_unnatural_death="true", within_seven_years_of_marriage="true", cruelty_by_husband_or_relative="true",
                cruelty_soon_before_death="true", dowry_demand="true")
    assert run(engine, "f1", "304B", **base)[0] == "CONSISTENT"
    # Death nine years after the marriage: outside s.304B.
    assert run(engine, "f2", "304B", **{**base, "within_seven_years_of_marriage": "false"})[0] == "INCONSISTENT"
    # Cruelty proved, but no link to any dowry demand shown: cannot conclude.
    verdict, statuses = run(engine, "f3", "304B", **{**base, "dowry_demand": "unknown"})
    assert verdict == "INSUFFICIENT" and statuses["dowry_demand"] == "missing"
    assert engine.prolog is not None and list(engine.prolog.query("dowry_death('f1')")) == []  # f1 already cleaned up


@needs_prolog
def test_named_section_predicates(engine) -> None:
    with engine.session("g1", ["fact('g1', caused_death, true)", "fact('g1', act_type, rash)",
                               "fact('g1', intent_to_kill, false)", "fact('g1', knowledge_likely_death, false)"]):
        assert list(engine.prolog.query("negligent_death('g1')"))
        assert not list(engine.prolog.query("theft('g1')"))


@needs_prolog
def test_schema_and_rules_use_the_same_predicates(engine) -> None:
    used = {e.predicate for s in engine.sections() for e in s.elements}
    assert used == set(BY_NAME), f"rules-only: {used - set(BY_NAME)}, schema-only: {set(BY_NAME) - used}"
    assert len(engine.sections()) >= 5


# ---------------------------------------------------------------- schema and mapper

CASE = "The accused drove a truck rashly on the highway and killed a cyclist. He did not intend any harm."


def test_schema_validation() -> None:
    facts = {
        "caused_death": {"value": "true", "evidence": "killed a cyclist"},
        "act_type": {"value": "rash", "evidence": "drove a truck rashly"},
        "on_public_way": {"value": "true", "evidence": "on the motorway"},  # not in the text
        "intent_to_kill": {"value": "FALSE", "evidence": "He did not intend any harm."},
    }
    r = FactSchema().validate(facts, CASE)
    assert r.ok and r.facts["caused_death"]["value"] == "true" and r.facts["intent_to_kill"]["value"] == "false"
    assert r.facts["on_public_way"]["value"] == "unknown"  # ungrounded claim cannot decide a verdict
    assert any("on_public_way" in w and "not found" in w for w in r.warnings)
    assert r.facts["dowry_demand"] == {"value": "unknown", "evidence": ""}  # missing -> unknown
    bad = FactSchema().validate({"act_type": {"value": "reckless"}, "made_up": {"value": "true"}}, CASE)
    assert not bad.ok and any("made_up" in e for e in bad.errors) and any("reckless" in e for e in bad.errors)
    assert not FactSchema().validate(["not", "a", "dict"]).ok


def test_mapper_emits_fact3_and_rejects_injection() -> None:
    facts = {"caused_death": {"value": "true", "evidence": "x"}, "act_type": {"value": "rash", "evidence": "y"}}
    assert PredicateMapper.to_prolog_facts("v_1", facts) == ["fact('v_1', caused_death, true)", "fact('v_1', act_type, rash)"]
    with pytest.raises(ValueError):
        PredicateMapper.to_prolog_facts("x'), halt, ('", facts)
    with pytest.raises(ValueError):
        PredicateMapper.to_prolog_facts("v_1", {"caused_death": {"value": "true), halt, (true", "evidence": ""}})


# ------------------------------------------------------------- extraction repair path


class ScriptedLLM(LLMClient):
    """Returns the scripted replies in order and records every call."""

    model_id = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self.replies.pop(0)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield self.generate(messages)


VALID = json.dumps({
    "caused_death": {"value": "true", "evidence": "killed a cyclist"},
    "act_type": {"value": "rash", "evidence": "drove a truck rashly"},
    "intent_to_kill": {"value": "false", "evidence": "He did not intend any harm"},
    "drove_vehicle": {"value": "true", "evidence": "drove a truck"},
    "on_public_way": {"value": "true", "evidence": "on the highway"},
})


def test_extract_repairs_malformed_json_once() -> None:
    llm = ScriptedLLM(['{"caused_death": {"value": "true", "evidence": "killed a cyclist"}, "act_type": {', VALID])
    extractor = FactExtractor(llm)
    result = extractor.extract(CASE)
    assert result.ok and extractor.attempts == 2 and len(llm.calls) == 2
    assert result.facts["act_type"]["value"] == "rash"
    assert "not valid JSON" in llm.calls[1][-1].content  # the retry tells the model what was wrong
    assert llm.calls[1][-2].role == "assistant"  # ... and shows it its own broken reply
    assert any("repaired" in w for w in result.warnings)


def test_extract_repairs_schema_violation_and_tolerates_fences() -> None:
    llm = ScriptedLLM([json.dumps({"act_type": {"value": "reckless", "evidence": "x"}}), f"```json\n{VALID}\n```"])
    result = FactExtractor(llm).extract(CASE)
    assert result.ok and "reckless" in llm.calls[1][-1].content


def test_extract_gives_up_after_one_repair() -> None:
    with pytest.raises(ExtractionError):
        FactExtractor(ScriptedLLM(["not json", "still not json"])).extract(CASE)


def test_valid_first_reply_needs_no_repair() -> None:
    llm = ScriptedLLM([VALID])
    extractor = FactExtractor(llm)
    assert extractor.extract(CASE).ok and extractor.attempts == 1
    prompt = llm.calls[0]
    assert prompt[0].role == "system" and all(p.name in prompt[0].content for p in PREDICATES)
    assert sum(m.role == "assistant" for m in prompt) == 4  # few-shot examples


# ---------------------------------------------- service: clean-up, leakage, API


@pytest.fixture
def service(engine):
    from app.verification.verification_service import VerificationService

    return VerificationService(FactExtractor(ScriptedLLM([])), engine, extraction_runs=1)


def _all_facts(engine) -> list:
    return list(engine.prolog.query("fact(C, P, V)"))


@needs_prolog
def test_retract_runs_even_when_verdict_raises(service, engine, monkeypatch) -> None:
    service.extractor.llm.replies = [VALID]

    def boom(*args, **kwargs):
        assert _all_facts(engine), "facts should be asserted while the verdict runs"
        raise RuntimeError("query failed")

    monkeypatch.setattr(engine, "verdict", boom)
    with pytest.raises(RuntimeError):
        service.verify(CASE, "304A")
    assert _all_facts(engine) == []


@needs_prolog
def test_no_fact_leakage_between_sequential_verifications(service, engine) -> None:
    theft = json.dumps({k: {"value": "true", "evidence": "took the bicycle"} for k in
                        ("movable_property", "taken_from_possession", "without_consent", "dishonest_intention", "property_moved")})
    service.extractor.llm.replies = [VALID, theft]
    first = service.verify(CASE, "304A")
    assert _all_facts(engine) == []
    second = service.verify("PW-2 saw the accused as he took the bicycle from the stand and rode away.", "304A")
    assert _all_facts(engine) == []
    assert first.case_id != second.case_id
    assert first.facts["caused_death"]["value"] == "true"
    assert second.facts["caused_death"]["value"] == "unknown"  # nothing carried over from the first case
    assert second.status == "INSUFFICIENT"
    assert [a.section for a in second.alternatives][0] == "379" and second.alternatives[0].status == "CONSISTENT"


@needs_prolog
def test_verify_api(service) -> None:
    from app.verification.router import VerifyRouter

    service.extractor.llm.replies = [VALID]
    app = FastAPI()
    app.include_router(VerifyRouter(service).router)
    client = TestClient(app)
    sections = client.get("/api/verify/sections").json()["sections"]
    assert [s["section"] for s in sections] == ["279", "304A", "304B", "323", "337", "338", "379"]
    r = client.post("/api/verify", json={"case_text": CASE, "cited_section": "304a"}).json()
    assert r["cited_section"] == "304A" and r["status"] == "INSUFFICIENT"
    by_pred = {e["predicate"]: e for e in r["elements"]}
    assert by_pred["caused_death"]["status"] == "satisfied" and by_pred["caused_death"]["evidence"] == "killed a cyclist"
    assert by_pred["knowledge_likely_death"]["status"] == "missing"
    assert client.post("/api/verify", json={"case_text": CASE, "cited_section": "999"}).status_code == 422


@needs_prolog
def test_bound_verdict_query_cannot_skip_the_checklist(engine) -> None:
    """Regression: verdict(C, S, consistent) used to succeed for any section with no facts at all."""
    assert list(engine.prolog.query("verdict('nobody', '379', consistent)")) == []
    assert list(engine.prolog.query("offence('nobody', '304A')")) == []
    assert [r["V"] for r in engine.prolog.query("verdict('nobody', '379', V)")] == ["insufficient"]


def test_elided_evidence_quotes_are_accepted_in_order_only() -> None:
    from app.verification.fact_schema import _norm, quote_in_text

    text = _norm("Bal Krishan, who was standing near the shop, suffered multiple injuries on his head.")
    assert quote_in_text("Bal Krishan ... suffered multiple injuries", text)
    assert quote_in_text("Bal Krishan … suffered multiple injuries", text)
    assert not quote_in_text("suffered multiple injuries ... Bal Krishan", text)  # wrong order
    assert not quote_in_text("Bal Krishan ... died", text)


def test_consistent_extraction_drops_values_the_runs_disagree_on() -> None:
    other = json.loads(VALID)
    other["on_public_way"] = {"value": "unknown", "evidence": ""}
    other["act_type"] = {"value": "negligent", "evidence": "drove a truck rashly"}
    llm = ScriptedLLM([VALID, json.dumps(other)])
    result = FactExtractor(llm).extract_consistent(CASE, runs=2)
    assert len(llm.calls) == 2
    assert result.facts["on_public_way"]["value"] == "unknown"  # true vs unknown: no agreement
    assert any("on_public_way" in w and "disagreed" in w for w in result.warnings)
    assert result.facts["caused_death"]["value"] == "true"  # both runs agree
    assert result.facts["act_type"]["value"] == "rash"  # rash vs negligent: either satisfies every rule
