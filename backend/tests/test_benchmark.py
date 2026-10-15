"""The IPC benchmark: generated fact patterns, statute questions, answer scoring."""

import json
from pathlib import Path

import pytest

from app.benchmark.factgen import generate
from app.benchmark.metrics import cited_sections, refused, score_oos, score_scenario, score_statute
from app.benchmark.scenario_data import GROUPS, SCENARIOS
from app.benchmark.statute import build, terms_in
from app.config import REPO_ROOT, get_settings


# ------------------------------------------------------------------ fact patterns
@pytest.fixture(scope="module")
def cases():
    return generate()


def test_variant_kinds_have_the_verdict_the_rules_imply(cases) -> None:
    want = {"base": "CONSISTENT", "paraphrase": "CONSISTENT", "flip": "INCONSISTENT", "omit": "INSUFFICIENT"}
    for c in cases:
        assert c.gold_verdict == want[c.kind], c.id


def test_each_variant_changes_exactly_one_fact(cases) -> None:
    base = {c.group: c for c in cases if c.kind == "base"}
    for c in cases:
        b = base[c.group]
        diff = {k for k in b.facts if b.facts[k] != c.facts.get(k)}
        if c.kind in ("base", "paraphrase"):
            assert diff == set() and (c.kind == "base" or c.text != b.text)
        else:
            assert diff == {c.change["predicate"]}, c.id


def test_ids_unique_and_every_section_covered(cases) -> None:
    assert len({c.id for c in cases}) == len(cases)
    assert {c.section for c in cases} == {"279", "304A", "304B", "323", "337", "338", "379"}


# ------------------------------------------------------------------ statute questions and scenarios
def test_statute_questions_have_gold_from_the_statute() -> None:
    qs = build(get_settings().corpus_chunks_path)
    by_id = {q.id: q for q in qs}
    assert by_id["pun-302"].gold_terms == ["death", "life"]
    assert by_id["pun-379"].gold_terms == ["three year"]
    assert "def-378" in by_id and len(qs) > 300


def test_scenario_groups_reference_real_sections() -> None:
    from app.core.corpus import load_chunks

    ipc = {c.section for c in load_chunks(get_settings().corpus_chunks_path) if c.section}
    assert all(s in ipc for sections in GROUPS.values() for s in sections)
    assert all(g in GROUPS for _, g in SCENARIOS) and len({q for q, _ in SCENARIOS}) == len(SCENARIOS)


def test_published_benchmark_matches_the_generators() -> None:
    p = REPO_ROOT / "benchmark" / "fact_patterns.jsonl"
    if not p.exists():
        pytest.skip("benchmark not built")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows if r["kind"] != "natural"] == [c.id for c in generate()], \
        "benchmark/ is stale: run backend/scripts/build_benchmark.py"


# ------------------------------------------------------------------ scoring
@pytest.mark.parametrize("text, ipc, other", [
    ("Theft is punishable under Section 379 of the IPC.", ["379"], []),
    ("Under Sections 302 and 34 IPC, and Section 103 of the BNS", ["302", "34"], ["103"]),
    ("This falls under s. 304-B IPC and Section 498A.", ["304B", "498A"], []),
    ("Section 154 CrPC requires an FIR; 420 IPC also applies.", ["420"], ["154"]),
    ("Section 25 of the Arms Act and Section 307 IPC", ["307"], ["25"]),
    ("Section 303(2) BNS (formerly Section 379 IPC)", ["379"], ["303"]),
])
def test_cited_sections(text: str, ipc: list[str], other: list[str]) -> None:
    got = cited_sections(text)
    assert got["ipc"] == ipc and got["other"] == other


def test_scoring_rules() -> None:
    item = {"kind": "punishment", "section": "379", "gold_terms": ["three year"]}
    assert score_statute(item, "Section 379 IPC: up to three years, or fine, or both.")["punishment_correct"]
    wrong = score_statute(item, "Under Section 379 IPC theft is punished with up to five years.")
    assert wrong["punishment_wrong"] and wrong["section_correct"]
    assert score_statute(item, "Under Section 303 IPC ...")["bns_confusion"] is True  # BNS number given as IPC
    assert score_statute(item, "x", extra_cited=["379"])["section_correct"]  # [n] citation of the source
    assert score_scenario({"acceptable": ["378", "379", "380"]}, "This is theft under Section 380 IPC.")["hit"]
    assert score_scenario({"acceptable": ["378", "379"]}, "This is theft under Section 380 IPC.")["wrong_only"]
    assert score_oos("You may file a complaint under Section 406 IPC.")["criminalised"]
    assert refused("I don't know — the retrieved sources do not cover this question.")
    assert terms_in("punishable with death or imprisonment for life, or 10 years") == {"death", "life", "ten year"}
    assert terms_in("causing death by negligence") == set()


def test_state_amendment_sections_are_known() -> None:
    """An answer citing an inserted section it was given (Gujarat's s.379A) is not an invented section."""
    from app.benchmark.metrics import known_sections
    from app.core.models import Chunk

    chunks = [Chunk("IPC:379:a", "IPC", "Whoever commits theft shall be punished.", "IPC > Section 379", section="379"),
              Chunk("IPC:379:sa", "IPC", "after section 379, the following sections shall be inserted:- 379A. Snatching.-(1) "
                    "Whoever ... 379B.Snatching after preparation ... punishable under section 509B", 
                    "IPC > Chapter XVII > Section 379 > State Amendments", section="379")]
    assert known_sections(chunks) == {"379", "379A", "379B", "509B"}
