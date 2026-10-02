"""LegalDocumentParser on real IPC text.

`fixtures/ipc_excerpt.txt` is a committed excerpt (sections 302-304B and 378) of the official
bare act as extracted from the India Code PDF. The full-PDF test runs only when the PDF is
present in data/raw_corpus/ (it is git-ignored).
"""

import re
from pathlib import Path

import pytest

from app.config import REPO_ROOT
from app.core.models import Chunk
from app.core.parser import LegalDocumentParser

FIXTURE = Path(__file__).parent / "fixtures" / "ipc_excerpt.txt"
RAW_CORPUS = REPO_ROOT / "data" / "raw_corpus"


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    text = "\n".join(l for l in FIXTURE.read_text(encoding="utf-8").splitlines() if not l.startswith("#"))
    parser = LegalDocumentParser(doc_id="IPC")
    return parser.chunk(parser.parse_text(text))


def by_path(chunks: list[Chunk], path: str) -> Chunk:
    matches = [c for c in chunks if c.citation_path == path]
    assert len(matches) == 1, f"expected one chunk at {path!r}, got {[c.citation_path for c in chunks]}"
    return matches[0]


def test_section_302_murder(chunks: list[Chunk]) -> None:
    c = by_path(chunks, "IPC > Chapter XVI > Section 302")
    assert c.section == "302" and c.chapter == "XVI" and c.title == "Punishment for murder"
    assert c.text == (
        "Section 302. Punishment for murder\n"
        "Whoever commits murder shall be punished with death or imprisonment for life, "
        "and shall also be liable to fine."
    )


def test_section_304a_is_separate_from_its_state_amendment(chunks: list[Chunk]) -> None:
    c = by_path(chunks, "IPC > Chapter XVI > Section 304A")
    assert c.text == (
        "Section 304A. Causing death by negligence\n"
        "Whoever causes the death of any person by doing any rash or negligent act not amounting to "
        "culpable homicide, shall be punished with imprisonment of either description for a term which "
        "may extend to two years, or with fine, or with both."
    )
    amendments = [c for c in chunks if c.citation_path.startswith("IPC > Chapter XVI > Section 304A > State Amendments")]
    assert amendments and all(c.section == "304A" for c in amendments)
    assert "Himachal Pradesh" in amendments[0].text


def test_section_304b_subsections_stay_in_one_chunk(chunks: list[Chunk]) -> None:
    c = by_path(chunks, "IPC > Chapter XVI > Section 304B")
    assert "\n(1) Where the death of a woman is caused" in c.text
    assert "\nExplanation.— For the purposes of this sub-section" in c.text
    assert "\n(2) Whoever commits dowry death shall be punished" in c.text


def test_section_378_theft_in_next_chapter_with_illustrations(chunks: list[Chunk]) -> None:
    theft = [c for c in chunks if c.section == "378"]
    assert theft[0].citation_path == "IPC > Chapter XVII > Section 378"
    assert theft[0].text.startswith("Section 378. Theft\nWhoever, intending to take dishonestly any movable property")
    assert all(c.citation_path.startswith("IPC > Chapter XVII > Section 378") for c in theft)
    assert any("(a) A cuts down a tree on Z's ground" in c.text for c in theft)


def test_no_amendment_markers_or_footnotes_leak(chunks: list[Chunk]) -> None:
    for c in chunks:
        assert not re.search(r"[\[\]*]", c.text), c.citation_path
        assert not re.search(r"\b(Subs|Ins)\. by Act", c.text), c.citation_path
        assert len(c.text) <= 1400


@pytest.mark.skipif(not any(RAW_CORPUS.glob("*.pdf")), reason="IPC PDF not in data/raw_corpus/")
def test_full_pdf_every_listed_section_is_parsed() -> None:
    pdf = next(RAW_CORPUS.glob("*.pdf"))
    parser = LegalDocumentParser(doc_id="IPC")
    text = parser.extract_pdf_text(pdf)
    nodes = parser.parse_text(text)
    toc = parser.toc_sections(text)
    found = {n.section for n in nodes if n.kind == "section"}
    assert len(toc) > 500
    assert [s for s in toc if s not in found] == []
    chunks = parser.chunk(nodes)
    assert by_path(chunks, "IPC > Chapter XVI > Section 302").text.startswith("Section 302. Punishment for murder")
    assert all(len(c.text) <= 1400 for c in chunks)
    assert not any("This Bare Act" in c.text or re.search(r"\b(Subs|Ins)\. by Act", c.text) for c in chunks
                   if "State Amendments" not in c.citation_path)
