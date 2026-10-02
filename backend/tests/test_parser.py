"""LegalDocumentParser mechanics on a synthetic bare-act-style document.

(Real IPC sections are checked in test_parser_ipc.py against an excerpt of the official text.)
"""

import re

import pytest

from app.core.parser import LegalDocumentParser

SYNTHETIC_ACT = """THE SAMPLE CODE
ARRANGEMENT OF SECTIONS
CHAPTER I
PRELIMINARY
SECTIONS
1. Short title.
2. Definitions.

CHAPTER I
PRELIMINARY
1. Short title.—This Act may be called the Sample Code.
2. Definitions.—In this Code, unless the context otherwise requires,—
(a) "vehicle" means any mechanically propelled conveyance;
(b) "road" includes any bridge or causeway.
CHAPTER IIA
OF OFFENCES
12A. Rash driving.—(1) Whoever drives a vehicle rashly on a road shall be punished with fine.
(2) Whoever repeats the offence shall be punished with imprisonment which may extend to one
year.
Explanation.—A person drives rashly when he ignores a signal.
Illustrations
(a) A drives through a red light. A has committed an offence under this section.
(b) A drives within the speed limit. A has not committed an offence.
"""


@pytest.fixture
def parser() -> LegalDocumentParser:
    return LegalDocumentParser(doc_id="SC")


def test_table_of_contents_is_ignored_and_sections_found(parser: LegalDocumentParser) -> None:
    nodes = parser.parse_text(SYNTHETIC_ACT)
    assert [n.section for n in nodes] == ["1", "2", "12A"]
    assert nodes[0].text == "This Act may be called the Sample Code."
    assert nodes[0].title == "Short title"


def test_hierarchy_and_citation_paths(parser: LegalDocumentParser) -> None:
    s12a = parser.parse_text(SYNTHETIC_ACT)[2]
    assert s12a.citation_path == "SC > Chapter IIA > Section 12A"
    labels = [c.label for c in s12a.children]
    # "(1)" sits on the heading line itself but still becomes a sub-section
    assert labels == ["(1)", "(2)", "Explanation", "Illustrations"]
    assert s12a.children[1].citation_path == "SC > Chapter IIA > Section 12A > (2)"
    # line-wrapped continuation is joined back into the sub-section
    assert s12a.children[1].text.endswith("may extend to one year.")
    illus = s12a.children[3]
    assert [c.citation_path for c in illus.children] == [
        "SC > Chapter IIA > Section 12A > Illustrations > (a)",
        "SC > Chapter IIA > Section 12A > Illustrations > (b)",
    ]
    clauses = parser.parse_text(SYNTHETIC_ACT)[1].children
    assert [c.citation_path for c in clauses] == ["SC > Chapter I > Section 2 > (a)", "SC > Chapter I > Section 2 > (b)"]


def test_short_section_is_one_chunk_with_section_path(parser: LegalDocumentParser) -> None:
    chunks = parser.chunk(parser.parse_text(SYNTHETIC_ACT))
    by_section = {c.section: c for c in chunks}
    assert by_section["1"].citation_path == "SC > Chapter I > Section 1"
    assert by_section["1"].text == "Section 1. Short title\nThis Act may be called the Sample Code."
    assert by_section["12A"].citation_path == "SC > Chapter IIA > Section 12A"
    assert "(a) A drives through a red light." in by_section["12A"].text
    assert {c.role for c in chunks} == {"Law"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def _long_section(n_clauses: int = 40) -> str:
    clauses = "\n".join(
        f"({chr(97 + i // 26) if i >= 26 else ''}{chr(97 + i % 26)}) Clause {i} describes conduct number {i} "
        f"in some detail so that it takes up space. It has a second sentence too."
        for i in range(n_clauses)
    )
    return f"CHAPTER V\nOF THINGS\n50. Long section.—Whoever does any of the following,—\n{clauses}\n"


def test_long_section_chunks_are_bounded_sentence_aligned_and_overlapping(parser: LegalDocumentParser) -> None:
    nodes = parser.parse_text(_long_section())
    chunks = parser.chunk(nodes, max_chunk_chars=600, overlap=150)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.text) <= 600
        assert c.text.startswith("Section 50. Long section\n")
        assert re.search(r"[.;:—,]$", c.text), f"chunk ends mid-sentence: {c.text[-40:]!r}"
        assert c.citation_path.startswith("SC > Chapter V > Section 50")
    # overlap: each chunk after the first repeats the previous chunk's last sentence
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.text.rsplit(". ", 1)[-1]
        assert last_sentence in nxt.text
    # a chunk spanning several clauses gets a sibling range in its path
    assert any(re.search(r"Section 50 > \(\w+\)–\(\w+\)$", c.citation_path) for c in chunks)


def test_every_sentence_survives_chunking(parser: LegalDocumentParser) -> None:
    nodes = parser.parse_text(_long_section())
    joined = "\n".join(c.text for c in parser.chunk(nodes, max_chunk_chars=600, overlap=150))
    for i in range(40):
        assert f"Clause {i} describes conduct number {i}" in joined


def test_fallback_to_paragraphs_without_section_markers(parser: LegalDocumentParser) -> None:
    text = "First paragraph about the facts.\nIt wraps a line.\n\nSecond paragraph about the ruling."
    nodes = parser.parse_text(text, doc_id="Judgment")
    assert [n.citation_path for n in nodes] == ["Judgment > ¶1", "Judgment > ¶2"]
    chunks = parser.chunk(nodes)
    assert [c.citation_path for c in chunks] == ["Judgment > ¶1", "Judgment > ¶2"]
    assert chunks[0].text == "First paragraph about the facts. It wraps a line."
