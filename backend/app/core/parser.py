"""LegalDocumentParser: bare-act text -> hierarchical Nodes -> bounded, citation-tagged Chunks."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.core.models import Chunk, Node
from app.core.text_utils import split_oversized, split_sentences

# "CHAPTER XVI", "CHAPTER VA", "CHAPTER IXA"
_CHAPTER = re.compile(r"^CHAPTER\s+([IVXLC]+[A-Z]?)\b\.?\s*(.*)$")
# "304A. Causing death by negligence.—Whoever ..." — the title must be followed by a dash,
# which separates real section headings from table-of-contents entries and footnotes.
_SECTION = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s*(?=[A-Z“\"‘'(])(.{2,250}?)\.?\s*[—–]+\s*(.*)$")
# Looser form seen in the India Code text ("17 “Government”.—"), trusted only for numbers that
# appear in the table of contents; the title must still start like a heading.
_SECTION_LOOSE = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.?\s+(?=[A-Z“\"‘'(])(.{2,250}?)\.?\s*[—–]+\s*(.*)$")
_SUBSECTION = re.compile(r"^\((\d{1,2}[A-Z]?)\)\s*(.*)$")
_CLAUSE = re.compile(r"^\(([a-z]{1,2}|[ivx]{1,5})\)\s*(.*)$")
_BLOCK = re.compile(r"^(Explanations?(?:\s*\d+)?|Illustrations?|Exceptions?(?:\s*\d+)?)\.?\s*[—–.:-]*\s*(.*)$")
_ALL_CAPS = re.compile(r"^[A-Z][A-Z ,'&()-]+$")
# Italic sub-headings inside a chapter, e.g. "Of offences affecting life", "Of theft".
_SUBHEADING = re.compile(r"^Of [A-Za-z ,'()-]{2,120}$")
_STATE_AMENDMENTS = re.compile(r"^STATE AMENDMENTS?\b", re.IGNORECASE)
_PAGE_FOOTER = re.compile(r"This Bare Act is a government source", re.IGNORECASE)
# First line of an amendment footnote: "1. Subs. by Act 26 of 1955, s. 117 ...", "3. Ins. by ...".
_FOOTNOTE = re.compile(
    # "13. [Omitted.]" (contents) and "61. [Sentence of ...] Rep. by ..." (body) are sections, not notes.
    r"^\d{1,2}\.(?!\s*\[).*?(\b(?:Subs|Ins|Omitted|Rep|Added|Renumbered|Earlier|Inserted|Substituted|Repealed)\b"
    r"|\bby\s+(?:the\s+)?(?:Act|Ord|A\.\s?O\.|Adaptation|Regulation|Order)\b|w\.e\.f\.)"
)
# A footnote line by how it opens ("1. Subs. by ...", "2. The words ... omitted", "1. Ins. by ...").
# Section titles never open this way, so such a line is never body text, wherever it lands.
_FOOTNOTE_START = re.compile(
    r"^\d{1,2}\.\s*(?:Subs\.|Ins\.|Omitted|Rep\.|The\s|Added|Certain\s+words|Cl\.|Sub-|Ss?\.\s|Now\b|Earlier|"
    r"Original|Renumbered|For\s|See\s|Vide\b|In\s)"
)
# Table-of-contents entry ("304A. Causing death by negligence.") — no dash, unlike a real heading.
_TOC_ENTRY = re.compile(r"^(\d{1,3}[A-Z]{0,2})\.\s*(?=[A-Z“\"‘'(\[])")
# Leftover amendment brackets ("[304A. ...]", "[India]") and omission asterisks ("* * *").
_AMENDMENT_MARKS = re.compile(r"[\[\]*]")
# A repealed/omitted section listed in the body without a heading dash:
# "61. Sentence of forfeiture of property. Rep. by the Indian Penal Code (Amendment) Act, 1921 ..."
_REPEALED = re.compile(r"\bRep\.|\b(?:Repealed|Omitted|omitted)\b")


@dataclass
class _Line:
    text: str
    blank_before: bool


class LegalDocumentParser:
    """Parses bare-act style legal text into Section > Sub-section > Clause nodes.

    Explanation / Illustration / Exception blocks become first-level children of their
    section (clauses inside them nest under the block). Text that does not contain any
    section headings falls back to paragraph-level nodes.
    """

    def __init__(self, doc_id: str = "IPC") -> None:
        self.doc_id = doc_id

    # ---------------------------------------------------------------- parsing

    def parse_pdf(self, path: str | Path, doc_id: str | None = None) -> list[Node]:
        return self.parse_text(self.extract_pdf_text(path), doc_id=doc_id)

    @staticmethod
    def extract_pdf_text(path: str | Path) -> str:
        """PDF -> body text in reading order, with page furniture removed.

        - Lines are re-sorted top-to-bottom per page (pdfminer's box order can put a
          heading after the text beneath it).
        - Superscript footnote numbers (the "1" in "1[304A.") are dropped: characters much
          smaller than the rest of their line.
        - Amendment footnotes are dropped: the run of small-font lines at the bottom of a
          page, from the first line that looks like "1. Subs. by Act ..." downwards.
          (Illustrations share the footnote font size, so size alone is not enough.)
        - Bare page numbers and the "This Bare Act is ..." page footer are dropped.
        """
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTChar, LTTextContainer, LTTextLine

        pages: list[list[tuple[float, float, float, str]]] = []  # (y, x, size, text)
        size_counts: Counter[float] = Counter()
        for page in extract_pages(str(path)):
            lines: list[tuple[float, float, float, str]] = []
            for box in page:
                if not isinstance(box, LTTextContainer):
                    continue
                for line in box:
                    if not isinstance(line, LTTextLine):
                        continue
                    sizes = Counter(round(o.size, 1) for o in line if isinstance(o, LTChar))
                    if not sizes:
                        continue
                    size = sizes.most_common(1)[0][0]
                    size_counts[size] += sum(sizes.values())
                    text = "".join(
                        o.get_text() for o in line if not isinstance(o, LTChar) or o.size >= size - 1.5
                    ).strip()
                    if text and not text.isdigit() and not _PAGE_FOOTER.search(text):
                        lines.append((round(line.y0, 0), line.x0, size, text))
            lines.sort(key=lambda ln: (-ln[0], ln[1]))
            # Pieces on the same baseline (e.g. "1." and its title in separate boxes) form one line.
            merged: list[tuple[float, float, float, str]] = []
            for ln in lines:
                if merged and abs(merged[-1][0] - ln[0]) <= 2:
                    y, x, size, text = merged[-1]
                    merged[-1] = (y, x, max(size, ln[2]), f"{text} {ln[3]}")
                else:
                    merged.append(ln)
            pages.append(merged)
        if not size_counts:
            return ""
        body = size_counts.most_common(1)[0][0]

        out: list[str] = []
        for lines in pages:
            # Bottom run of small-font lines; footnotes start at the first footnote-looking line.
            run_start = len(lines)
            while run_start > 0 and lines[run_start - 1][2] < body - 0.5:
                run_start -= 1
            cut = next(
                (i for i in range(run_start, len(lines)) if _FOOTNOTE.match(lines[i][3]) or _FOOTNOTE_START.match(lines[i][3])),
                len(lines),
            )
            out.extend(text for _, _, _, text in lines[:cut])
            out.append("")  # page break
        return "\n".join(out)

    def parse_file(self, path: str | Path, doc_id: str | None = None) -> list[Node]:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            return self.parse_pdf(path, doc_id=doc_id)
        return self.parse_text(path.read_text(encoding="utf-8"), doc_id=doc_id)

    def parse_text(self, text: str, doc_id: str | None = None) -> list[Node]:
        doc_id = doc_id or self.doc_id
        lines = self._lines(text)
        if not any(_SECTION.match(line.text) for line in lines):
            return self._parse_paragraphs(text, doc_id)
        return self._parse_sections(lines, doc_id)

    def _parse_sections(self, lines: list[_Line], doc_id: str) -> list[Node]:
        sections: list[Node] = []
        chapter = ""
        in_chapter_title = False
        section: Node | None = None  # node receiving sub-nodes (a section, or its state amendments)
        level1: Node | None = None  # current sub-section or Explanation/Illustration block
        current: Node | None = None  # node receiving continuation text
        in_amendment = False
        # Section numbers listed in the table of contents (before the first real heading) are
        # the authoritative set of central sections.
        toc, body_start = self._split_toc(lines)
        # Chapter heading lines just above the first body section still apply to it.
        while body_start > 0 and not _TOC_ENTRY.match(lines[body_start - 1].text):
            body_start -= 1
        toc_set = set(toc)

        for line in self._merge_wrapped_headings(lines[body_start:], toc_set):
            text = line.text
            if m := _CHAPTER.match(text):
                chapter = m.group(1)
                in_chapter_title = not m.group(2)
                section = level1 = current = None
                in_amendment = False
                continue
            if in_chapter_title and _ALL_CAPS.match(text):
                continue  # chapter titles can wrap over several all-caps lines
            in_chapter_title = False
            if _SUBHEADING.match(text):
                continue  # "Of offences affecting life" — a heading, not section text
            if _FOOTNOTE_START.match(text):
                continue  # an amendment footnote that the PDF extraction did not catch
            if section is not None and _STATE_AMENDMENTS.match(text):
                # State-specific amendments become their own top-level node so they are never
                # chunked together with (and mistaken for) the central text of the section.
                base = next(n for n in reversed(sections) if n.kind == "section")
                section = Node(
                    "paragraph",
                    "State Amendments",
                    f"{base.citation_path} > State Amendments",
                    title=f"State amendments to Section {base.section}",
                    chapter=base.chapter,
                    section=base.section,
                )
                sections.append(section)
                level1, current = None, section
                in_amendment = True
                continue
            m = self._heading(text, toc_set)
            if m and in_amendment and toc_set and m.group(1) not in toc_set:
                m = None  # a state-inserted section (e.g. 376F) is amendment text, not central law
            if not m and (r := _TOC_ENTRY.match(text)) and r.group(1) in toc_set and _REPEALED.search(text):
                # Repealed section: its own small node, so its note doesn't run into its neighbour.
                number = r.group(1)
                section = Node(
                    "section", number, self._path(doc_id, chapter, f"Section {number}"),
                    text[r.end():].strip(), "Repealed", chapter, number,
                )
                sections.append(section)
                level1, current, in_amendment = None, section, False
                continue
            if m:
                in_amendment = False
                number, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
                path = self._path(doc_id, chapter, f"Section {number}")
                section = Node("section", number, path, "", title, chapter, number)
                sections.append(section)
                level1, current = None, section
                if sub := _SUBSECTION.match(body):  # "304B. Dowry death.—(1) Where ..."
                    level1 = current = self._child(section, "subsection", f"({sub.group(1)})", sub.group(2))
                else:
                    section.text = body
                continue
            if section is None:
                continue  # preamble / table of contents before the first real section

            if m := _BLOCK.match(text):
                label = re.sub(r"\s+", " ", m.group(1)).strip()
                level1 = self._child(section, "paragraph", label, m.group(2))
                current = level1
            elif m := _SUBSECTION.match(text):
                level1 = self._child(section, "subsection", f"({m.group(1)})", m.group(2))
                current = level1
            elif m := _CLAUSE.match(text):
                current = self._child(level1 or section, "clause", f"({m.group(1)})", m.group(2))
            else:
                current = current or section
                if current.text.endswith("-") and text[:1].islower():  # word hyphenated across lines
                    current.text = current.text[:-1] + text
                else:
                    current.text = f"{current.text} {text}".strip()
        return sections

    @staticmethod
    def _heading(text: str, toc: set[str]) -> "re.Match[str] | None":
        if _FOOTNOTE_START.match(text):
            return None
        if m := _SECTION.match(text):
            return m
        m = _SECTION_LOOSE.match(text)
        return m if m and m.group(1) in toc else None

    @classmethod
    def _merge_wrapped_headings(cls, lines: list[_Line], toc: set[str]) -> list[_Line]:
        """Rejoin a long heading that wraps before its dash ("75. Enhanced punishment ... after" /
        "... previous conviction.—Whoever ..."). Only for numbers listed in the table of contents."""
        out: list[_Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _TOC_ENTRY.match(line.text)
            if m and m.group(1) in toc and not cls._heading(line.text, toc) and not _REPEALED.search(line.text):
                joined = line.text
                for j in range(i + 1, min(i + 3, len(lines))):
                    if _TOC_ENTRY.match(lines[j].text) or _CHAPTER.match(lines[j].text):
                        out.append(line)  # e.g. "59. Repealed." followed by the next section
                        i += 1
                        break
                    joined = f"{joined} {lines[j].text}"
                    if cls._heading(joined, toc) or _REPEALED.search(joined):
                        out.append(_Line(joined, line.blank_before))
                        i = j + 1
                        break
                else:
                    out.append(line)
                    i += 1
                continue
            out.append(line)
            i += 1
        return out

    @staticmethod
    def toc_sections(lines: "list[_Line] | str") -> list[str]:
        """Section numbers from the arrangement-of-sections table that precedes the body."""
        return LegalDocumentParser._split_toc(lines)[0]

    @staticmethod
    def _split_toc(lines: "list[_Line] | str") -> tuple[list[str], int]:
        """(table-of-contents section numbers, index of the first body line).

        The table lists every section once, in order; the body starts where the numbering
        restarts at the first listed section with a real heading ("1. Title ...—This Act ...").
        Without such a restart there is no table: ([], 0).
        """
        if isinstance(lines, str):
            lines = LegalDocumentParser._lines(lines)
        toc: list[str] = []
        for i, line in enumerate(lines):
            m = _TOC_ENTRY.match(line.text)
            if not m:
                continue
            if toc and m.group(1) == toc[0] and _SECTION.match(line.text):
                return toc, i
            toc.append(m.group(1))
        return [], 0

    def _parse_paragraphs(self, text: str, doc_id: str) -> list[Node]:
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        return [
            Node("paragraph", f"¶{i}", f"{doc_id} > ¶{i}", p)
            for i, p in enumerate((p for p in paragraphs if p), start=1)
        ]

    @staticmethod
    def _child(parent: Node, kind: str, label: str, text: str) -> Node:
        node = Node(
            kind,  # type: ignore[arg-type]
            label,
            f"{parent.citation_path} > {label}",
            text.strip(),
            chapter=parent.chapter,
            section=parent.section,
        )
        parent.children.append(node)
        return node

    @staticmethod
    def _path(doc_id: str, chapter: str, leaf: str) -> str:
        return " > ".join(p for p in (doc_id, f"Chapter {chapter}" if chapter else "", leaf) if p)

    @staticmethod
    def _lines(text: str) -> list[_Line]:
        out: list[_Line] = []
        blank = False
        for raw in text.replace("\r", "").split("\n"):
            line = re.sub(r"\d+\[", "", raw)  # "1[304A." amendment marker in plain-text sources
            line = _AMENDMENT_MARKS.sub(" ", line)
            line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
            line = re.sub(r" +([.,;:])", r"\1", line)  # "negligence ." -> "negligence."
            if not line:
                blank = True
                continue
            out.append(_Line(line, blank))
            blank = False
        return out

    # --------------------------------------------------------------- chunking

    def chunk(self, nodes: list[Node], max_chunk_chars: int = 1400, overlap: int = 150) -> list[Chunk]:
        """Pack each top-level node's sentences into chunks of <= max_chunk_chars.

        Chunks never cross a top-level node (section/paragraph) boundary and never split
        a sentence, unless one sentence alone exceeds the budget (then it is broken at
        clause punctuation). Consecutive chunks of one node share up to `overlap` chars
        of whole trailing sentences.
        """
        chunks: list[Chunk] = []
        for node in nodes:
            chunks.extend(self._chunk_node(node, max_chunk_chars, overlap))
        return chunks

    def _chunk_node(self, node: Node, max_chars: int, overlap: int) -> list[Chunk]:
        if node.kind == "section":
            header = f"Section {node.label}. {node.title}\n"
        else:
            header = f"{node.title}\n" if node.title else ""
        budget = max_chars - len(header)
        if budget <= overlap:
            raise ValueError("max_chunk_chars too small for section header plus overlap")

        # (path, unit_index, sentence) triples in reading order; units join with newlines.
        pieces: list[tuple[str, int, str]] = []
        for unit_index, unit in enumerate(self._units(node)):
            label = f"{unit.label} " if unit.kind in ("subsection", "clause") else ""
            label = f"{unit.label}.— " if unit.kind == "paragraph" and unit is not node else label
            sentences = split_sentences(f"{label}{unit.text}".strip())
            for sentence in sentences:
                for piece in split_oversized(sentence, budget - overlap):
                    pieces.append((unit.citation_path, unit_index, piece))

        chunks: list[Chunk] = []
        start = 0
        carried: list[tuple[str, int, str]] = []
        while start < len(pieces):
            body = list(carried)
            end = start
            while end < len(pieces) and len(self._join(body + [pieces[end]])) <= budget:
                body.append(pieces[end])
                end += 1
            if end == start:  # cannot fit even with overlap: drop overlap and retry
                carried = []
                continue
            new = pieces[start:end]
            chunks.append(self._make_chunk(node, header + self._join(body), [p[0] for p in new]))
            carried = self._tail(new, overlap)
            start = end
        if len(chunks) == 1:  # the whole node fits: cite the node itself, not a span of its parts
            chunks[0] = self._make_chunk(node, chunks[0].text, [node.citation_path])
        return chunks

    @staticmethod
    def _units(node: Node) -> list[Node]:
        units: list[Node] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.text or current is node or current.kind in ("subsection", "clause", "paragraph"):
                units.append(current)
            stack.extend(reversed(current.children))
        return [u for u in units if u.text or u.kind != "section"] or [node]

    @staticmethod
    def _join(pieces: list[tuple[str, int, str]]) -> str:
        out = ""
        prev_unit: int | None = None
        for _, unit, text in pieces:
            if prev_unit is None:
                out = text
            else:
                out += (" " if unit == prev_unit else "\n") + text
            prev_unit = unit
        return out

    @staticmethod
    def _tail(pieces: list[tuple[str, int, str]], overlap: int) -> list[tuple[str, int, str]]:
        tail: list[tuple[str, int, str]] = []
        size = 0
        for piece in reversed(pieces):
            size += len(piece[2]) + 1
            if size > overlap:
                break
            tail.insert(0, piece)
        return tail

    def _make_chunk(self, node: Node, text: str, paths: list[str]) -> Chunk:
        doc_id = node.citation_path.split(" > ", 1)[0]
        citation_path = self._span_path(paths)
        digest = hashlib.sha1(f"{citation_path}\n{text}".encode()).hexdigest()[:12]
        return Chunk(
            chunk_id=f"{doc_id}:{node.section or node.label}:{digest}",
            doc_id=doc_id,
            text=text,
            citation_path=citation_path,
            chapter=node.chapter,
            section=node.section,
            title=node.title,
        )

    @staticmethod
    def _span_path(paths: list[str]) -> str:
        """Deepest common path; if the chunk spans siblings, append a range like "(a)–(d)"."""
        split = [p.split(" > ") for p in paths]
        common: list[str] = []
        for parts in zip(*split):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        depth = len(common)
        first = split[0][depth] if len(split[0]) > depth else None
        last = split[-1][depth] if len(split[-1]) > depth else None
        if first and last and first != last:
            common.append(f"{first}–{last}")
        return " > ".join(common)
