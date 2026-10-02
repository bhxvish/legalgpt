"""SentenceSegmenter: judgment text -> sentences for annotation."""

import re

from app.core.text_utils import LEGAL_ABBREVIATIONS, split_sentences

# Abbreviations in Indian judgments that always lead into something else ("S. 304A",
# "State v. Kumar", "Smt. Rao", "PW. 2", "u/s. 313"), so a following capital never starts a new
# sentence. Extends the bare-act list. Lower-cased, final period removed.
# Deliberately absent: abbreviations that often END a sentence ("... under Section 302 I.P.C.",
# "... Cr.P.C.", "... & Ors.", "etc.") — a capitalised word after them does start a new sentence.
JUDGMENT_ABBREVIATIONS: frozenset[str] = LEGAL_ABBREVIATIONS | frozenset(
    {
        "cr", "crl", "hon", "hon'ble", "ld", "smt", "sri", "shri", "kum", "km", "pw", "dw", "cw",
        "ex", "exh", "exts", "u/s", "r/w", "sec", "w.p", "s.l.p", "sl", "nos", "addl", "asst", "dy",
        "insp", "const", "s.i", "a.s.i", "dist", "vill", "p.s", "approx", "fig", "crl.a", "crl.r.p",
        "edn", "art", "ch",
    }
)

_PARA_NUMBER = re.compile(r"^(?:\d{1,3}\.|\(\d{1,3}\)|\[\d{1,3}\])\s+(?=\S)")
_TERMINAL = re.compile(r"[.?!:;\"”’)\]]$")


class SentenceSegmenter:
    """Splits a judgment into sentences.

    - Never splits after legal abbreviations ("State v. Kumar", "S. 304A", "Cr.P.C.") or
      single-letter initials ("M.K. Sharma").
    - Re-joins lines wrapped by PDF/text export, but keeps short heading lines
      ("IN THE SUPREME COURT OF INDIA") as their own segments.
    - Drops leading paragraph numbers ("12. The appellant ..." -> "The appellant ...").
    """

    def __init__(self, abbreviations: frozenset[str] = JUDGMENT_ABBREVIATIONS, heading_max_chars: int = 70) -> None:
        self.abbreviations = abbreviations
        self.heading_max_chars = heading_max_chars

    def segment(self, text: str, case_id: str = "") -> list[str]:
        sentences: list[str] = []
        for unit in self._units(text):
            unit = _PARA_NUMBER.sub("", unit)
            for sentence in split_sentences(unit, self.abbreviations):
                sentence = _PARA_NUMBER.sub("", sentence).strip()
                if sentence and re.search(r"\w", sentence):
                    sentences.append(sentence)
        return sentences

    def _units(self, text: str) -> list[str]:
        """Blank-line paragraphs, with wrapped lines inside them re-joined."""
        units: list[str] = []
        for para in re.split(r"\n\s*\n", text.replace("\r", "")):
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in para.split("\n")]
            lines = [ln for ln in lines if ln]
            current = ""
            for i, line in enumerate(lines):
                current = self._join(current, line) if current else line
                nxt = lines[i + 1] if i + 1 < len(lines) else None
                if nxt is None or not self._continues(line, nxt):
                    units.append(current)
                    current = ""
        return units

    def _continues(self, line: str, nxt: str) -> bool:
        """Does `nxt` continue the sentence that `line` is part of?"""
        if _PARA_NUMBER.match(nxt):
            return False
        if line.endswith(("-", ",")) or nxt[:1].islower():
            return True
        if _TERMINAL.search(line):
            # A line ending in a period may still be mid-sentence ("... under S." / "304A"):
            # let the sentence splitter decide by keeping such lines together.
            return True
        return len(line) > self.heading_max_chars  # long unterminated line = wrapped text

    @staticmethod
    def _join(current: str, line: str) -> str:
        if current.endswith("-") and line[:1].islower():
            return current[:-1] + line
        return f"{current} {line}"
