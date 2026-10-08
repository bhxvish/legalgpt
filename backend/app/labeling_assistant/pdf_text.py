"""Judgment text from PDFs: pdfminer extraction plus the clean-up court PDFs need.

Shared by scripts/fetch_sc_judgments.py and the Review tab's upload (JudgmentIntake).
"""

import io
import re
from collections import Counter

# The judgment proper starts after the reporter's editorial headnote (SCR PDFs); absent -> keep all.
_BODY_START = re.compile(r"^\s*(?:JUDGMENT|JUDGEMENT|ORDER)\s*$|The Judgment of the Court was delivered by", re.I)


def clean_judgment_text(text: str) -> str:
    """Re-join ligature splits ("ﬁ led"), cut an SCR editorial headnote, drop page numbers and
    running headers (lines repeated on many pages)."""
    text = re.sub(r"([ﬀ-ﬆ])\s+(?=[a-z])", r"\1", text)  # "ﬁ led" -> "ﬁled" (NFKC later -> "filed")
    lines = text.split("\n")
    start = next((i + 1 for i, line in enumerate(lines) if _BODY_START.search(line)), 0)
    body = lines[start:]
    counts = Counter(line.strip() for line in body if line.strip())

    def running_header(line: str) -> bool:
        s = line.strip()
        if counts[s] >= 3 and len(s) <= 90:
            return True
        # case-title / coram headers repeat on alternate pages, so twice is enough for them
        return counts[s] >= 2 and bool(re.fullmatch(r"\[[^\]]*\]|[^a-z]{3,90}\bv\.[^a-z]{3,90}", s))

    kept = [line for line in body if not re.fullmatch(r"\s*\d{1,4}\s*", line) and not running_header(line)]
    return "\n".join(kept).strip()


def looks_garbled(text: str) -> bool:
    """Text layers that lost their spaces produce very long 'words' ("ofdeceased", "convictedby")."""
    words = re.findall(r"[A-Za-z]+", text)
    return not words or sum(len(w) > 20 for w in words) / len(words) > 0.01


def pdf_to_judgment_text(data: bytes) -> str:
    from pdfminer.high_level import extract_text

    return clean_judgment_text(extract_text(io.BytesIO(data)))
