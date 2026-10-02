"""JudgmentCollector: raw judgment text files -> normalized, screened JudgmentRecords."""

import hashlib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

from app.annotation.models import JudgmentRecord

ENGLISH_STOPWORDS = frozenset(
    "the of and to in is that was by for a an on with as be it this which or from at he his not are "
    "has have had were been shall under said".split()
)
CRIMINAL_TERMS = (
    "accused", "prosecution", "convict", "acquit", "offence", "fir", "charge-sheet", "chargesheet",
    "cr.p.c", "code of criminal procedure", "penal code", "i.p.c", "ipc", "sessions", "bail",
    "murder", "punishable", "imprisonment", "complainant", "investigating officer",
)
MIN_CRIMINAL_TERMS = 3
MIN_CHARS = 500

_SECTION_LIST = r"\d{1,3}[A-Z]{0,2}(?:\s*\(\d+\))?(?:\s*(?:,|/|&|and|r/w|read\s+with)\s*\d{1,3}[A-Z]{0,2}(?:\s*\(\d+\))?)*"
_IPC = r"(?:I\.\s?P\.\s?C\.?|IPC|Indian\s+Penal\s+Code|Penal\s+Code)"
# "Sections 302, 307 and 34 of the IPC", "u/s 498A IPC", "S. 304A I.P.C."
_CITED_WITH_KEYWORD = re.compile(
    rf"(?:\bsections?|\bsecs?\.?|\bss?\.|\bu/s\.?)\s*({_SECTION_LIST})\s*(?:of\s+(?:the\s+)?)?{_IPC}", re.I
)
# "302/34 IPC", "304B IPC"
_CITED_BARE = re.compile(rf"\b({_SECTION_LIST})\s+{_IPC}", re.I)
# Court names, searched in the header only (the body mentions other courts).
_COURTS = (
    re.compile(r"supreme court of india", re.I),
    re.compile(r"high court of (?:judicature (?:at|for) )?[a-z][a-z&. ]*?(?=\s+(?:at|bench)\b|\s*[\n,]|$)", re.I | re.M),
    re.compile(r"\b(?!the\b|hon'?ble\b|learned\b|said\b)[a-z]+(?: (?!high\b)[a-z]+)? high court\b", re.I),
    re.compile(r"court of (?:the )?(?:additional )?sessions judge[^\n,]*", re.I),
)
_SMALL_WORDS = {"of", "at", "for", "the", "and"}
# "A versus B", "A vs. B", "A v. B" between two capitalised party names
_TITLE = re.compile(r"\S\s+(?:versus|vs\.?|v\.)\s+[A-Z]", re.I)
_DECIDED_ON = re.compile(r"\bon\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+(\d{4})\b")
_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")


def normalize_text(text: str) -> str:
    """Unicode-normalize, unify whitespace, keep line structure (the segmenter uses it)."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def content_hash(text: str) -> str:
    """Hash insensitive to whitespace and case, so re-exported copies of a judgment collide."""
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug[:80] or "judgment"


class JudgmentCollector:
    """Screens judgment files and stores eligible ones under `store_dir`:
    `<case_id>.txt` (normalized text) plus one line per case in `index.jsonl`."""

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.jsonl"

    # ---------------------------------------------------------------- ingest

    def ingest_file(self, path: str | Path) -> JudgmentRecord:
        path = Path(path)
        return self.ingest_text(self._read(path), case_id=slugify(path.stem), source_path=str(path))

    def ingest_text(self, text: str, case_id: str, source_path: str = "") -> JudgmentRecord:
        text = normalize_text(text)
        return JudgmentRecord(
            case_id=case_id,
            court=self.detect_court(text),
            decision_year=self.detect_year(text),
            raw_text=text,
            sections_cited=self.detect_sections(text),
            source_path=source_path,
            sha256=content_hash(text),
            title=self.detect_title(text),
        )

    @staticmethod
    def detect_title(text: str) -> str:
        """The cause title from the header: the first short line naming parties, e.g.
        "RAJO @ RAJWA versus THE STATE OF BIHAR" or "Narender vs State Of Delhi on 12 October, 2021"."""
        for line in text.split("\n")[:15]:
            line = line.strip()
            if 5 < len(line) <= 200 and _TITLE.search(line):
                return line
        return ""

    @staticmethod
    def _read(path: Path) -> str:
        raw = path.read_bytes()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp1252", errors="replace")

    # -------------------------------------------------------------- metadata

    @staticmethod
    def detect_sections(text: str) -> list[str]:
        found: set[str] = set()
        for pattern in (_CITED_WITH_KEYWORD, _CITED_BARE):
            for m in pattern.finditer(text):
                for num in re.findall(r"\d{1,3}[A-Za-z]{0,2}", re.sub(r"\(\d+\)", "", m.group(1))):
                    found.add(num.upper())
        return sorted(found, key=lambda s: (int(re.match(r"\d+", s).group()), s))  # type: ignore[union-attr]

    @staticmethod
    def detect_court(text: str) -> str:
        head = text[:1500]
        for pattern in _COURTS:
            if m := pattern.search(head):
                words = re.sub(r"\s+", " ", m.group(0)).strip().split(" ")
                return " ".join(w.lower() if i and w.lower() in _SMALL_WORDS else w.capitalize() for i, w in enumerate(words))
        return ""

    @staticmethod
    def detect_year(text: str) -> int | None:
        head = text[:3000]
        if m := _DECIDED_ON.search(head):
            return int(m.group(1))
        years = [int(y) for y in _YEAR.findall(head) if int(y) <= date.today().year]
        return max(years) if years else None

    # ------------------------------------------------------------- screening

    def is_eligible(self, record: JudgmentRecord) -> tuple[bool, str]:
        text = record.raw_text
        if len(text) < MIN_CHARS:
            return False, f"too short ({len(text)} chars < {MIN_CHARS})"
        if not self._is_english(text):
            return False, "not English (too few Latin letters or English function words)"
        for entry in self.index():
            if entry["sha256"] == record.sha256:
                return False, f"duplicate of already-collected case {entry['case_id']}"
            if entry["case_id"] == record.case_id:
                return False, f"case_id {record.case_id!r} already used by a different judgment"
        if not record.sections_cited:
            lowered = text.lower()
            hits = [t for t in CRIMINAL_TERMS if t in lowered]
            if len(hits) < MIN_CRIMINAL_TERMS:
                return False, f"not a criminal case (no IPC sections cited; criminal terms found: {hits or 'none'})"
        return True, "eligible"

    @staticmethod
    def _is_english(text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        latin = sum(c.isascii() for c in letters) / len(letters)
        words = re.findall(r"[a-z]+", text.lower())
        function_words = sum(w in ENGLISH_STOPWORDS for w in words) / max(len(words), 1)
        return latin >= 0.9 and function_words >= 0.15

    # --------------------------------------------------------------- storage

    def collect(self, record: JudgmentRecord) -> JudgmentRecord:
        """Screen and, if eligible, persist. Returns the record with its status set."""
        ok, reason = self.is_eligible(record)
        record.selection_status = "eligible" if ok else "rejected"
        record.rejection_reason = "" if ok else reason
        if ok:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            (self.store_dir / f"{record.case_id}.txt").write_text(record.raw_text, encoding="utf-8")
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.index_entry(), ensure_ascii=False) + "\n")
        return record

    def index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return [json.loads(line) for line in self.index_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load(self, case_id: str) -> JudgmentRecord:
        entry = next((e for e in self.index() if e["case_id"] == case_id), None)
        if entry is None:
            raise KeyError(f"no collected judgment {case_id!r}")
        text = (self.store_dir / f"{case_id}.txt").read_text(encoding="utf-8")
        return JudgmentRecord(raw_text=text, **entry)
