"""Automatic scoring of free-text answers against the benchmark's gold labels.

Everything here is a deterministic text check, so different systems are scored identically:
  cited sections      "Section 379", "Sections 302 and 34", "s. 304-B", "379 IPC" ... (IPC context only;
                      numbers followed by BNS / CrPC / "... Act" are separated out)
  invented sections   cited IPC numbers that do not exist in the IPC
  BNS confusion       the answer gives the offence's *BNS* number as if it were the IPC section
                      (benchmark/ipc_bns_map.json, approximate), or mentions the BNS at all
  punishment terms    "two years", "life", "death" stated vs the statute's terms
  refusal             the answer declines ("I don't know", "not covered", "cannot advise" ...)
"""

import json
import re
from pathlib import Path

from app.benchmark.statute import terms_in
from app.config import REPO_ROOT

_NUM = r"\d{1,3}(?:-?[A-Z]{1,2}(?![A-Za-z]))?"  # 304B, 304-B (the suffix is attached, never "103 of")
_LIST = re.compile(rf"\b(?:Sections?|Secs?\.|Ss?\.|§§?|u/s\.?|under section)\s*({_NUM}(?:\s*(?:\(\d+\))?\s*(?:,|and|or|&|/|r/w|read with)\s*(?:Sections?\s*)?{_NUM})*)",
                   re.IGNORECASE)
_BARE = re.compile(rf"\b({_NUM})\s*(?:\(\d+\)\s*)?(?:of the\s+)?(?:IPC|I\.P\.C\.?|Indian Penal Code)\b")
_OTHER = re.compile(r"^\W{0,3}(?:\(\d+\)\s*)?(?:of (?:the )?)?(BNS\b|Bharatiya Nyaya|BNSS|Cr\.?P\.?C|Code of Criminal Procedure|[A-Z][\w ]{0,40}\bAct\b|Constitution)")
_REFUSAL = re.compile(r"I don't know|I do not know|not covered|do not cover|don't cover|cannot (?:answer|help|provide|advise)|"
                      r"can't (?:answer|help|advise)|outside (?:the |my )?scope|not (?:a |an )?(?:IPC|criminal) (?:offence|matter)|"
                      r"no (?:IPC )?offence", re.IGNORECASE)


def _norm(n: str) -> str:
    return re.sub(r"[\s-]", "", n).upper()


def cited_sections(text: str) -> dict[str, list[str]]:
    """{"ipc": [...], "other": [...]} in order of first mention."""
    # Unicode spaces ("Sec. 324"), Unicode hyphens ("120‑B"), markdown bold
    text = re.sub(r"[‐-―]", "-", re.sub(r"\s", " ", text)).replace("*", "")
    ipc, other = [], []
    for pattern in (_LIST, _BARE):
        for m in pattern.finditer(text):
            tail = text[m.end(): m.end() + 60]
            nums = [_norm(x) for x in re.findall(_NUM, m.group(1))]
            target = other if _OTHER.match(tail) and pattern is _LIST else ipc
            for n in nums:
                if n not in target:
                    target.append(n)
    return {"ipc": ipc, "other": other}


def refused(text: str) -> bool:
    return bool(_REFUSAL.search(re.sub(r"\s", " ", text)[:400]))


def mentions_bns(text: str) -> bool:
    return bool(re.search(r"\bBNS\b|Bharatiya Nyaya Sanhita", text))


_BNS = None


def bns_number(ipc_section: str) -> str | None:
    global _BNS
    if _BNS is None:
        p = REPO_ROOT / "benchmark" / "ipc_bns_map.json"
        _BNS = json.loads(p.read_text(encoding="utf-8"))["map"] if p.exists() else {}
    return _BNS.get(ipc_section)


def score_statute(item: dict, answer: str, extra_cited: list[str] | None = None, ipc_sections: set[str] | None = None) -> dict:
    cites = cited_sections(answer)
    ipc = list(dict.fromkeys(cites["ipc"] + (extra_cited or [])))
    out = {"refused": refused(answer), "section_correct": item["section"] in ipc, "cited": ipc,
           "invented": [s for s in ipc if ipc_sections is not None and s not in ipc_sections],
           "mentions_bns": mentions_bns(answer)}
    bns = bns_number(item["section"])
    out["bns_confusion"] = bool(bns and bns != item["section"] and bns in ipc and item["section"] not in ipc)
    if item["kind"] == "punishment":
        got, want = terms_in(answer), set(item["gold_terms"])
        out["punishment_correct"] = want <= got
        out["punishment_wrong"] = bool(got) and not (got & want)
    return out


def score_scenario(item: dict, answer: str, extra_cited: list[str] | None = None, ipc_sections: set[str] | None = None) -> dict:
    ipc = list(dict.fromkeys(cited_sections(answer)["ipc"] + (extra_cited or [])))
    ok = set(item["acceptable"])
    return {"refused": refused(answer) and not ipc, "hit": bool(ok & set(ipc)), "wrong_only": bool(ipc) and not (ok & set(ipc)),
            "cited": ipc, "invented": [s for s in ipc if ipc_sections is not None and s not in ipc_sections],
            "mentions_bns": mentions_bns(answer)}


def score_oos(answer: str, extra_cited: list[str] | None = None) -> dict:
    ipc = list(dict.fromkeys(cited_sections(answer)["ipc"] + (extra_cited or [])))
    return {"refused": refused(answer), "criminalised": bool(ipc), "cited": ipc}


def known_sections(chunks) -> set[str]:
    """Every IPC section a correct answer may cite: the corpus's sections, plus sections that state
    amendments insert ("379A", "509B" in Gujarat or Chhattisgarh). Without the latter, an answer citing
    an amendment it was given as a source counted as an invented section."""
    secs = {c.section for c in chunks if c.section}
    for c in chunks:
        if "State Amendments" in c.citation_path:
            # "section 509B" in the text, or an inserted section's heading ("379A. Snatching.")
            secs |= {m.upper() for m in re.findall(r"\bsections?\s+(\d{1,3}[A-Z]{1,2})\b", c.text, re.IGNORECASE)}
            secs |= {m.upper() for m in re.findall(r"(?:^|\s)(\d{1,3}[A-Z]{1,2})\.\s*[A-Z]", c.text)}
    return secs


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
