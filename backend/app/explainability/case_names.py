"""Case names in answers: does a sentence citing a judgment [n] name that judgment, or another one?

The citation-faithfulness evaluation (backend/scripts/eval_citation_faithfulness.py) found sentences
that cite one judgment while naming a different case ("In X v. Y ... [4]" where [4] is Z v. W). The
same check runs on every chat answer and shows up in "Why this answer?".
"""

import re
from dataclasses import asdict, dataclass

# a full stop after these is not a sentence end ("Mohd. Rafiq v. State of M.P.")
_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\[\"'])|\n+")
_ABBREV = re.compile(r"(?:\b(?:v|vs|No|s|ss|Mohd|Md|Smt|Shri|Sri|Dr|Mr|Mrs|Ms|Ltd|Co|Ors|Anr|Pvt|Govt|Art|Cr|Crl|cl|para|etc)|\b[A-Z])\.$")
_COMMON = {"state", "of", "the", "union", "india", "and", "anr", "ors", "others", "another", "through", "thr", "mr", "smt",
           "shri", "sri", "etc", "v", "vs", "versus", "uttar", "pradesh", "madhya", "andhra", "himachal", "arunachal", "tamil",
           "nadu", "west", "bengal", "nct", "delhi", "govt", "government", "station", "house", "officer", "rep", "by", "public",
           "prosecutor", "central", "bureau", "investigation"}


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in _END.split(text):
        if out and _ABBREV.search(out[-1]):
            out[-1] += " " + piece
        else:
            out.append(piece)
    return [s.strip(" -*") for s in out if s.strip(" -*")]


def party_words(title: str) -> set[str]:
    """Distinctive words of a case title ("Mohd. Rafiq v. State of M.P." -> {"mohd", "rafiq"})."""
    return {w for w in re.findall(r"[a-z]{3,}", (title or "").lower()) if w not in _COMMON}


@dataclass
class CaseNameFlag:
    sentence: str
    cited_marker: int
    cited_case: str
    named_case: str  # the other judgment source the sentence names

    def to_dict(self) -> dict:
        return asdict(self)


def check_case_names(answer: str, sources) -> list[CaseNameFlag]:
    """Sentences that cite a judgment [n] but name another judgment among the sources, and not [n]."""
    judgments = {s.marker: s for s in sources if getattr(s, "kind", "statute") == "judgment"}
    if len(judgments) < 2:
        return []
    words = {m: party_words(s.title) for m, s in judgments.items()}
    flags = []
    for sent in sentences(answer):
        low = sent.lower()
        mentioned = {m for m, ws in words.items() if any(re.search(rf"\b{w}\b", low) for w in ws)}
        for n in dict.fromkeys(int(x) for x in re.findall(r"\[(\d+)\]", sent)):
            if n not in judgments or n in mentioned:
                continue
            # another judgment named by a word it does not share with the cited one
            other = next((m for m in mentioned if m != n
                          and any(re.search(rf"\b{w}\b", low) for w in words[m] - words[n])), None)
            if other is not None:
                flags.append(CaseNameFlag(sent[:400], n, judgments[n].title, judgments[other].title))
    return flags
