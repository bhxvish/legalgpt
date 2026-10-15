"""Statute questions built from the IPC text, with answers that are correct by construction.

  punishment   "What is the punishment for <offence>?" for every section of the form "Whoever
               <conduct>, shall be punished with <punishment>": gold = the section and the terms in
               its punishment clause ("two years", "life", "death", fine)
  definition   "What is <term> under the IPC?" for sections that define an offence or concept:
               gold = the section
"""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.corpus import load_chunks
from app.finetuning.ipc_qa import EVAL_SECTIONS, main_chunk, offence_parts

_WORDS = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
          "9": "nine", "10": "ten", "12": "twelve", "14": "fourteen", "20": "twenty"}
TERM = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fourteen|twenty)[\s-]+(years?|months?)\b",
                  re.IGNORECASE)
LIFE = re.compile(r"imprisonment for life|life imprisonment|imprisonment for the remainder", re.IGNORECASE)
# "death" only as a punishment ("punished with death", "death penalty"), not in "causing death"
DEATH = re.compile(r"\b(?:with|by|sentence of|penalty of)\s+death\b|\bdeath (?:penalty|sentence)\b", re.IGNORECASE)
# definition sections the automatic search misses (their wording is not "is said to ...")
EXTRA_DEFINITIONS = ("34", "120A", "141", "299", "300", "320", "321", "322", "339", "340", "383", "390", "403", "405",
                     "425", "463", "499", "503", "107", "96", "97")


@dataclass
class StatuteQuestion:
    id: str
    kind: str  # punishment | definition
    question: str
    section: str
    title: str
    gold_terms: list[str] = field(default_factory=list)  # punishment terms in the statute (lower case)
    lora_held_out: bool = False  # the LoRA adapter never saw this section in training
    invalid: str = ""  # why the question is not scored (kept in the file so the seed-7 sample is unchanged)
    # every punishment term anywhere in the section. Graded sections (s.213: seven years if the screened
    # offence is capital, three if it carries life) make another tier a fair answer, not a wrong one
    allowed_terms: list[str] = field(default_factory=list)


def terms_in(text: str) -> set[str]:
    """Punishment terms in canonical form: "two year", "six month", "life", "death"."""
    text = re.sub(r"\s", " ", text)  # every Unicode space (models write "three years")
    out = {f"{_WORDS.get(m.group(1).lower(), m.group(1).lower())} {m.group(2).lower().rstrip('s')}"
           for m in TERM.finditer(text)}
    if LIFE.search(text):
        out.add("life")
    if DEATH.search(text):
        out.add("death")
    return out


GENERIC_TITLES = {"punishment", "punishments"}


def _topic(title: str) -> str:
    t = re.sub(r"^(punishment for|punishment of)\s+", "", title.strip().rstrip("."), flags=re.I)
    return t[:1].lower() + t[1:] if not t[1:2].isupper() else t


def build(corpus_path: Path, seed: int = 13) -> list[StatuteQuestion]:
    rng = random.Random(seed)
    chunks = load_chunks(corpus_path)
    sections = sorted({c.section for c in chunks if c.section}, key=lambda s: (int(re.match(r"\d+", s).group()), s))
    held = set(EVAL_SECTIONS)
    out: list[StatuteQuestion] = []
    templates = ["What is the punishment for {t} under the IPC?", "How long can someone be imprisoned for {t}?",
                 "What penalty does the Indian Penal Code prescribe for {t}?"]
    for s in sections:
        c = main_chunk(chunks, s)
        if c is None or not c.title:
            continue
        parts = offence_parts(c)
        if parts:
            gold = sorted(terms_in("punished with " + parts[1]))  # the clause follows "punished with"
            if gold:
                q = rng.choice(templates).format(t=_topic(c.title))
                # a bare generic title (s.311 is titled "Punishment") gives "... prescribe for punishment?"
                bad = "topic is a generic word, not an offence" if _topic(c.title).lower() in GENERIC_TITLES else ""
                allowed = sorted(set(gold) | terms_in(c.text))
                out.append(StatuteQuestion(f"pun-{s}", "punishment", q, s, c.title.strip(), gold, s in held, bad, allowed))
        body = c.text.lower()
        if not parts and ("said to" in body or s in EXTRA_DEFINITIONS):
            term = re.sub(r"[“”\"]", "", c.title.strip().rstrip(".")).split(". ")[0]
            q = rng.choice(["What is {t} under the IPC?", "How does the Indian Penal Code define {t}?",
                            "Which IPC section defines {t}, and what does it say?"]).format(t=term[:1].lower() + term[1:])
            out.append(StatuteQuestion(f"def-{s}", "definition", q, s, c.title.strip(), [], s in held))
    return out
