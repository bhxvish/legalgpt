"""Sentence splitting that respects common legal abbreviations."""

import re

# Abbreviations whose trailing period must not end a sentence ("S. 304A", "State v. Kumar", "Act No. 45").
LEGAL_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "s", "ss", "sec", "secs", "no", "nos", "sch", "ins", "subs", "cl", "cls", "art", "arts",
        "v", "vs", "viz", "i.e", "e.g", "w.e.f", "rs", "mr", "mrs", "dr", "hon'ble", "ltd",
        "co", "govt", "ord", "reg", "ch", "para", "paras", "p", "pp", "vol", "sub-s", "illus",
    }
)

_BOUNDARY = re.compile(r"(?<=[.?!])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, but never after a legal abbreviation
    or a single capital initial (e.g. "A." in IPC illustrations)."""
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        candidate = text[start : m.start()].rstrip()
        last_word = candidate.rsplit(None, 1)[-1] if candidate else ""
        token = last_word.rstrip(".").lstrip("(\"'[").lower()
        if token in LEGAL_ABBREVIATIONS or re.fullmatch(r"[A-Z]", last_word.rstrip(".")):
            continue
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_oversized(sentence: str, max_chars: int) -> list[str]:
    """Last resort for a single sentence longer than a whole chunk: break at
    clause punctuation (; : ,) and finally at whitespace. Each piece <= max_chars."""
    if len(sentence) <= max_chars:
        return [sentence]
    for sep in ("; ", ": ", ", ", " "):
        parts = sentence.split(sep)
        if len(parts) == 1:
            continue
        pieces: list[str] = []
        current = ""
        for i, part in enumerate(parts):
            token = part + (sep.rstrip() if i < len(parts) - 1 else "")
            joined = f"{current} {token}".strip() if current else token
            if len(joined) <= max_chars:
                current = joined
            else:
                if current:
                    pieces.append(current)
                current = token
        if current:
            pieces.append(current)
        if all(len(p) <= max_chars for p in pieces):
            return pieces
    return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
