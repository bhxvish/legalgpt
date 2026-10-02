"""Canonical citation markers.

Models differ in how they write citations; everything downstream (the chat stream, the
CitationValidator, the ModelComparator) works on one canonical form, [n].
"""

import re
from collections.abc import Iterable, Iterator

# Models differ in how they write citations: gpt-oss emits 【1】 and [1†L2-L4]. Markers are
# canonicalized to [n] so the UI (and Phase 5's CitationValidator) see one format.
_CANONICAL = str.maketrans({chr(0x3010): "[", chr(0x3011): "]", chr(0xFF3B): "[", chr(0xFF3D): "]", chr(0x202F): " ", chr(0x00A0): " "})  # CJK and full-width brackets, narrow and no-break spaces
_MARKER = re.compile(r"\[(\d+)(?:[" + chr(0x2020) + chr(0x2021) + r"][^\]]*)?\]")  # [1], [1<dagger>L2-L4], [1<double dagger>...]
_MAX_PENDING = 32  # longest bracket we hold back waiting for its "]"


def _canonical_markers(text: str) -> str:
    return _MARKER.sub(lambda m: f"[{m.group(1)}]", text)


def normalize_stream(fragments: Iterable[str]) -> Iterator[str]:
    """Canonicalize citation markers in a token stream. A marker can be split across fragments
    ("[", "1†L2", "-L4]"), so text from an unclosed "[" is held back until it closes."""
    pending = ""
    for fragment in fragments:
        text = pending + fragment.translate(_CANONICAL)
        open_at = text.rfind("[")
        if open_at != -1 and "]" not in text[open_at:] and len(text) - open_at <= _MAX_PENDING:
            text, pending = text[:open_at], text[open_at:]
        else:
            pending = ""
        if text:
            yield _canonical_markers(text)
    if pending:
        yield _canonical_markers(pending)
