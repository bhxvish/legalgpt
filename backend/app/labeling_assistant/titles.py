"""Cause titles for the Review tab, read from the judgment header.

The manual collector does not detect titles for raw judgments; the Review list uses this so a
reviewer sees "RAJO @ RAJWA versus THE STATE OF BIHAR" rather than a case id.
"""

import re

# "A versus B", "A vs. B", "A v. B" between two capitalised party names
_TITLE = re.compile(r"\S\s+(?:versus|vs\.?|v\.)\s+[A-Z]", re.I)


def detect_title(text: str) -> str:
    """The cause title from the header: the first short line naming parties, e.g.
    "RAJO @ RAJWA versus THE STATE OF BIHAR" or "Narender vs State Of Delhi on 12 October, 2021"."""
    for line in text.split("\n")[:15]:
        line = line.strip()
        if 5 < len(line) <= 200 and _TITLE.search(line):
            return line
    return ""
