"""IPC -> BNS (Bharatiya Nyaya Sanhita, 2023, in force from 1 July 2024) section correspondence.

From the official BPRD correspondence table (backend/scripts/build_bns_map.py -> bns_map.json).
Shown next to IPC sections in the UI; the answers themselves are not changed (no model writes BNS
numbers, so none can be invented). An IPC section with no entry has no direct BNS equivalent:
it was omitted or folded into another provision (e.g. s.124A sedition, s.377, s.497).
"""

import json
from functools import lru_cache
from pathlib import Path

MAP_PATH = Path(__file__).with_name("bns_map.json")


@lru_cache(maxsize=1)
def _map() -> dict:
    try:
        return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"ipc_to_bns": {}, "source": ""}


def source() -> str:
    return _map().get("source", "")


def bns_for(section: str) -> list[str]:
    """BNS section(s) for an IPC section ("302" -> ["103"]); [] when there is no direct equivalent."""
    return [m["bns"] for m in _map()["ipc_to_bns"].get(str(section).upper(), [])]


def annotate(sources: list) -> list:
    """Set `bns` on statute sources (in place) and return them."""
    for s in sources:
        if getattr(s, "kind", "statute") == "statute" and getattr(s, "section", ""):
            s.bns = bns_for(s.section) or ["none"]
    return sources


def base(section: str) -> str:
    """'3(5)' -> '3', '65(1)' -> '65': the indexed section a sub-section belongs to."""
    return str(section).split("(")[0].strip()


@lru_cache(maxsize=1)
def _reverse() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ipc, rows in _map()["ipc_to_bns"].items():
        for r in rows:
            out.setdefault(base(r["bns"]), [])
            if ipc not in out[base(r["bns"])]:
                out[base(r["bns"])].append(ipc)
    return out


def ipc_for(bns_section: str) -> list[str]:
    """IPC section(s) a BNS section replaced ("103" -> ["302"]); [] for a provision new in the BNS."""
    return _reverse().get(base(bns_section), [])


def annotate_bns(sources: list) -> list:
    """Set `ipc` on BNS statute sources (in place): the IPC section(s) each one replaced."""
    for s in sources:
        if getattr(s, "kind", "statute") == "statute" and getattr(s, "section", ""):
            s.ipc = ipc_for(s.section) or ["new"]
    return sources


_BNS_NAMED = r"\b(?:BNS|Bharatiya\s+Nyaya|Nyaya\s+Sanhita)\b"
_IPC_NAMED = r"\b(?:IPC|Indian\s+Penal\s+Code)\b"
_MONTHS = r"(?:jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def choose_law(question: str, requested: str = "auto") -> tuple[str, str]:
    """("ipc" | "bns", why). Auto: the BNS when the question names it or dates the events on or after
    1 July 2024 (when the BNS came into force); otherwise the IPC, which still governs offences
    committed before that date and the existing case law."""
    import re

    if requested in ("ipc", "bns"):
        return requested, "chosen"
    q = question
    if re.search(_BNS_NAMED, q, re.I):
        return "bns", "the question names the BNS"
    if re.search(_IPC_NAMED, q, re.I):
        return "ipc", "the question names the IPC"
    if re.search(r"\b20(?:2[5-9]|[3-9]\d)\b", q) or re.search(rf"\b{_MONTHS}\w*\.?,?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?2024\b", q, re.I) \
            or re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\w*,?\s+2024\b", q, re.I) \
            or re.search(r"\b(?:0?[7-9]|1[0-2])[/.-]2024\b", q):
        return "bns", "the events are dated on or after 1 July 2024"
    return "ipc", "default: the IPC (offences before 1 July 2024, and most case law)"
