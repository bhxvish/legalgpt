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
