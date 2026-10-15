"""QueryRewriter: an everyday account of events -> the IPC offences it may involve, for retrieval.

People describe what happened ("a caller pretending to be from my bank got my OTP"); the statute
uses its own words ("cheating by personation", s.419). The embedding model and the lay-term lexicon
(app/core/lay_terms.py) bridge only part of that gap, so an LLM names the likely offences and IPC
sections. Its output only steers *retrieval*: the statute text of those sections is fetched, and
the answer must still be written from (and cite) that text. The rewrite is shown in the Why panel
and disclosed in the retrieval proof as an unproven input.

The LLM also says whether the matter is criminal at all. A rewrite is used only when it says so
*and* the scope check agrees (Retriever.rewrite_gate), so civil questions ("my landlord won't
return my deposit") are still refused.
"""

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field

from app.core.models import Message

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You map a person's description of events to the Indian Penal Code, 1860 (IPC).
Reply with JSON only, no other text:
{"criminal": true or false, "offences": [{"name": "<offence>", "section": "<IPC section number>"}]}

- "criminal" is false when the matter is civil, consumer, contractual, employment, family-law, tax or
  administrative and involves no IPC offence. Then "offences" is [].
- Otherwise list at most 3 offences, the most likely first. Use the IPC section that defines or punishes
  each offence, e.g. 420 for cheating, 302 for murder, 307 for attempt to murder, 304A for causing death
  by negligence. Use IPC numbers, never BNS numbers.
- Judge only from the facts given. Do not give advice."""

_SECTION = re.compile(r"^\s*(?:s\.|section)?\s*(\d{1,3}[A-Za-z]{0,2})\b")
MAX_OFFENCES = 3


@dataclass
class Rewrite:
    criminal: bool
    offences: list[tuple[str, str]] = field(default_factory=list)  # (name, section)
    model_id: str = ""

    @property
    def text(self) -> str:
        """The wording appended to the search query. "section N" makes the retriever fetch that
        section's text (QueryExpander.section_refs)."""
        return "; ".join(f"{name} (section {sec})" for name, sec in self.offences)

    @classmethod
    def from_text(cls, text: str) -> "Rewrite":
        """Inverse of `text` (a disclosed rewrite, e.g. in a retrieval proof bundle)."""
        found = re.findall(r"\s*(.+?) \(section (\w+)\)(?:;|$)", text)
        return cls(True, [(n.strip(), s) for n, s in found])

    def to_dict(self) -> dict:
        return {"criminal": self.criminal, "offences": [{"name": n, "section": s} for n, s in self.offences],
                "model": self.model_id}


def parse(reply: str, model_id: str = "") -> Rewrite | None:
    """The JSON object in the model's reply, or None if there is none."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)  # reasoning models' visible thoughts
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("criminal"), bool):
        return None
    offences: list[tuple[str, str]] = []
    for o in data.get("offences") or []:
        if not isinstance(o, dict):
            continue
        m = _SECTION.match(str(o.get("section", "")))
        name = re.sub(r"[\s;()]+", " ", str(o.get("name", ""))).strip()[:80]  # ";" and "()" delimit `text`
        if m and name and m.group(1).upper() not in {s for _, s in offences}:
            offences.append((name, m.group(1).upper()))
    return Rewrite(data["criminal"], offences[:MAX_OFFENCES] if data["criminal"] else [], model_id)


class QueryRewriter:
    def __init__(self, cache_size: int = 512) -> None:
        self._cache: OrderedDict[tuple[str, str], Rewrite | None] = OrderedDict()
        self.cache_size = cache_size

    def rewrite(self, question: str, llm, strict: bool = False) -> Rewrite | None:
        """None when there is no usable rewrite (a failed call, a rate limit, an unreadable reply);
        retrieval then runs without one. A refusal needs no model, so a used-up quota must not
        turn it into an error; failures are not cached, so the next question tries again.
        `strict` (benchmarks): a failed call raises instead, so an answer is never silently made
        without the rewrite it was meant to have."""
        key = (llm.model_id, question.strip())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            ask = getattr(llm, "generate_brief", llm.generate)  # little hidden reasoning where supported
            reply = ask([Message("system", REWRITE_PROMPT), Message("user", question)])
        except Exception as exc:  # retrieval works without it
            if strict:
                raise
            logger.warning("query rewrite failed (%s): %s", llm.model_id, exc)
            return None
        out = parse(reply, llm.model_id)
        self._cache[key] = out
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out
