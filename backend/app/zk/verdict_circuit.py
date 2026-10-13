"""The fact check as an arithmetic circuit that EZKL can prove.

Facts are encoded as a vector of small integers (FactEncoder):
  boolean predicate   one slot: true = 1, false = -1, unknown = 0
  act_type            act_known (1 if not unknown) + one slot per act type (1 if it is that type)

Every element of every encoded section becomes one row of a fixed integer matrix W, so that
W @ facts gives one status per element: 1 satisfied, -1 violated, 0 missing - exactly the
semantics of rules/core.pl (element_status/4):
  required true / false     +x / -x of the predicate's slot
  act_type in S             2 * sum(slot for s in S) - act_known
The verdict follows from the statuses as in compute_verdict/3: any violated -> INCONSISTENT,
else any missing -> INSUFFICIENT, else CONSISTENT. W is baked into the circuit as constants, so
the circuit's verification key commits to the rules.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.verification.fact_schema import ACT_TYPES, PREDICATES

STATUS = {1: "satisfied", -1: "violated", 0: "missing"}
ACT_VALUES = tuple(v for v in ACT_TYPES if v != "unknown")


class FactEncoder:
    """Slot layout over the shared predicate schema."""

    def __init__(self) -> None:
        self.slots: list[str] = []
        for p in PREDICATES:
            if p.name == "act_type":
                self.slots += ["act_type:known"] + [f"act_type={v}" for v in ACT_VALUES]
            else:
                self.slots.append(p.name)
        self.index = {s: i for i, s in enumerate(self.slots)}

    def encode(self, facts: dict[str, Any]) -> list[int]:
        """facts: {predicate: "value"} or {predicate: {"value": ...}}; missing = unknown."""
        x = [0] * len(self.slots)
        for p in PREDICATES:
            raw = facts.get(p.name, "unknown")
            value = str(raw.get("value", "unknown") if isinstance(raw, dict) else raw).strip().lower()
            if value not in p.values:
                raise ValueError(f"{p.name}: {value!r} is not one of {list(p.values)}")
            if p.name == "act_type":
                if value != "unknown":
                    x[self.index["act_type:known"]] = 1
                    x[self.index[f"act_type={value}"]] = 1
            else:
                x[self.index[p.name]] = {"true": 1, "false": -1, "unknown": 0}[value]
        return x


@dataclass(frozen=True)
class Row:
    section: str
    title: str
    predicate: str
    required: tuple[str, ...]


class VerdictLayout:
    """The element matrix W and the meaning of each output row, built from the rule definitions
    (PrologEngine.sections()). Serialisable, so a verifier needs neither Prolog nor this code's
    rules directory - only the published layout, whose hash is in the manifest."""

    def __init__(self, rows: list[Row], slots: list[str], W: list[list[int]]) -> None:
        self.rows, self.slots, self.W = rows, slots, W

    @classmethod
    def from_sections(cls, sections) -> "VerdictLayout":
        enc = FactEncoder()
        rows, W = [], []
        for s in sorted(sections, key=lambda s: s.section):
            for e in s.elements:
                w = [0] * len(enc.slots)
                if e.predicate == "act_type":
                    w[enc.index["act_type:known"]] = -1
                    for v in e.required:
                        w[enc.index[f"act_type={v}"]] = 2
                else:
                    req = set(e.required)
                    if req == {"true"}:
                        w[enc.index[e.predicate]] = 1
                    elif req == {"false"}:
                        w[enc.index[e.predicate]] = -1
                    else:
                        raise ValueError(f"s.{s.section} {e.predicate}: required {e.required} has no linear encoding")
                rows.append(Row(s.section, s.title, e.predicate, tuple(e.required)))
                W.append(w)
        return cls(rows, enc.slots, W)

    def to_dict(self) -> dict[str, Any]:
        return {"slots": self.slots, "W": self.W,
                "rows": [{"section": r.section, "title": r.title, "predicate": r.predicate, "required": list(r.required)}
                         for r in self.rows]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerdictLayout":
        return cls([Row(r["section"], r["title"], r["predicate"], tuple(r["required"])) for r in d["rows"]], d["slots"], d["W"])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def evaluate(self, x: list[int]) -> list[int]:
        """Reference (non-ZK) evaluation of the circuit: W @ x, then the validity penalty."""
        return [sum(w * v for w, v in zip(row, x)) for row in self.W] + [validity_penalty(self.slots, x)]

    def sections(self) -> list[str]:
        return list(dict.fromkeys(r.section for r in self.rows))

    def decode(self, outputs: list[int]) -> dict[str, dict[str, Any]]:
        """Per section: verdict and element statuses, from the circuit's output vector
        (statuses, then the validity penalty, which must be 0)."""
        if len(outputs) != len(self.rows) + 1:
            raise ValueError(f"expected {len(self.rows) + 1} outputs, got {len(outputs)}")
        statuses, penalty = outputs[:-1], outputs[-1]
        if penalty != 0:
            raise ValueError(f"the proven facts are not a valid encoding (penalty {penalty})")
        if len(statuses) != len(self.rows):
            raise ValueError(f"expected {len(self.rows)} statuses, got {len(statuses)}")
        out: dict[str, dict[str, Any]] = {}
        for r, v in zip(self.rows, statuses):
            if v not in STATUS:
                raise ValueError(f"s.{r.section} {r.predicate}: status {v} is not -1, 0 or 1")
            sec = out.setdefault(r.section, {"section": r.section, "title": r.title, "elements": []})
            sec["elements"].append({"predicate": r.predicate, "required": list(r.required), "status": STATUS[v]})
        for sec in out.values():
            st = [e["status"] for e in sec["elements"]]
            sec["status"] = "INCONSISTENT" if "violated" in st else "INSUFFICIENT" if "missing" in st else "CONSISTENT"
        return out


def validity_penalty(slots: list[str], x: list[int]) -> int:
    """0 exactly when x is a well-formed encoding (see torch_module); reference version."""
    known = x[slots.index("act_type:known")]
    acts = [v for s, v in zip(slots, x) if s.startswith("act_type=")]
    bools = [v for s, v in zip(slots, x) if not s.startswith("act_type")]
    return (sum((v * v * v - v) ** 2 for v in bools) + sum((v * v - v) ** 2 for v in acts + [known])
            + (known - sum(acts)) ** 2)


def torch_module(layout: VerdictLayout):
    """x (1 x slots) -> [statuses (rows) ..., penalty] (1 x rows+1).

    The facts are private, so the circuit itself must reject inputs that are not a real encoding
    (e.g. two act types at once, or a "true" of 5): the last output is a penalty that is 0 only
    when every boolean slot is in {-1, 0, 1}, every act slot and act_known are in {0, 1}, and
    act_known equals the number of act types set. Verifiers require it to be 0.
    Imported lazily: torch is only needed to build the circuit.
    """
    import torch

    acts = [i for i, s in enumerate(layout.slots) if s.startswith("act_type=")]
    known = layout.slots.index("act_type:known")
    bools = [i for i, s in enumerate(layout.slots) if not s.startswith("act_type")]

    class VerdictModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("W", torch.tensor(layout.W, dtype=torch.float32))
            self.register_buffer("bools", torch.tensor(bools))
            self.register_buffer("binary", torch.tensor(acts + [known]))
            self.register_buffer("acts", torch.tensor(acts))
            self.known = known

        def forward(self, x):
            b = x[:, self.bools]
            a = x[:, self.binary]
            k = x[:, self.known]
            pen = ((b * b * b - b) ** 2).sum(dim=1) + ((a * a - a) ** 2).sum(dim=1) \
                + (k - x[:, self.acts].sum(dim=1)) ** 2
            return torch.cat([x @ self.W.T, pen.view(-1, 1)], dim=1)

    return VerdictModule()
