"""ReviewSelector: decide which model predictions a human must check."""

import math
import random
from dataclasses import dataclass, field

from app.labeling_assistant.models import Prediction


@dataclass
class ReviewBatch:
    """One case's predictions, partitioned:
    - mandatory: confidence below tau_conf — always reviewed;
    - audit: a random sample of the confident ones — reviewed to estimate how often the model is
      confidently wrong on this case;
    - auto: the remaining confident predictions — accepted unless the audit fails."""

    case_id: str
    mandatory: list[Prediction] = field(default_factory=list)
    audit: list[Prediction] = field(default_factory=list)
    auto: list[Prediction] = field(default_factory=list)
    tau_conf: float = 0.0
    audit_rate: float = 0.0

    @property
    def all(self) -> list[Prediction]:
        return self.mandatory + self.audit + self.auto


class ReviewSelector:
    def __init__(self, tau_conf: float = 0.7, audit_rate: float = 0.1, seed: int = 13, min_audit: int = 1) -> None:
        if not 0 < tau_conf <= 1 or not 0 <= audit_rate <= 1:
            raise ValueError("tau_conf must be in (0, 1] and audit_rate in [0, 1]")
        self.tau_conf = tau_conf
        self.audit_rate = audit_rate
        self.seed = seed
        self.min_audit = min_audit

    def select(self, predictions: list[Prediction], case_id: str = "") -> ReviewBatch:
        """Never trusts confident predictions blindly: at least `min_audit` of them (when any exist)
        and `audit_rate` of them overall are sampled for review."""
        low = [p for p in predictions if p.confidence < self.tau_conf]
        high = [p for p in predictions if p.confidence >= self.tau_conf]
        n_audit = min(len(high), max(self.min_audit if high else 0, math.ceil(self.audit_rate * len(high))))
        rng = random.Random(f"{self.seed}:{case_id}")  # reproducible per case
        audited = set(id(p) for p in rng.sample(high, n_audit))
        return ReviewBatch(
            case_id=case_id,
            mandatory=low,
            audit=[p for p in high if id(p) in audited],
            auto=[p for p in high if id(p) not in audited],
            tau_conf=self.tau_conf,
            audit_rate=self.audit_rate,
        )
