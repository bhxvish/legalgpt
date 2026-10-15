"""FallbackClient: try several models in order, moving on when one is out of quota or unavailable.

Used by the fact check (Verify tab): Groq first, then the free hosted models in EXTRA_MODELS
(Gemini), then the local LoRA model. A model that hit a usage limit is skipped for a while
(an hour for a daily limit, a minute for a per-minute one), so later requests do not wait on it.
`model_id` names the model that answered the current request (per thread), so results report
which model actually extracted the facts.
"""

import logging
import threading
import time
from collections.abc import Iterator

from app.core.llm_client import LLMClient, LLMClientError, LLMRateLimitError
from app.core.models import Message

logger = logging.getLogger(__name__)

DAILY_COOLDOWN_S = 3600.0
MINUTE_COOLDOWN_S = 60.0


class FallbackClient(LLMClient):
    def __init__(self, clients: list[LLMClient], clock=time.monotonic) -> None:
        if not clients:
            raise ValueError("FallbackClient needs at least one model")
        self.clients = list(clients)
        self._clock = clock
        self._skip_until: dict[int, float] = {}  # id(client) -> time until which it is skipped
        self._local = threading.local()

    def add(self, client: LLMClient) -> None:
        if all(c is not client for c in self.clients):
            self.clients.append(client)

    @property
    def model_id(self) -> str:
        used = getattr(self._local, "used", None)
        return used.model_id if used else self.clients[0].model_id

    @property
    def note(self) -> str:
        """Why a later model answered (e.g. Groq's daily limit), for the current request; '' if the first did."""
        return getattr(self._local, "note", "")

    def _cooldown(self, client: LLMClient, err: LLMClientError) -> None:
        daily = "daily" in str(err).lower()
        self._skip_until[id(client)] = self._clock() + (DAILY_COOLDOWN_S if daily else MINUTE_COOLDOWN_S)

    def generate(self, messages: list[Message]) -> str:
        self._local.used, self._local.note = None, ""
        errors: list[str] = []
        now = self._clock()
        candidates = [c for c in self.clients if self._skip_until.get(id(c), 0) <= now] or self.clients
        for client in candidates:
            try:
                out = client.generate(messages)
            except LLMRateLimitError as e:
                self._cooldown(client, e)
                errors.append(f"{client.model_id}: {e}")
                logger.warning("fallback: %s is out of quota, trying the next model", client.model_id)
            except LLMClientError as e:
                errors.append(f"{client.model_id}: {e}")
                logger.warning("fallback: %s failed (%s), trying the next model", client.model_id, e)
            else:
                self._local.used = client
                skipped = [c.model_id for c in self.clients[: self.clients.index(client)]]
                if skipped:
                    self._local.note = f"{', '.join(skipped)} unavailable (usage limit or error); used {client.model_id}"
                return out
        raise LLMClientError("No model could answer. " + " | ".join(errors))

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield self.generate(messages)
