"""LLMClient interface and its implementations.

Phase 1 ships GroqClient only. Phase 4 adds AdapterClient (LoRA-tuned model) behind
the same interface, so callers must depend on LLMClient, never on GroqClient directly.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from app.core.models import Message


class LLMClientError(RuntimeError):
    """Raised when the backing model cannot be reached or is misconfigured."""


class LLMClient(ABC):
    model_id: str

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Return the full completion for `messages`."""

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the completion incrementally as text fragments."""


class GroqClient(LLMClient):
    """Chat model served by Groq (default openai/gpt-oss-120b; reasoning tokens arrive separately and are not streamed)."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        max_tokens: int = 2048,  # reasoning models spend part of this budget on hidden reasoning
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise LLMClientError("GROQ_API_KEY is not set; add it to .env (see .env.example).")
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def _payload(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=self._payload(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=self._payload(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in stream:
            if event.choices and (delta := event.choices[0].delta.content):
                yield delta
