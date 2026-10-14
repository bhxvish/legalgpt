"""OpenAICompatClient: any chat API that speaks the OpenAI /chat/completions format.

One client covers several providers (free tiers marked):
  gemini      Google Gemini via its OpenAI-compatible endpoint      GEMINI_API_KEY      (free tier in AI Studio)
  github      GitHub Models (OpenAI, Meta, Mistral ... models)      GITHUB_TOKEN        (free, rate-limited)
  openai      OpenAI                                                OPENAI_API_KEY      (paid)
  openrouter  OpenRouter (many providers; some ":free" models)      OPENROUTER_API_KEY  (pay as you go)

A model is named "<provider>:<model>", e.g. "gemini:gemini-2.5-flash" or "github:openai/gpt-4o-mini".
Keys are read from the environment or .env and never logged. httpx only, no provider SDKs.
"""

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.llm_client import LLMClient, LLMClientError, LLMRateLimitError
from app.core.models import Message

PROVIDERS: dict[str, tuple[str, str]] = {  # name -> (base URL, key variable)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def provider_key(provider: str, env_file: Path | None = None) -> str:
    """The provider's key from the environment, else from .env (the backend's settings file)."""
    var = PROVIDERS[provider][1]
    if os.environ.get(var):
        return os.environ[var].strip()
    if env_file is None:
        from app.config import REPO_ROOT

        env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{var}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def parse_spec(spec: str) -> tuple[str, str]:
    """"gemini:gemini-2.5-flash" -> ("gemini", "gemini-2.5-flash")."""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise ValueError(f"model {spec!r}: expected <provider>:<model> with provider one of {', '.join(PROVIDERS)}")
    return provider, model


class OpenAICompatClient(LLMClient):
    def __init__(self, provider: str, model: str, api_key: str | None = None, max_tokens: int = 2048,
                 temperature: float = 0.1, timeout: float = 120.0, transport: Any = None) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider, self.model = provider, model
        self.model_id = f"{provider}:{model}"
        self.base_url, self.key_var = PROVIDERS[provider]
        self.api_key = api_key if api_key is not None else provider_key(provider)
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self._transport = transport  # tests pass an httpx.MockTransport

    # ------------------------------------------------------------------ HTTP
    def _client(self):
        import httpx

        if not self.api_key:
            raise LLMClientError(f"{self.key_var} is not set: add it to .env to use {self.model_id}.")
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport,
                            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def _body(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        return {"model": self.model, "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": self.max_tokens, "temperature": self.temperature, "stream": stream}

    def _error(self, status: int, text: str) -> LLMClientError:
        name = f"{self.provider} ({self.model})"
        if status in (401, 403):
            return LLMClientError(f"{name} rejected the key in {self.key_var} (HTTP {status}); check it in .env.")
        if status == 404:
            return LLMClientError(f"{name}: model not found (HTTP 404); check the model name.")
        if status == 429:
            daily = bool(re.search(r"per ?day|PerDay|daily|quota", text, re.IGNORECASE))
            what = f"The {self.provider} daily usage limit has been reached" if daily else \
                f"{self.provider} is receiving too many requests right now (rate limit)"
            m = re.search(r'"retry[_-]?after[^0-9]*(\d+)|retry in ([\d.]+)s', text, re.IGNORECASE)
            secs = next((g for g in (m.groups() if m else ()) if g), None)
            wait = f"Try again in about {max(1, round(float(secs) / 60))} minutes" if secs else "Try again later"
            return LLMRateLimitError(f"{what}. {wait}.")
        return LLMClientError(f"{name} returned HTTP {status}: {text[:300]}")

    # ------------------------------------------------------------------ LLMClient
    def generate(self, messages: list[Message]) -> str:
        with self._client() as http:
            r = http.post("/chat/completions", json=self._body(messages, False))
        if r.status_code != 200:
            raise self._error(r.status_code, r.text)
        choice = r.json()["choices"][0]
        return choice["message"].get("content") or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        with self._client() as http, http.stream("POST", "/chat/completions", json=self._body(messages, True)) as r:
            if r.status_code != 200:
                raise self._error(r.status_code, r.read().decode("utf-8", "replace"))
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
