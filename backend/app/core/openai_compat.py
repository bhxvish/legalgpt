"""OpenAICompatClient: any chat API that speaks the OpenAI /chat/completions format.

One client covers several providers (free tiers marked):
  gemini      Google Gemini via its OpenAI-compatible endpoint      GEMINI_API_KEY      (free tier in AI Studio)
  openai      OpenAI                                                OPENAI_API_KEY      (paid)
  openrouter  OpenRouter (many providers; some ":free" models)      OPENROUTER_API_KEY  (pay as you go)

A model is named "<provider>:<model>", e.g. "gemini:gemini-3.5-flash-lite" or "openai:gpt-4o-mini".
(GitHub Models was retired by GitHub on 30 July 2026 and is not supported.)
Keys are read from the environment or .env and never logged. httpx only, no provider SDKs.
A key variable may list several keys, comma-separated (GEMINI_API_KEY=key1,key2): on a rate or daily
limit the next key is used, as GroqClient does with GROQ_API_KEY.
"""

import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from app.core.llm_client import LLMClient, LLMClientError, LLMRateLimitError
from app.core.models import Message

RETRYABLE = {500, 502, 503, 504}


def _message(text: str) -> str:
    """The error message from a provider's JSON error body, else the start of the text."""
    try:
        body = json.loads(text)
        body = body[0] if isinstance(body, list) else body
        return str(body["error"]["message"])[:300]
    except (ValueError, KeyError, IndexError, TypeError):
        return text[:300]


PROVIDERS: dict[str, tuple[str, str]] = {  # name -> (base URL, key variable)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
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
                 temperature: float = 0.1, timeout: float = 75.0, transport: Any = None,
                 extra_body: dict[str, Any] | None = None) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.provider, self.model = provider, model
        self.model_id = f"{provider}:{model}"
        self.base_url, self.key_var = PROVIDERS[provider]
        keys = api_key if api_key is not None else provider_key(provider)
        self.api_keys = [k.strip() for k in (keys or "").split(",") if k.strip()]
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.exhausted: dict[str, float] = {}  # key -> when its daily limit was hit (skipped for 6 hours)
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self._transport = transport  # tests pass an httpx.MockTransport
        self.extra_body = extra_body or {}  # provider-specific fields, e.g. OpenRouter's "provider" routing
        self.retry_waits = (3.0, 8.0)  # after "overloaded" (500/502/503/504) replies; then the next model takes over
        self.usage = {"in": 0, "out": 0}  # billed tokens so far (generate only; out includes reasoning)

    # ------------------------------------------------------------------ HTTP
    def _client(self):
        import httpx

        if not self.api_key:
            raise LLMClientError(f"{self.key_var} is not set: add it to .env to use {self.model_id}.")
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport,
                            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def _body(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        body = {"model": self.model, "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": self.max_tokens, "temperature": self.temperature, "stream": stream}
        if self.provider == "openai" and re.match(r"(gpt-5|o\d)", self.model):
            # OpenAI's reasoning models take max_completion_tokens (which includes reasoning tokens)
            # and only the default temperature
            body["max_completion_tokens"] = body.pop("max_tokens")
            del body["temperature"]
        return body | self.extra_body

    def _error(self, status: int, text: str) -> LLMClientError:
        name = f"{self.provider} ({self.model})"
        if status in (401, 403):
            return LLMClientError(f"{name} rejected the key in {self.key_var} (HTTP {status}); check it in .env.")
        if status == 404:  # the provider's message often names the replacement model
            return LLMClientError(f"{name}: model not found (HTTP 404); check the model name. {_message(text)}")
        if status == 429:
            daily = bool(re.search(r"per ?day|PerDay|daily", text, re.IGNORECASE))
            what = f"The {self.provider} daily usage limit has been reached" if daily else \
                f"{self.provider} is receiving too many requests right now (rate limit)"
            m = re.search(r'"retry[_-]?after[^0-9]*(\d+)|retry in ([\d.]+)s', text, re.IGNORECASE)
            secs = next((g for g in (m.groups() if m else ()) if g), None)
            wait = f"Try again in about {max(1, round(float(secs) / 60))} minutes" if secs else "Try again later"
            return LLMRateLimitError(f"{what}. {wait}.")
        if status in RETRYABLE:
            return LLMClientError(f"{name} is overloaded (HTTP {status}); try again later. {_message(text)}")
        return LLMClientError(f"{name} returned HTTP {status}: {_message(text)}")

    def _post(self, http, messages: list[Message]):
        for wait in (*self.retry_waits, None):
            try:
                r = http.post("/chat/completions", json=self._body(messages, False))
            except httpx.TransportError as e:  # hung or dropped: down, like an error reply, so a fallback can step in
                raise LLMClientError(f"{self.provider} ({self.model}) did not answer ({type(e).__name__}).") from e
            if r.status_code not in RETRYABLE or wait is None:
                return r
            time.sleep(wait)

    def _next_key(self, error: Exception | None = None) -> bool:
        """Switch to the next usable key after a rate limit; False when there is none. A key whose
        daily limit was hit is skipped for 6 hours, so a per-minute limit on the last good key does not
        send the next call to a key that is out for the day."""
        if error is not None and "daily" in str(error).lower():
            self.exhausted[self.api_key] = time.time()
        usable = [k for k in self.api_keys if k != self.api_key and time.time() - self.exhausted.get(k, 0) > 6 * 3600]
        if not usable:
            return False
        i = self.api_keys.index(self.api_key) if self.api_key in self.api_keys else -1
        self.api_key = next(k for k in self.api_keys[i + 1:] + self.api_keys[:i + 1] if k in usable)
        import logging

        logging.getLogger(__name__).warning("%s rate limit: switched to API key %d of %d", self.provider,
                                            self.api_keys.index(self.api_key) + 1, len(self.api_keys))
        return True

    # ------------------------------------------------------------------ LLMClient
    def generate(self, messages: list[Message]) -> str:
        for _ in range(max(1, len(self.api_keys))):
            try:
                return self._generate(messages)
            except LLMRateLimitError as e:
                if not self._next_key(e):
                    raise
        return self._generate(messages)

    def _generate(self, messages: list[Message]) -> str:
        with self._client() as http:
            r = self._post(http, messages)
        if r.status_code != 200:
            raise self._error(r.status_code, r.text)
        try:
            reply = r.json()
            choice = reply["choices"][0]
        except (ValueError, KeyError, IndexError):
            raise LLMClientError(f"{self.model_id}: unexpected reply (not a chat completion): {r.text[:120]!r}")
        used = reply.get("usage") or {}
        self.usage["in"] += used.get("prompt_tokens") or 0
        self.usage["out"] += used.get("completion_tokens") or 0
        return (choice.get("message") or {}).get("content") or ""  # a blocked reply has no message

    def generate_brief(self, messages: list[Message]) -> str:
        """generate() with low reasoning effort on reasoning models (short classification calls such
        as the query rewrite), as GroqClient.generate_brief does; other models answer as usual."""
        low = None
        if self.provider == "openai" and re.match(r"(gpt-5|o\d)", self.model):
            low = {"reasoning_effort": "low"}
        elif self.provider == "openrouter" and "gpt-oss" in self.model:
            low = {"reasoning": {"effort": "low"}}  # OpenRouter's form of the same setting
        if low:
            saved = self.extra_body
            self.extra_body = saved | low
            try:
                return self.generate(messages)
            finally:
                self.extra_body = saved
        return self.generate(messages)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for attempt in range(max(1, len(self.api_keys))):
            started = False
            try:
                for piece in self._stream(messages):
                    started = True
                    yield piece
                return
            except LLMRateLimitError as e:  # a limit is reported before the first token: try the next key
                if started or attempt + 1 >= len(self.api_keys) or not self._next_key(e):
                    raise

    def _stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            for wait in (*self.retry_waits, None):  # retry only before the first token
                with self._client() as http, http.stream("POST", "/chat/completions", json=self._body(messages, True)) as r:
                    if r.status_code in RETRYABLE and wait is not None:
                        time.sleep(wait)
                        continue
                    if r.status_code != 200:
                        raise self._error(r.status_code, r.read().decode("utf-8", "replace"))
                    yield from self._deltas(r)
                    return
        except httpx.TransportError as e:
            raise LLMClientError(f"{self.provider} ({self.model}) did not answer ({type(e).__name__}).") from e

    @staticmethod
    def _deltas(r) -> Iterator[str]:
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
