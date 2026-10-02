"""Groq usage-limit and auth errors reach the user as plain sentences, not raw API JSON."""

import json
from types import SimpleNamespace

import groq
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.chat_router import ChatRouter
from app.core.llm_client import GroqClient, LLMClientError, LLMRateLimitError, groq_error
from tests.conftest import StubLLM

# The exact error the UI showed after a day of testing on the free tier.
DAILY = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
         "on tokens per day (TPD): Limit 200000, Used 198893, Requested 2422. Please try again in 9m28.08s. "
         "Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing")
PER_MINUTE = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
              "on tokens per minute (TPM): Limit 8000, Used 4613, Requested 4042. Please try again in 4.9125s.")


def _status_error(cls: type, status: int, message: str) -> Exception:
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    return cls(f"Error code: {status} - {body}", response=response, body=body)


class FailingCompletions:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def create(self, **_: object) -> None:
        raise self.exc


def _client(exc: Exception) -> GroqClient:
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions(exc)))
    return GroqClient(api_key="unused", client=fake)


def test_daily_limit_message_is_readable() -> None:
    err = groq_error(_status_error(groq.RateLimitError, 429, DAILY))
    assert isinstance(err, LLMRateLimitError)
    assert str(err) == ("The hosted model's daily usage limit (Groq free tier) has been reached. "
                        "Try again in about 10 minutes.")
    assert "org_" not in str(err) and "{" not in str(err)


def test_per_minute_limit_message() -> None:
    err = groq_error(_status_error(groq.RateLimitError, 429, PER_MINUTE))
    assert str(err) == ("The hosted model is receiving too many requests right now (Groq rate limit). "
                        "Try again in about 5 seconds.")


def test_auth_error_and_passthrough() -> None:
    err = groq_error(_status_error(groq.AuthenticationError, 401, "Invalid API Key"))
    assert isinstance(err, LLMClientError) and "GROQ_API_KEY" in str(err)
    other = ValueError("unrelated")
    assert groq_error(other) is other


@pytest.mark.parametrize("call", ["generate", "stream"])
def test_client_raises_the_readable_error(call: str) -> None:
    client = _client(_status_error(groq.RateLimitError, 429, DAILY))
    with pytest.raises(LLMRateLimitError, match="daily usage limit"):
        result = getattr(client, call)([])
        list(result) if call == "stream" else result


class RateLimitedLLM(StubLLM):
    model_id = "openai/gpt-oss-120b"

    def stream(self, messages):  # type: ignore[override]
        raise groq_error(_status_error(groq.RateLimitError, 429, DAILY))
        yield ""  # pragma: no cover


def _error_message(models: dict, model: str | None = None) -> str:
    router = ChatRouter(retriever=None, llm=models["groq"], models=models, default_model="groq")  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(router.router)
    body = TestClient(app).post("/api/chat", json={"question": "hello", "mode": "general", "model": model}).text
    events = {b.split("\n")[0][7:]: json.loads(b.split("\n")[1][6:]) for b in body.strip().split("\n\n")}
    return events["error"]["message"]


def test_chat_suggests_the_local_model_only_when_it_exists() -> None:
    with_adapter = _error_message({"groq": RateLimitedLLM(), "adapter": StubLLM()})
    assert with_adapter.startswith("The hosted model's daily usage limit") and "Tuned (LoRA)" in with_adapter
    assert "Tuned (LoRA)" not in _error_message({"groq": RateLimitedLLM()})
