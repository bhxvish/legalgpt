"""OpenAI-compatible providers (Gemini, OpenAI, OpenRouter) with a mock transport."""

import json

import httpx
import pytest

from app.core.llm_client import LLMClientError, LLMRateLimitError
from app.core.models import Message
from app.core.openai_compat import PROVIDERS, OpenAICompatClient, parse_spec, provider_key

MSGS = [Message("system", "s"), Message("user", "What is the punishment for theft?")]


def _client(handler, provider: str = "gemini", key: str = "test-key") -> OpenAICompatClient:
    client = OpenAICompatClient(provider, "gemini-2.5-flash", api_key=key, transport=httpx.MockTransport(handler))
    client.retry_waits = (0, 0, 0)
    return client


def test_generate_sends_an_openai_style_request() -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Section 379 [1]."}}]})

    assert _client(handler).generate(MSGS) == "Section 379 [1]."
    assert seen["url"] == PROVIDERS["gemini"][0] + "/chat/completions"
    assert seen["auth"] == "Bearer test-key" and seen["body"]["model"] == "gemini-2.5-flash"
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"] and seen["body"]["stream"] is False


def test_stream_yields_deltas_until_done() -> None:
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n" for t in ["Section ", "379"])
    body += "data: [DONE]\n\n"
    client = _client(lambda req: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    assert "".join(client.stream(MSGS)) == "Section 379"


@pytest.mark.parametrize("status, text, error, phrase", [
    (401, "bad key", LLMClientError, "GEMINI_API_KEY"),
    (404, '[{"error": {"message": "use models/gemini-3.8-flash"}}]', LLMClientError, "not found.*gemini-3.8-flash"),
    (503, '{"error": {"message": "high demand"}}', LLMClientError, "overloaded.*high demand"),
    (429, '{"error": "Resource has been exhausted (e.g. check quota). GenerateRequestsPerDay"}', LLMRateLimitError, "daily usage limit"),
    (429, '{"error": "Rate limit exceeded"}', LLMRateLimitError, "too many requests"),
])
def test_errors_are_readable(status: int, text: str, error: type, phrase: str) -> None:
    with pytest.raises(error, match=phrase):
        _client(lambda req: httpx.Response(status, text=text)).generate(MSGS)


def test_overloaded_replies_are_retried() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert _client(handler).generate(MSGS) == "ok" and len(calls) == 3


def test_a_non_json_reply_is_a_clear_error() -> None:
    with pytest.raises(LLMClientError, match="unexpected reply"):
        _client(lambda req: httpx.Response(200, text="OK\r\n")).generate(MSGS)


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert provider_key("openrouter", env_file=tmp_path / "none.env") == ""
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="sk-or-example"\n', encoding="utf-8")
    assert provider_key("openrouter", env_file=tmp_path / ".env") == "sk-or-example"
    with pytest.raises(LLMClientError, match="OPENROUTER_API_KEY is not set"):
        OpenAICompatClient("openrouter", "some/model", api_key="").generate(MSGS)


def test_parse_spec() -> None:
    assert parse_spec("openrouter:meta/llama:free") == ("openrouter", "meta/llama:free")
    for bad in ("nope:model", "github:openai/gpt-4o-mini"):  # GitHub Models was retired
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_extra_models_appear_in_the_model_list(monkeypatch) -> None:
    from app.config import Settings
    from app.main import build_models

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = Settings(EXTRA_MODELS="gemini:gemini-2.5-flash, openrouter:some/model, bad")
    models, notes, _ = build_models(s)
    assert "gemini:gemini-2.5-flash" in models and models["gemini:gemini-2.5-flash"].model_id == "gemini:gemini-2.5-flash"
    assert "openrouter:some/model" not in models and "OPENROUTER_API_KEY" in notes["openrouter:some/model"]


def test_openai_reasoning_models_get_their_parameters() -> None:
    old = OpenAICompatClient("openai", "gpt-4o-mini", api_key="k")._body(MSGS, False)
    new = OpenAICompatClient("openai", "gpt-5.4-mini", api_key="k")._body(MSGS, False)
    assert "max_tokens" in old and "temperature" in old
    assert "max_tokens" not in new and "temperature" not in new and new["max_completion_tokens"] == 2048
