"""OpenAI-compatible providers (Gemini, GitHub Models, OpenAI, OpenRouter) with a mock transport."""

import json

import httpx
import pytest

from app.core.llm_client import LLMClientError, LLMRateLimitError
from app.core.models import Message
from app.core.openai_compat import PROVIDERS, OpenAICompatClient, parse_spec, provider_key

MSGS = [Message("system", "s"), Message("user", "What is the punishment for theft?")]


def _client(handler, provider: str = "gemini", key: str = "test-key") -> OpenAICompatClient:
    return OpenAICompatClient(provider, "gemini-2.5-flash", api_key=key, transport=httpx.MockTransport(handler))


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
    (404, "no model", LLMClientError, "model not found"),
    (429, '{"error": "Resource has been exhausted (e.g. check quota). GenerateRequestsPerDay"}', LLMRateLimitError, "daily usage limit"),
    (429, '{"error": "Rate limit exceeded"}', LLMRateLimitError, "too many requests"),
])
def test_errors_are_readable(status: int, text: str, error: type, phrase: str) -> None:
    with pytest.raises(error, match=phrase):
        _client(lambda req: httpx.Response(status, text=text)).generate(MSGS)


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert provider_key("github", env_file=tmp_path / "none.env") == ""
    (tmp_path / ".env").write_text('GITHUB_TOKEN="ghp_example"\n', encoding="utf-8")
    assert provider_key("github", env_file=tmp_path / ".env") == "ghp_example"
    with pytest.raises(LLMClientError, match="GITHUB_TOKEN is not set"):
        OpenAICompatClient("github", "openai/gpt-4o-mini", api_key="").generate(MSGS)


def test_parse_spec() -> None:
    assert parse_spec("github:openai/gpt-4o-mini") == ("github", "openai/gpt-4o-mini")
    with pytest.raises(ValueError):
        parse_spec("nope:model")


def test_extra_models_appear_in_the_model_list(monkeypatch) -> None:
    from app.config import Settings
    from app.main import build_models

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(EXTRA_MODELS="gemini:gemini-2.5-flash, openai:gpt-4o-mini, bad")
    models, notes, _ = build_models(s)
    assert "gemini:gemini-2.5-flash" in models and models["gemini:gemini-2.5-flash"].model_id == "gemini:gemini-2.5-flash"
    assert "openai:gpt-4o-mini" not in models and "OPENAI_API_KEY" in notes["openai:gpt-4o-mini"]
