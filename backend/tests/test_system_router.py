from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.system_router import SystemRouter


def test_overview_reports_live_counts_and_results() -> None:
    chat = SimpleNamespace(retriever=SimpleNamespace(store=SimpleNamespace(count=lambda: 812)),
                           judgments=SimpleNamespace(store=SimpleNamespace(count=lambda: 28055)),
                           models={"groq": SimpleNamespace(model_id="openai/gpt-oss-120b")}, default_model="groq",
                           rewriter=object(), offline=False)
    app = FastAPI()
    app.include_router(SystemRouter(get_settings(), chat, SimpleNamespace(sections=[1] * 7)).router)
    body = TestClient(app).get("/api/system").json()
    live = body["live"]
    assert live["ipc_chunks"] == 812 and live["judgment_passages"] == 28055 and live["rule_sections"] == 7
    assert live["default_model"] == "openai/gpt-oss-120b" and live["query_rewrite"] is True
    assert live["bns_sections"] > 500
    assert all(e["source"] for entries in body["results"]["stages"].values() for e in entries)


def test_a_missing_part_shows_as_null() -> None:
    app = FastAPI()
    app.include_router(SystemRouter(get_settings(), SimpleNamespace(), None).router)
    live = TestClient(app).get("/api/system").json()["live"]
    assert live["ipc_chunks"] is None and live["rule_sections"] is None
