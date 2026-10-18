"""GET /api/system: what the System overview page shows.

`live`: counts read from this server's indexes and stores right now (each null if unavailable).
`results`: the headline evaluation results, from docs/reports/system_results.json; every entry names
the section of docs/reports/paper_analysis.md (or the report file) it comes from.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter

from app.config import REPO_ROOT, Settings

logger = logging.getLogger(__name__)

RESULTS = REPO_ROOT / "docs" / "reports" / "system_results.json"


def _safe(fn) -> Any:
    try:
        return fn()
    except Exception:  # a missing index or store: the page shows "—"
        logger.debug("system overview: a live count failed", exc_info=True)
        return None


class SystemRouter:
    def __init__(self, settings: Settings, chat_router: Any = None, verifier: Any = None) -> None:
        self.settings, self.chat, self.verifier = settings, chat_router, verifier
        self.router = APIRouter(prefix="/api", tags=["system"])
        self.router.add_api_route("/system", self.overview, methods=["GET"])

    def live(self) -> dict[str, Any]:
        from app.annotation.store import AnnotationStore
        from app.core.bns import _map
        from app.labeling_assistant.intake import latest_checkpoint

        s, chat = self.settings, self.chat
        judgments = getattr(chat, "judgments", None)
        library = REPO_ROOT / "data" / "judgment_library" / "sc"
        return {
            "ipc_chunks": _safe(lambda: chat.retriever.store.count()),
            "judgment_passages": _safe(lambda: judgments.store.count()) if judgments is not None else None,
            "library_judgments": _safe(lambda: len(list(library.glob("*.json")))),
            "annotated_judgments": _safe(lambda: len(AnnotationStore(s.annotation_store_dir).case_ids())),
            "classifier": _safe(lambda: latest_checkpoint(s.rrl_model_dir).name),
            "chat_models": _safe(lambda: [m.model_id for m in chat.models.values()]),
            "default_model": _safe(lambda: chat.models[chat.default_model].model_id),
            "rule_sections": _safe(lambda: len(self.verifier.sections)) if self.verifier is not None else None,
            "bns_sections": _safe(lambda: len(_map()["ipc_to_bns"])),
            "query_rewrite": _safe(lambda: chat.rewriter is not None),
            "offline": _safe(lambda: bool(chat.offline)),
        }

    def overview(self) -> dict[str, Any]:
        results = _safe(lambda: json.loads(RESULTS.read_text(encoding="utf-8"))) or {}
        return {"live": self.live(), "results": results}
