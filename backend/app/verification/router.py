"""VerifyRouter: POST /api/verify and GET /api/verify/sections."""

from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.llm_client import LLMClientError
from app.verification.fact_extractor import ExtractionError
from app.verification.verification_service import DISCLAIMER, VerificationService


class VerifyRequest(BaseModel):
    case_text: str = Field(min_length=40, max_length=20000)
    cited_section: str = Field(min_length=1, max_length=10)
    model: str | None = Field(default=None, max_length=120)  # None: the free models in turn, until one answers


class VerifyRouter:
    def __init__(self, service: VerificationService, zk=None,
                 pick: Callable[[str], VerificationService | None] | None = None) -> None:
        self.service = service
        self.pick = pick  # the service for one chosen model, alone: its failure is reported, never swapped
        self.zk = zk  # optional ZkService: ?prove=true adds a zero-knowledge proof of the verdict
        self.router = APIRouter(prefix="/api/verify", tags=["verify"])
        self.router.add_api_route("", self.verify, methods=["POST"])
        self.router.add_api_route("/sections", self.sections, methods=["GET"])

    def sections(self) -> dict[str, Any]:
        return {
            "disclaimer": DISCLAIMER,
            "sections": [
                {"section": s.section, "title": s.title,
                 "elements": [{"predicate": e.predicate, "description": e.description, "required": list(e.required)} for e in s.elements]}
                for s in self.service.sections
            ],
        }

    def verify(self, req: VerifyRequest, prove: bool = False) -> dict[str, Any]:
        service = self.service
        if req.model:
            service = self.pick(req.model) if self.pick else None
            if service is None:
                raise HTTPException(422, f"unknown model {req.model!r}")
        try:
            result = service.verify(req.case_text, req.cited_section).to_dict()
        except ValueError as e:
            if isinstance(e, ExtractionError):
                raise HTTPException(502, str(e)) from None
            raise HTTPException(422, str(e)) from None
        except LLMClientError as e:
            raise HTTPException(503, str(e)) from None
        except httpx.HTTPError as e:  # a provider that hung or dropped the connection
            raise HTTPException(503, f"The model did not answer ({type(e).__name__}).") from None
        if prove:
            if self.zk is None:
                result["proof_error"] = "zero-knowledge proofs are not available on this server"
            else:
                try:
                    result["proof"] = self.zk.prove_verdict(result["facts"], result["cited_section"])
                except Exception as e:  # the verdict is still useful without a proof
                    result["proof_error"] = str(e) or type(e).__name__
        return result
