"""VerifyRouter: POST /api/verify and GET /api/verify/sections."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.llm_client import LLMClientError
from app.verification.fact_extractor import ExtractionError
from app.verification.verification_service import DISCLAIMER, VerificationService


class VerifyRequest(BaseModel):
    case_text: str = Field(min_length=40, max_length=20000)
    cited_section: str = Field(min_length=1, max_length=10)


class VerifyRouter:
    def __init__(self, service: VerificationService, zk=None) -> None:
        self.service = service
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
        try:
            result = self.service.verify(req.case_text, req.cited_section).to_dict()
        except ValueError as e:
            if isinstance(e, ExtractionError):
                raise HTTPException(502, str(e)) from None
            raise HTTPException(422, str(e)) from None
        except LLMClientError as e:
            raise HTTPException(503, str(e)) from None
        if prove:
            if self.zk is None:
                result["proof_error"] = "zero-knowledge proofs are not available on this server"
            else:
                try:
                    result["proof"] = self.zk.prove_verdict(result["facts"], result["cited_section"])
                except Exception as e:  # the verdict is still useful without a proof
                    result["proof_error"] = str(e) or type(e).__name__
        return result
