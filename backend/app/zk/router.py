"""ZkRouter: /api/zk - commitments manifest, proofs, and verification of proofs and signed answers.

  GET  /api/zk/manifest              published commitments + what this server can prove / verify
  GET  /api/zk/inclusion/{chunk_id}  Merkle inclusion proof of a chunk in the committed corpus
  POST /api/zk/prove/verdict         {"facts", "section"}  -> proof bundle (facts stay private)
  POST /api/zk/prove/retrieval       {"question"}          -> proof bundle over all chunks
  POST /api/zk/verify                {"bundle", "facts"?, "question"?, "answer"?} -> checks
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.zk.attest import verify_attestation
from app.zk.prover import ZkError, ZkUnavailable
from app.zk.service import ZkService


class ProveVerdictRequest(BaseModel):
    facts: dict[str, Any]
    section: str | None = Field(default=None, max_length=10)


class ProveRetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class VerifyBundleRequest(BaseModel):
    bundle: dict[str, Any]
    facts: dict[str, Any] | None = None  # optional disclosure, checked against the commitment
    question: str | None = None  # for attestations: check the signed question / answer text
    answer: str | None = None


class ZkRouter:
    def __init__(self, service: ZkService) -> None:
        self.service = service
        self.router = APIRouter(prefix="/api/zk", tags=["zk"])
        self.router.add_api_route("/manifest", self.manifest, methods=["GET"])
        self.router.add_api_route("/inclusion/{chunk_id}", self.inclusion, methods=["GET"])
        self.router.add_api_route("/prove/verdict", self.prove_verdict, methods=["POST"])
        self.router.add_api_route("/prove/retrieval", self.prove_retrieval, methods=["POST"])
        self.router.add_api_route("/verify", self.verify, methods=["POST"])

    def manifest(self) -> dict[str, Any]:
        return self.service.manifest

    def inclusion(self, chunk_id: str) -> dict[str, Any]:
        try:
            return self.service.inclusion(chunk_id)
        except KeyError:
            raise HTTPException(404, f"chunk {chunk_id!r} is not in the committed corpus") from None

    def prove_verdict(self, req: ProveVerdictRequest) -> dict[str, Any]:
        try:
            return self.service.prove_verdict(req.facts, req.section)
        except ZkUnavailable as e:
            raise HTTPException(503, str(e)) from None
        except ZkError as e:
            raise HTTPException(500, str(e)) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    def prove_retrieval(self, req: ProveRetrievalRequest) -> dict[str, Any]:
        try:
            return self.service.prove_retrieval(req.question)
        except ZkUnavailable as e:
            raise HTTPException(503, str(e)) from None
        except ZkError as e:
            raise HTTPException(500, str(e)) from None

    def verify(self, req: VerifyBundleRequest) -> dict[str, Any]:
        b = req.bundle
        try:
            if "signature" in b and "record" in b:
                trusted = self.service.attestor.public_key if self.service.attestor else None
                return {"kind": "attestation", **verify_attestation(b, trusted, req.question, req.answer)}
            if b.get("kind") == "verdict":
                return self.service.verify_verdict(b, req.facts)
            if b.get("kind") == "retrieval":
                return self.service.verify_retrieval(b)
        except ZkUnavailable as e:
            raise HTTPException(503, str(e)) from None
        except (KeyError, TypeError, ValueError) as e:
            return {"kind": b.get("kind", "unknown"), "valid": False,
                    "checks": [{"name": "bundle is well-formed", "ok": False, "detail": f"{type(e).__name__}: {e}"}]}
        raise HTTPException(422, "unknown bundle: expected a verdict or retrieval proof, or a signed answer")
