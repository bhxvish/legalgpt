"""Signed answer records (attestation, not zero knowledge).

The answering LLM is far too large to prove, so each legal answer gets a record signed with the
server's Ed25519 key: hashes of the question, the exact prompt, the answer, the cited chunks
(Merkle leaves of the committed corpus) and the model / adapter that produced it. It proves the
answer came from this server with these inputs and has not been altered since - not that the
model computed it correctly or that it is right.
"""

import base64
import time
from pathlib import Path
from typing import Any

from app.zk.commit import canonical, sha256_hex

RECORD_TYPE = "legalgpt.answer.v1"


class Attestor:
    def __init__(self, key_path: Path, context: dict[str, Any] | None = None) -> None:
        """`context`: fixed facts added to every record (corpus root, adapter hash, ...)."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key_path = Path(key_path)
        if key_path.exists():
            self._key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        else:  # first run: create the server's signing key (git-ignored, never published)
            self._key = Ed25519PrivateKey.generate()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(self._key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption()))
        self.public_key = base64.b64encode(self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        self.context = dict(context or {})

    def sign(self, record: dict[str, Any]) -> dict[str, Any]:
        body = {"type": RECORD_TYPE, "signed_at": int(time.time()), **self.context, **record}
        sig = self._key.sign(canonical(body))
        return {"record": body, "signature": base64.b64encode(sig).decode(), "public_key": self.public_key,
                "alg": "Ed25519"}

    def answer_record(self, *, question: str, prompt: Any, answer: str, sources: list, model_id: str,
                      refused: bool, follow_up: bool, leaf_of, rewrite: str | None = None) -> dict[str, Any]:
        extra = {"retrieval_rewrite": rewrite} if rewrite else {}  # the LLM rewrite that steered retrieval
        return self.sign({**extra,
            "question_sha256": sha256_hex(question.encode()),
            "prompt_sha256": sha256_hex(canonical(prompt)),
            "answer_sha256": sha256_hex(answer.encode()),
            "model": model_id,
            "refused": refused,
            "follow_up_context": follow_up,  # retrieval also searched with the previous question
            "sources": [{"marker": s.marker, "chunk_id": s.chunk_id, "leaf": leaf_of(s.chunk_id)} for s in sources],
        })


def verify_attestation(att: dict[str, Any], trusted_public_key: str | None = None,
                       question: str | None = None, answer: str | None = None) -> dict[str, Any]:
    """Signature check, plus optional checks that a given question / answer text is the one signed."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    checks = []
    key = att.get("public_key", "")
    if trusted_public_key is not None:
        checks.append({"name": "signed by this server's key", "ok": key == trusted_public_key})
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(key)).verify(
            base64.b64decode(att["signature"]), canonical(att["record"]))
        sig_ok = True
    except (InvalidSignature, ValueError, KeyError, TypeError):
        sig_ok = False
    checks.append({"name": "signature valid (record not altered)", "ok": sig_ok})
    rec = att.get("record", {})
    if question is not None:
        checks.append({"name": "question matches", "ok": rec.get("question_sha256") == sha256_hex(question.encode())})
    if answer is not None:
        checks.append({"name": "answer text matches", "ok": rec.get("answer_sha256") == sha256_hex(answer.encode())})
    return {"valid": all(c["ok"] for c in checks), "checks": checks, "record": rec}
