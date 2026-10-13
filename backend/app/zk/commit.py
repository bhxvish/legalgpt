"""Commitments: hashes that pin down exactly which corpus, rules and model a proof or a signed
answer refers to.

  corpus root      Merkle root (SHA-256) over the IPC chunks, one leaf per chunk in chunk_id
                   order; inclusion proofs show a cited chunk is part of the committed corpus
  embeddings       SHA-256 of the float32 embedding matrix the retrieval circuit is built from
  rules            SHA-256 of every rules/*.pl file, and of the verdict circuit's element matrix
  adapter          SHA-256 of the LoRA adapter's files (weights + config)

They are collected in a manifest (scripts/zk_setup.py writes docs/zk/manifest.json).
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.core.models import Chunk


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, no whitespace) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def chunk_leaf(chunk: Chunk) -> str:
    fields = {"chunk_id": chunk.chunk_id, "citation_path": chunk.citation_path, "section": chunk.section,
              "role": chunk.role, "title": chunk.title, "text": chunk.text}
    return sha256_hex(b"\x00" + canonical(fields))  # 0x00 / 0x01 prefixes separate leaves from nodes


def _node(left: str, right: str) -> str:
    return sha256_hex(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


class MerkleTree:
    """Binary SHA-256 Merkle tree; an odd node at the end of a level is paired with itself."""

    def __init__(self, leaves: Sequence[str]) -> None:
        if not leaves:
            raise ValueError("a Merkle tree needs at least one leaf")
        self.levels: list[list[str]] = [list(leaves)]
        while len(self.levels[-1]) > 1:
            level = self.levels[-1]
            self.levels.append([_node(level[i], level[i + 1] if i + 1 < len(level) else level[i])
                                for i in range(0, len(level), 2)])

    @property
    def root(self) -> str:
        return self.levels[-1][0]

    def proof(self, index: int) -> list[dict[str, str]]:
        """Sibling hashes from the leaf up; `side` says where the sibling sits."""
        path = []
        for level in self.levels[:-1]:
            sib = index ^ 1
            sibling = level[sib] if sib < len(level) else level[index]
            path.append({"side": "left" if sib < index else "right", "hash": sibling})
            index //= 2
        return path


def verify_inclusion(leaf: str, path: Iterable[dict[str, str]], root: str) -> bool:
    h = leaf
    for step in path:
        h = _node(step["hash"], h) if step["side"] == "left" else _node(h, step["hash"])
    return h == root


def corpus_order(chunks: Iterable[Chunk]) -> list[Chunk]:
    """The committed order of chunks: by chunk_id. The retrieval circuit uses the same order."""
    return sorted(chunks, key=lambda c: c.chunk_id)


def corpus_tree(chunks: Iterable[Chunk]) -> MerkleTree:
    return MerkleTree([chunk_leaf(c) for c in corpus_order(chunks)])


def embeddings_hash(matrix: np.ndarray) -> str:
    m = np.ascontiguousarray(matrix, dtype="<f4")
    return sha256_hex(str(m.shape).encode() + m.tobytes())


def files_hash(paths: Iterable[Path], text: bool = False) -> dict[str, Any]:
    """Per-file SHA-256 plus one combined hash over (name, hash) pairs. `text` hashes with LF line
    endings, so a Windows checkout (CRLF) and a Linux one give the same hash."""
    files = {}
    for p in sorted(Path(x) for x in paths):
        data = p.read_bytes()
        files[p.name] = sha256_hex(data.replace(b"\r\n", b"\n") if text else data)
    return {"files": files, "sha256": sha256_hex(canonical(files))}


def rules_hash(rules_dir: Path) -> dict[str, Any]:
    return files_hash(sorted(Path(rules_dir).glob("*.pl")), text=True)


def adapter_hash(adapter_dir: Path) -> dict[str, Any]:
    """Hash of the files that determine the adapter's behaviour (not training logs)."""
    adapter_dir = Path(adapter_dir)
    keep = [p for p in adapter_dir.iterdir() if p.is_file() and (p.suffix in (".safetensors", ".bin")
                                                                  or p.name in ("adapter_config.json",))]
    out = files_hash(keep)
    out["name"] = adapter_dir.name
    return out
