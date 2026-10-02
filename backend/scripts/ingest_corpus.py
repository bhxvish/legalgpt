"""Parse a legal text (PDF or .txt), chunk it, embed it, and upsert into ChromaDB.

Usage (from the repo root):
    python backend/scripts/ingest_corpus.py data/raw_corpus/the_indian_penal_code,_1860.pdf --reset
    python backend/scripts/ingest_corpus.py <file> --dry-run --show 304A,302
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.parser import LegalDocumentParser  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--doc-id", default="IPC")
    ap.add_argument("--max-chunk-chars", type=int, default=1400)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--reset", action="store_true", help="drop the collection before ingesting")
    ap.add_argument("--dry-run", action="store_true", help="parse and chunk only; no embedding/upsert")
    ap.add_argument("--show", default="", help="comma-separated section numbers whose chunks to print")
    ap.add_argument("--dump", type=Path, help="write all chunks to this JSONL file for inspection")
    args = ap.parse_args()

    settings = get_settings()
    parser = LegalDocumentParser(doc_id=args.doc_id)
    t0 = time.perf_counter()
    if args.path.suffix.lower() == ".pdf":
        text = parser.extract_pdf_text(args.path)
    else:
        text = args.path.read_text(encoding="utf-8")
    nodes = parser.parse_text(text, doc_id=args.doc_id)
    toc = parser.toc_sections(text)
    if toc:
        found = {n.section for n in nodes if n.kind == "section"}
        missing = [s for s in toc if s not in found]
        extra = sorted(found - set(toc))
        print(f"table of contents lists {len(toc)} sections; {len(toc) - len(missing)} found in body")
        print(f"  missing from body: {missing or 'none'}")
        print(f"  in body but not in contents: {extra or 'none'}")
    chunks = parser.chunk(nodes, args.max_chunk_chars, args.overlap)
    sizes = [len(c.text) for c in chunks]
    print(f"parsed {len(nodes)} top-level nodes -> {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s")
    if sizes:
        print(f"chunk chars: min={min(sizes)} max={max(sizes)} mean={sum(sizes) / len(sizes):.0f}")
    chapters = Counter(c.chapter for c in chunks)
    print(f"chapters: {len(chapters)}; sections: {len({c.section for c in chunks if c.section})}")

    for section in filter(None, (s.strip().upper() for s in args.show.split(","))):
        print(f"\n===== Section {section} =====")
        for c in (c for c in chunks if c.section == section):
            print(f"--- {c.citation_path} ({len(c.text)} chars)\n{c.text}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, **c.metadata(), "text": c.text}, ensure_ascii=False) + "\n")
        print(f"wrote {args.dump}")

    if args.dry_run:
        return 0

    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    if args.reset:
        store.reset()
    t0 = time.perf_counter()
    embeddings = EmbeddingService(settings.embedding_model).embed([c.text for c in chunks])
    store.upsert(chunks, embeddings)
    print(
        f"embedded + upserted {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s; "
        f"collection '{settings.chroma_collection}' now has {store.count()} chunks at {settings.chroma_persist_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
