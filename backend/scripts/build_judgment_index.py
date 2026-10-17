"""Build the judgment library index (app/judgments/library.py) used for case citations.

Judgments:
  - data/judgment_library/sc/*.txt (+ .json metadata), fetched with
    fetch_sc_judgments.py --library (Supreme Court, CC-BY-4.0);
  - the annotated collection (data/raw_judgments), using its stored sentences and labels.
Roles: the latest label for annotated sentences (human where a person labelled or checked it);
otherwise the newest sentence-role checkpoint in data/models/rrl (the ensemble) predicts them.
Writes the Chroma collection "judgments" (replaced on every run) and
data/judgment_library/passages.jsonl for inspection.

Usage (from the repo root; the GPU environment is faster for the role labelling):
    .venv-gpu\\Scripts\\python backend/scripts/build_judgment_index.py
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.judgments.library import COLLECTION, JudgmentDoc, load_library_texts, passages, tidy_title  # noqa: E402
from app.labeling_assistant.dataset import RRLExample  # noqa: E402
from app.labeling_assistant.segmenter import AssistedSegmenter  # noqa: E402

LIBRARY = REPO_ROOT / "data" / "judgment_library"


def corpus_docs(s) -> list[JudgmentDoc]:
    store = AnnotationStore(s.annotation_store_dir)
    latest: dict[tuple[str, int], str] = {}  # (case, sentence index) -> latest label
    for line in store.labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["case_id"], int(r["idx"]))] = r["label"]
    docs = []
    for e in JudgmentCollector(s.raw_judgments_dir).index():
        sentences = store.load_segments(e["case_id"])
        if not sentences:
            continue
        roles = [latest.get((e["case_id"], i)) for i in range(len(sentences))]
        if sum(r is not None for r in roles) < len(roles) // 2:
            continue  # mostly unlabelled: leave it to the classifier path below
        title = tidy_title(e.get("title") or e["case_id"])
        docs.append(JudgmentDoc(e["case_id"], title, e.get("court", ""), e.get("decision_year"), sentences, roles, "corpus"))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path, default=LIBRARY / "sc")
    ap.add_argument("--no-corpus", action="store_true", help="index only the fetched library")
    args = ap.parse_args()
    s = get_settings()
    t0 = time.time()

    docs = [] if args.no_corpus else corpus_docs(s)
    print(f"annotated collection: {len(docs)} judgments with stored labels")
    fetched = load_library_texts(args.library)
    if fetched:
        from app.labeling_assistant.intake import latest_checkpoint
        from app.labeling_assistant.sequence import EnsembleClassifier, is_ensemble
        from app.labeling_assistant.classifier import RRLClassifier

        ckpt = latest_checkpoint(s.rrl_model_dir)
        clf = EnsembleClassifier.load(ckpt) if is_ensemble(ckpt) else RRLClassifier.load(ckpt)
        segmenter = AssistedSegmenter()
        print(f"labelling {len(fetched)} fetched judgments with {ckpt.name}")
        for n, (case_id, title, court, year, text) in enumerate(fetched, 1):
            sentences = segmenter.segment(text, case_id)
            if len(sentences) < 5:
                continue
            preds = clf.predict([RRLExample(f"{case_id}::{i}", case_id, i, t) for i, t in enumerate(sentences)])
            docs.append(JudgmentDoc(case_id, title, court, year, sentences, [p.label for p in preds], "library"))
            if n % 25 == 0:
                print(f"  {n}/{len(fetched)} ({time.time() - t0:.0f}s)", flush=True)

    chunks = [c for d in docs for c in passages(d)]
    roles = Counter(c.role for c in chunks)
    print(f"{len(docs)} judgments -> {len(chunks)} passages {dict(roles)}")
    embedder = EmbeddingService(s.embedding_model)
    vecs = embedder.embed([c.text for c in chunks])
    store = VectorStore(s.chroma_persist_dir, COLLECTION)
    store.reset()
    store = VectorStore(s.chroma_persist_dir, COLLECTION)
    store.upsert(chunks, vecs)
    LIBRARY.mkdir(parents=True, exist_ok=True)
    with open(LIBRARY / "passages.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, "case": c.title, "role": c.role, "citation": c.citation_path,
                                "text": c.text}, ensure_ascii=False) + "\n")
    print(f"indexed {store.count()} passages in collection '{COLLECTION}' ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
