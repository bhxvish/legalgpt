"""Convert the public InRhetoricalRoles corpus (Kalamkar et al., LREC 2022) to this project's 5 roles.

Source: huggingface.co/datasets/opennyaiorg/InRhetoricalRoles (gated: accept its terms on Hugging
Face first). Licence: CC BY-SA 4.0 per the authors' repository (Apache-2.0 on the dataset card); it
is used for training only and never redistributed. Cite: Kalamkar et al., "Corpus for Automatic
Structuring of Legal Documents", LREC 2022.

Role mapping onto the LabelScheme definitions (label_scheme.py):
  FAC, RLC (ruling by the lower court)          -> Facts      (lower courts' decisions are Facts)
  STA (statute)                                 -> Law Applied
  PRE_RELIED, PRE_NOT_RELIED                    -> Precedent
  ARG_PETITIONER, ARG_RESPONDENT                -> Argument
  ANALYSIS, Ratio, RPC (ruling by present court) -> Ruling    (the present court's own reasoning)
  PREAMBLE, ISSUE, NONE                         -> no label: kept as context, never trained on
Known mismatch: ANALYSIS sentences stating a general principle without a case would be Law Applied
in our guideline; the public labels cannot separate them.

Judgments that overlap with our own corpus (by shared sentences) are dropped, so the
cross-validation test folds stay unseen.

Usage:
    python backend/scripts/build_rrl_public.py --train train.json --dev dev.json --version v0.3
Writes data/rrl_public/{train,dev}.jsonl (one sentence per line: sentence_id, case_id, idx, text, label).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402

MAP = {
    "FAC": "Facts", "RLC": "Facts",
    "STA": "Law Applied",
    "PRE_RELIED": "Precedent", "PRE_NOT_RELIED": "Precedent",
    "ARG_PETITIONER": "Argument", "ARG_RESPONDENT": "Argument",
    "ANALYSIS": "Ruling", "RATIO": "Ruling", "RPC": "Ruling",
    "PREAMBLE": None, "ISSUE": None, "NONE": None,
}


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def documents(path: Path):
    """Yield (doc_id, [(text, original_role)]) from the Label Studio export the corpus uses."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("documents") or list(data.values())
    for n, doc in enumerate(data):
        try:
            results = doc["annotations"][0]["result"]
        except (KeyError, IndexError, TypeError):
            raise SystemExit(f"unexpected record format in {path}: keys {list(doc)[:8]}")
        segs = []
        for r in sorted(results, key=lambda r: r["value"].get("start", 0)):
            v = r["value"]
            role = (v.get("labels") or ["NONE"])[0].upper()
            text = " ".join(str(v.get("text", "")).split())
            if text:
                segs.append((text, role))
        yield str(doc.get("id", n)), segs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--version", default="v0.3", help="our snapshot, for the overlap check")
    ap.add_argument("--max-overlap", type=float, default=0.2, help="drop a public judgment sharing more than this share of sentences with ours")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "rrl_public")
    args = ap.parse_args()

    _, ours = AnnotationStore(get_settings().annotation_store_dir).load_version(args.version, verify=True)
    our_sentences = {norm(s.text) for s in ours if len(norm(s.text)) > 40}
    args.out.mkdir(parents=True, exist_ok=True)
    for split, path in (("train", args.train), ("dev", args.dev)):
        roles, kept, dropped, n_sent = Counter(), 0, 0, 0
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for doc_id, segs in documents(path):
                long = [norm(t) for t, _ in segs if len(norm(t)) > 40]
                if long and sum(t in our_sentences for t in long) / len(long) > args.max_overlap:
                    dropped += 1
                    continue
                kept += 1
                for i, (text, role) in enumerate(segs):
                    unknown = role not in MAP
                    label = None if unknown else MAP[role]
                    roles[role if not unknown else f"?{role}"] += 1
                    f.write(json.dumps({"sentence_id": f"pub-{split}-{doc_id}-{i}", "case_id": f"pub-{split}-{doc_id}",
                                        "idx": i, "text": text, "label": label}) + "\n")
                    n_sent += 1
        print(f"{split}: {kept} judgments kept, {dropped} dropped as overlapping ours, {n_sent} sentences")
        print("  roles:", dict(roles.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
