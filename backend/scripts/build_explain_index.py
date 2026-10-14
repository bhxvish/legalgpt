"""Build the "similar labelled examples" index for the Review tab's explanations.

Embeds every human-checked sentence of the newest frozen corpus with the newest classifier's own
pooled representation (sentence + context, as the classifier reads it) and saves
<checkpoint>/neighbours.npz (+ .json with ids, labels and texts). Re-run after retraining.

Usage (from the repo root; the GPU environment is much faster):
    .venv-gpu\\Scripts\\python backend/scripts/build_explain_index.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import RRLClassifier  # noqa: E402
from app.labeling_assistant.dataset import RRLDataset, context_texts, is_human_checked  # noqa: E402
from app.labeling_assistant.explain import ClassifierExplainer  # noqa: E402
from app.labeling_assistant.intake import latest_checkpoint  # noqa: E402


def main() -> int:
    s = get_settings()
    store = AnnotationStore(s.annotation_store_dir)
    version = sorted(p.name for p in (s.annotation_store_dir / "versions").iterdir() if p.is_dir())[-1]
    _, sentences = store.load_version(version, verify=True)
    checked = [x for x in sentences if is_human_checked(x)]
    examples = RRLDataset.examples_from(checked)
    ckpt = latest_checkpoint(s.rrl_model_dir)
    clf = RRLClassifier.load(ckpt)
    ctx = context_texts(examples) if clf.context else {}
    vecs = ClassifierExplainer(clf).pooled([e.text for e in examples], [ctx.get(e.sentence_id, "") for e in examples])
    out = ckpt / "neighbours.npz"
    np.savez_compressed(out, vectors=vecs.astype(np.float16))
    meta = [{"sentence_id": e.sentence_id, "case_id": e.case_id, "label": e.label, "text": e.text} for e in examples]
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"{len(examples)} human-checked sentences from corpus {version}, embedded with {ckpt.name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
