import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_blind_samples import scores  # noqa: E402


def test_scores_pool_accuracy_f1_and_confidence_split():
    rows = [{"case": "a", "gold": "Facts", "pred": "Facts", "confident": True},
            {"case": "a", "gold": "Facts", "pred": "Ruling", "confident": False},
            {"case": "b", "gold": "Ruling", "pred": "Ruling", "confident": True},
            {"case": "b", "gold": "Precedent", "pred": "Precedent", "confident": True}]
    s = scores(rows)
    assert s["n"] == 4 and s["judgments"] == 2 and s["accuracy"] == 0.75
    assert s["confident"] == {"n": 3, "accuracy": 1.0} and s["flagged"] == {"n": 1, "accuracy": 0.0}
    assert s["per_role_f1"]["Facts"] == 0.667 and s["confusion"] == {"Facts -> Ruling": 1}
    assert scores([]) == {"n": 0}
