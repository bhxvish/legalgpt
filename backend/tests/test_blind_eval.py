"""blind_eval_answers.py: the sheet hides which model wrote A/B, and score unblinds correctly."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blind_eval_answers as be  # noqa: E402


def _answers(out: Path, n: int = 6) -> None:
    items = [{"question": f"q{i}", "sources": [], "answers": {"groq": {"text": f"groq answer {i}", "seconds": 1},
                                                             "adapter": {"text": f"lora answer {i}", "seconds": 1}}}
             for i in range(n)]
    out.with_suffix(".answers.json").write_text(json.dumps({"meta": {"groq_model": "g", "adapter": "v0.3"},
                                                            "items": items}), encoding="utf-8")


def test_sheet_hides_models_and_score_unblinds(tmp_path: Path) -> None:
    out = tmp_path / "blind"
    _answers(out)
    be.sheet(Namespace(out=out, seed=7))
    sheet_text = out.with_suffix(".sheet.json").read_text(encoding="utf-8")
    assert "groq\"" not in sheet_text and "adapter" not in sheet_text  # identities only in the key
    key = {k["id"]: k for k in json.loads(out.with_suffix(".key.json").read_text(encoding="utf-8"))["key"]}
    assert {key[i]["A"] for i in key} == {"groq", "adapter"}  # shuffled, not always the same order

    data = json.loads(sheet_text)
    for r in data["rows"]:  # rate every Groq answer 5s and every LoRA answer 3s
        for slot in ("A", "B"):
            v = 5 if r[slot].startswith("groq") else 3
            r["ratings"][slot] = {c: v for c in be.CRITERIA}
    out.with_suffix(".sheet.json").write_text(json.dumps(data), encoding="utf-8")
    be.score(Namespace(out=out, rater="test"))
    report = out.with_suffix(".report.md").read_text(encoding="utf-8")
    assert "| **mean of all three** | **5.00** | **3.00** |" in report
    assert "Groq better on 6, LoRA better on 0, tie on 0" in report


def test_score_refuses_unrated_sheet(tmp_path: Path) -> None:
    out = tmp_path / "blind"
    _answers(out, 2)
    be.sheet(Namespace(out=out, seed=1))
    with pytest.raises(SystemExit, match="not fully rated"):
        be.score(Namespace(out=out, rater="test"))
