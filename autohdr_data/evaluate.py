"""Score AutoHDR restoration predictions against FPHDR-style ground-truth labels.

Ground truth comes from labels/*.json (see autohdr_data.schema: ``txt`` is always the true
character, ``grade`` its damage severity). Predictions are expected as a JSONL file, one
record per page, keyed by the same image file name:

    {"file_name": "FS_2_2_1.jpg", "chars": [{"char_id": "...", "txt": "predicted char"}, ...]}

infer_pipeline.py does not emit this shape natively - write a small adapter once you've
inspected its actual output (during setup/run_smoke_test.sh) that maps char_id -> predicted
text into this JSONL format.

Reports overall + per-damage-grade character accuracy. grade=None chars (never damaged) are
reported separately as a legibility sanity check, not folded into the restoration score.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .schema import iter_dataset

_DAMAGE_GRADES = ("light", "medium", "severe")


def load_predictions(pred_path: str | Path) -> dict[str, dict[str, str]]:
    """file_name -> {char_id: predicted_txt}"""
    preds: dict[str, dict[str, str]] = {}
    with Path(pred_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds[rec["file_name"]] = {c["char_id"]: c["txt"] for c in rec["chars"]}
    return preds


def evaluate(images_dir: str | Path, labels_dir: str | Path, pred_path: str | Path) -> dict:
    preds = load_predictions(pred_path)
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    missing_pred_pages = []

    for image_path, page in iter_dataset(images_dir, labels_dir):
        page_preds = preds.get(image_path.name)
        if page_preds is None:
            missing_pred_pages.append(image_path.name)
            continue
        for ch in page.chars:
            grade_key = ch.grade or "none"
            total[grade_key] += 1
            if page_preds.get(ch.char_id) == ch.txt:
                correct[grade_key] += 1

    def _bucket(keys: tuple[str, ...]) -> dict:
        c = sum(correct[k] for k in keys)
        t = sum(total[k] for k in keys)
        return {"correct": c, "total": t, "accuracy": (c / t) if t else None}

    report = {grade: _bucket((grade,)) for grade in (*_DAMAGE_GRADES, "none") if total[grade]}
    report["overall_damaged"] = _bucket(_DAMAGE_GRADES)
    if missing_pred_pages:
        report["_pages_missing_predictions"] = missing_pred_pages
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pred", required=True, help="Path to predictions JSONL")
    args = parser.parse_args()

    report = evaluate(args.images, args.labels, args.pred)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
