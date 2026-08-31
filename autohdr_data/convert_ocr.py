"""Convert FPHDR-style labels/*.json into OCR ground truth (char bbox + text), and
optionally cross-check the character vocabulary against AutoHDR's own dictionary
(``ckpt/dic_31524.txt`` in the cloned repo - one character per line).

Usage:
    python -m autohdr_data.convert_ocr --images path/to/images --labels path/to/labels \
        --out path/to/ocr_gt.jsonl [--dic path/to/dic_31524.txt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import PageLabel, iter_dataset


def load_dic(dic_path: str | Path) -> set[str]:
    text = Path(dic_path).read_text(encoding="utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def _page_record(image_name: str, page: PageLabel) -> dict:
    return {
        "file_name": image_name,
        "chars": [
            {"char_id": ch.char_id, "bbox": [ch.x, ch.y, ch.w, ch.h], "txt": ch.txt, "grade": ch.grade}
            for ch in page.chars
        ],
    }


def build_ocr_gt(
    images_dir: str | Path, labels_dir: str | Path, dic_path: str | Path | None = None
) -> tuple[list[dict], set[str]]:
    vocab = load_dic(dic_path) if dic_path else None
    oov: set[str] = set()
    records = []
    for image_path, page in iter_dataset(images_dir, labels_dir):
        records.append(_page_record(image_path.name, page))
        if vocab is not None:
            oov.update(ch.txt for ch in page.chars if ch.txt not in vocab)
    return records, oov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dic", default=None, help="Path to AutoHDR's ckpt/dic_31524.txt (optional)")
    args = parser.parse_args()

    records, oov = build_ocr_gt(args.images, args.labels, args.dic)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_chars = sum(len(r["chars"]) for r in records)
    print(f"Wrote {len(records)} pages / {total_chars} chars to {out_path}")
    if args.dic:
        print(f"Out-of-vocabulary characters vs {args.dic}: {len(oov)}")
        if oov:
            preview = " ".join(sorted(oov)[:50])
            print(f"  {preview}" + (" ..." if len(oov) > 50 else ""))


if __name__ == "__main__":
    main()
