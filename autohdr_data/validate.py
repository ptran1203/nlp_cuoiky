"""Sanity-check a dataset directory against the FPHDR-style images/+labels/ schema.

Checks:
  - every labels/*.json has a matching image, and vice versa
  - every char/column bbox lies within the image bounds
  - every char has a non-empty ``txt`` and a valid ``grade``
  - char/column ``idx`` values are unique within a page

Usage:
    python -m autohdr_data.validate --images path/to/images --labels path/to/labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .schema import GRADES, load_label


def validate_dataset(images_dir: str | Path, labels_dir: str | Path) -> list[str]:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    errors: list[str] = []

    image_stems = {p.stem: p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    label_paths = sorted(labels_dir.glob("*.json"))

    if not label_paths:
        errors.append(f"No .json label files found in {labels_dir}")
        return errors

    label_stems = {p.stem for p in label_paths}
    for stem in sorted(label_stems - image_stems.keys()):
        errors.append(f"labels/{stem}.json has no matching image in {images_dir}")
    for stem in sorted(image_stems.keys() - label_stems):
        errors.append(f"images/{stem}.* has no matching labels/{stem}.json")

    for label_path in label_paths:
        stem = label_path.stem
        if stem not in image_stems:
            continue  # already reported above

        try:
            page = load_label(label_path)
        except Exception as exc:  # noqa: BLE001 - surface any parse error as a validation error
            errors.append(f"{label_path.name}: failed to parse ({exc})")
            continue

        with Image.open(image_stems[stem]) as im:
            img_w, img_h = im.size

        seen_char_idx: set[int] = set()
        for ch in page.chars:
            if ch.grade not in GRADES:
                errors.append(f"{label_path.name}: char idx={ch.idx} has invalid grade {ch.grade!r}")
            if ch.idx in seen_char_idx:
                errors.append(f"{label_path.name}: duplicate char idx={ch.idx}")
            seen_char_idx.add(ch.idx)
            if not ch.txt:
                errors.append(f"{label_path.name}: char idx={ch.idx} has empty txt")
            x0, y0, x1, y1 = ch.bbox_xyxy
            if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
                errors.append(
                    f"{label_path.name}: char idx={ch.idx} bbox {ch.bbox_xyxy} "
                    f"exceeds image bounds ({img_w}x{img_h})"
                )

        seen_col_idx: set[int] = set()
        for col in page.columns:
            if col.idx in seen_col_idx:
                errors.append(f"{label_path.name}: duplicate column idx={col.idx}")
            seen_col_idx.add(col.idx)
            x0, y0, x1, y1 = col.bbox_xyxy
            if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
                errors.append(
                    f"{label_path.name}: column idx={col.idx} bbox {col.bbox_xyxy} "
                    f"exceeds image bounds ({img_w}x{img_h})"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="Path to images/ directory")
    parser.add_argument("--labels", required=True, help="Path to labels/ directory")
    args = parser.parse_args()

    errors = validate_dataset(args.images, args.labels)
    if errors:
        print(f"Found {len(errors)} issue(s):")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Dataset OK: all images/labels paired, all bboxes in-bounds, all grades valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
