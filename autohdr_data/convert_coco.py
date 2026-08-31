"""Convert FPHDR-style labels/*.json into a COCO-format detection dataset for the
AutoHDR damage detector.

The repo's mmdet DINO config (`ckpt/damage_detect.py`) is a single-class detector rooted
at ``HDR_Dataset/`` - it just finds "a damaged region", not the damage grade. So every char
with grade in {light, medium, severe} becomes one box of category "damaged"; undamaged
chars (grade is null) are not included.

Usage:
    python -m autohdr_data.convert_coco --images path/to/images --labels path/to/labels \
        --out path/to/HDR_Dataset/annotations.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .schema import iter_dataset

CATEGORY_ID = 0
CATEGORY_NAME = "damaged"


def build_coco(images_dir: str | Path, labels_dir: str | Path) -> dict:
    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }
    ann_id = 0
    for img_id, (image_path, page) in enumerate(iter_dataset(images_dir, labels_dir)):
        with Image.open(image_path) as im:
            width, height = im.size
        coco["images"].append(
            {"id": img_id, "file_name": image_path.name, "width": width, "height": height}
        )
        for ch in page.chars:
            if not ch.is_damaged:
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY_ID,
                    "bbox": [ch.x, ch.y, ch.w, ch.h],
                    "area": ch.w * ch.h,
                    "iscrowd": 0,
                    # extra fields below are ignored by mmdet but kept for traceability
                    "grade": ch.grade,
                    "char_id": ch.char_id,
                }
            )
            ann_id += 1
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    coco = build_coco(args.images, args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(coco['images'])} images / {len(coco['annotations'])} damaged-char boxes to {out_path}")


if __name__ == "__main__":
    main()
