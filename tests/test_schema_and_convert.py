"""Tests for autohdr_data against a tiny synthetic fixture matching the note.txt schema.

No GPU, no real dataset, no network needed - the fixture is generated on the fly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autohdr_data import convert_coco, convert_ocr, evaluate, schema, validate

IMG_W, IMG_H = 200, 100

LABEL = {
    "columns": [
        {"x": 5, "y": 5, "w": 80, "h": 90, "column_id": "col_0", "idx": 0},
    ],
    "chars": [
        {"x": 10, "y": 10, "w": 20, "h": 20, "txt": "天", "cid": 1, "char_id": "c0", "idx": 0, "grade": None},
        {"x": 40, "y": 10, "w": 20, "h": 20, "txt": "地", "cid": 2, "char_id": "c1", "idx": 1, "grade": "light"},
        {"x": 70, "y": 10, "w": 20, "h": 20, "txt": "人", "cid": 3, "char_id": "c2", "idx": 2, "grade": "severe"},
    ],
}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    Image.new("RGB", (IMG_W, IMG_H)).save(images_dir / "page_0001.jpg")
    (labels_dir / "page_0001.json").write_text(json.dumps(LABEL, ensure_ascii=False), encoding="utf-8")

    return images_dir, labels_dir


def test_load_label_roundtrip(dataset_dir):
    _, labels_dir = dataset_dir
    page = schema.load_label(labels_dir / "page_0001.json")

    assert page.image_id == "page_0001"
    assert len(page.columns) == 1
    assert len(page.chars) == 3
    assert page.chars[0].grade is None
    assert page.chars[0].is_damaged is False
    assert page.chars[1].grade == "light"
    assert page.chars[2].is_damaged is True
    assert page.chars[2].bbox_xyxy == (70, 10, 90, 30)


def test_iter_dataset_pairs_image_and_label(dataset_dir):
    images_dir, labels_dir = dataset_dir
    pairs = list(schema.iter_dataset(images_dir, labels_dir))
    assert len(pairs) == 1
    image_path, page = pairs[0]
    assert image_path.name == "page_0001.jpg"
    assert page.image_id == "page_0001"


def test_iter_dataset_missing_image_raises(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "orphan.json").write_text(json.dumps(LABEL), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        list(schema.iter_dataset(tmp_path / "images_missing", labels_dir))


def test_validate_dataset_clean(dataset_dir):
    images_dir, labels_dir = dataset_dir
    assert validate.validate_dataset(images_dir, labels_dir) == []


def test_validate_dataset_flags_out_of_bounds_bbox_and_bad_grade(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (50, 50)).save(images_dir / "bad.jpg")

    bad_label = {
        "columns": [],
        "chars": [
            {"x": 40, "y": 40, "w": 30, "h": 30, "txt": "x", "cid": 1, "char_id": "c0", "idx": 0, "grade": "not_a_grade"},
        ],
    }
    (labels_dir / "bad.json").write_text(json.dumps(bad_label), encoding="utf-8")

    errors = validate.validate_dataset(images_dir, labels_dir)
    assert any("exceeds image bounds" in e for e in errors)
    assert any("invalid grade" in e for e in errors)


def test_validate_dataset_flags_unpaired_files(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (50, 50)).save(images_dir / "orphan_image.jpg")
    (labels_dir / "orphan_label.json").write_text(json.dumps({"columns": [], "chars": []}), encoding="utf-8")

    errors = validate.validate_dataset(images_dir, labels_dir)
    assert any("orphan_image" in e for e in errors)
    assert any("orphan_label" in e for e in errors)


def test_build_coco_only_includes_damaged_chars(dataset_dir):
    images_dir, labels_dir = dataset_dir
    coco = convert_coco.build_coco(images_dir, labels_dir)

    assert len(coco["images"]) == 1
    assert coco["images"][0] == {"id": 0, "file_name": "page_0001.jpg", "width": IMG_W, "height": IMG_H}
    assert len(coco["annotations"]) == 2  # only the light + severe chars
    assert {a["grade"] for a in coco["annotations"]} == {"light", "severe"}
    assert all(a["category_id"] == 0 for a in coco["annotations"])


def test_build_ocr_gt_includes_all_chars_and_flags_oov(dataset_dir, tmp_path):
    images_dir, labels_dir = dataset_dir
    dic_path = tmp_path / "dic.txt"
    dic_path.write_text("天\n地\n", encoding="utf-8")  # "人" deliberately missing

    records, oov = convert_ocr.build_ocr_gt(images_dir, labels_dir, dic_path)

    assert len(records) == 1
    assert len(records[0]["chars"]) == 3
    assert oov == {"人"}


def test_evaluate_scores_by_grade(dataset_dir, tmp_path):
    images_dir, labels_dir = dataset_dir
    pred_path = tmp_path / "preds.jsonl"
    predictions = {
        "file_name": "page_0001.jpg",
        "chars": [
            {"char_id": "c0", "txt": "天"},   # correct, undamaged
            {"char_id": "c1", "txt": "地"},   # correct, light
            {"char_id": "c2", "txt": "王"},   # wrong, severe
        ],
    }
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    report = evaluate.evaluate(images_dir, labels_dir, pred_path)

    assert report["light"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert report["severe"] == {"correct": 0, "total": 1, "accuracy": 0.0}
    assert report["none"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert report["overall_damaged"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert "_pages_missing_predictions" not in report


def test_evaluate_reports_missing_prediction_pages(dataset_dir, tmp_path):
    images_dir, labels_dir = dataset_dir
    pred_path = tmp_path / "empty_preds.jsonl"
    pred_path.write_text("", encoding="utf-8")

    report = evaluate.evaluate(images_dir, labels_dir, pred_path)
    assert report["_pages_missing_predictions"] == ["page_0001.jpg"]


# --- Real-dataset quirks (found by inspecting the actual FPHDR_FPHDR_syn.zip delivery) ---
# The synthetic subset (FPHDR_syn/) has no "columns" key, no per-char "char_id", and grade
# is the literal string "None"/"damaged" (not JSON null / a severity scale). The real subset
# (FPHDR/) mostly matches note.txt's schema but occasionally a column is missing "column_id"
# or a char is missing "txt". load_label() must tolerate all of this instead of crashing.

SYNTHETIC_STYLE_LABEL = {
    "name": "M5_image_0",
    "img_name": "M5_image_0",
    "width": IMG_W,
    "height": IMG_H,
    "chars": [
        {"x": 10, "y": 10, "w": 20, "h": 20, "txt": "可", "cid": 1, "idx": 0, "grade": "None"},
        {"x": 40, "y": 10, "w": 20, "h": 20, "txt": "久", "cid": 2, "idx": 1, "grade": "damaged"},
    ],
}


def test_load_label_tolerates_synthetic_subset_shape(tmp_path):
    label_path = tmp_path / "M5_image_0.json"
    label_path.write_text(json.dumps(SYNTHETIC_STYLE_LABEL, ensure_ascii=False), encoding="utf-8")

    page = schema.load_label(label_path)

    assert page.columns == []  # no "columns" key at all in this subset
    assert len(page.chars) == 2
    assert page.chars[0].grade is None  # string "None" normalized to None
    assert page.chars[0].is_damaged is False
    assert page.chars[0].char_id == "M5_image_0_0"  # synthesized, no "char_id" in source
    assert page.chars[1].grade == "damaged"
    assert page.chars[1].is_damaged is True


def test_load_label_tolerates_missing_column_id_and_txt(tmp_path):
    label = {
        "columns": [{"x": 0, "y": 0, "w": 50, "h": 50, "idx": 0}],  # no column_id
        "chars": [{"x": 5, "y": 5, "w": 10, "h": 10, "cid": 1, "char_id": "c0", "idx": 0, "grade": None}],  # no txt
    }
    label_path = tmp_path / "page_0002.json"
    label_path.write_text(json.dumps(label), encoding="utf-8")

    page = schema.load_label(label_path)

    assert page.columns[0].column_id == "page_0002_col_0"
    assert page.chars[0].txt == ""


def test_validate_dataset_flags_synthesized_empty_txt(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (50, 50)).save(images_dir / "page_0002.jpg")
    label = {"columns": [], "chars": [{"x": 5, "y": 5, "w": 10, "h": 10, "cid": 1, "char_id": "c0", "idx": 0, "grade": None}]}
    (labels_dir / "page_0002.json").write_text(json.dumps(label), encoding="utf-8")

    errors = validate.validate_dataset(images_dir, labels_dir)
    assert any("empty txt" in e for e in errors)
