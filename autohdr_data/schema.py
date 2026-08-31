"""Data schema for the FPHDR-style dataset (images/ + labels/) used by AutoHDR.

Label JSON format, as documented by the paper authors::

    {
      "columns": [
        {"x": .., "y": .., "w": .., "h": .., "column_id": "...", "idx": ..},
        ...
      ],
      "chars": [
        {
          "x": .., "y": .., "w": .., "h": .., "txt": "...", "cid": ..,
          "char_id": "...", "idx": .., "grade": "light|medium|severe|null"
        },
        ...
      ]
    }

``columns`` are column-level bounding boxes; ``chars`` are character-level boxes, where
``txt`` is always the ground-truth character (even when damaged) and ``grade`` records the
damage severity, or ``null``/absent for an undamaged character.

The actual delivered dataset (`FPHDR_FPHDR_syn.zip`, linked in note.txt) has two subsets
that diverge from that spec, confirmed by inspecting real label files:

- ``FPHDR/FPHDR/`` (real scans) matches the documented schema exactly: has ``columns``, each
  char has a ``char_id``, and ``grade`` is JSON ``null``/``"light"``/``"medium"``/``"severe"``.
- ``FPHDR_syn/FPHDR_syn/`` (synthetic scans) is simplified: **no** ``columns`` key at all, chars
  have **no** ``char_id`` field, and ``grade`` is only ever the *string* ``"None"`` (not JSON
  null) or ``"damaged"`` - a binary damaged/undamaged flag instead of a severity scale.

Everything below normalizes both variants into one shape: missing ``columns`` -> ``[]``,
missing ``char_id`` -> synthesized as ``"<page>_<idx>"``, and the string ``"None"`` -> ``None``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# "damaged" is the synthetic subset's only non-null grade (binary damaged/undamaged, no
# severity scale); "light"/"medium"/"severe" are the real subset's severity scale.
GRADES: tuple[Optional[str], ...] = ("light", "medium", "severe", "damaged", None)
_DAMAGE_GRADES = {"light", "medium", "severe", "damaged"}
_NULL_GRADE_STRINGS = {"", "null", "none"}
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Column:
    x: float
    y: float
    w: float
    h: float
    column_id: str
    idx: int

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass(frozen=True)
class Char:
    x: float
    y: float
    w: float
    h: float
    txt: str
    cid: int
    char_id: str
    idx: int
    grade: Optional[str]  # "light" | "medium" | "severe" | None

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def is_damaged(self) -> bool:
        return self.grade in _DAMAGE_GRADES


@dataclass(frozen=True)
class PageLabel:
    image_id: str
    columns: list[Column]
    chars: list[Char]


def _normalize_grade(raw: object) -> Optional[str]:
    """Map the JSON value to None/"light"/"medium"/"severe"/"damaged" - or pass an
    unrecognized value straight through so callers (in particular autohdr_data.validate) can
    flag it as bad data instead of the whole page failing to load.

    Handles both JSON null and the synthetic subset's literal string "None"/"none" as "no
    damage", case-insensitively."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in _NULL_GRADE_STRINGS:
        return None
    return raw  # type: ignore[return-value]


def load_label(json_path: str | Path) -> PageLabel:
    """Parse a single labels/*.json file into a PageLabel.

    Tolerates gaps observed in the actual delivered dataset (both subsets, not just the
    synthetic one): ``columns`` may be absent entirely (-> []), individual columns are
    sometimes missing ``column_id`` (-> synthesized as "<page>_col_<idx>"), individual chars
    are sometimes missing ``char_id`` (-> synthesized as "<page>_<idx>") and, rarely, ``txt``
    (-> ""; ``autohdr_data.validate`` flags empty-txt chars rather than this crashing on load).
    """
    json_path = Path(json_path)
    page_id = json_path.stem
    data = json.loads(json_path.read_text(encoding="utf-8"))

    columns = [
        Column(
            x=c["x"],
            y=c["y"],
            w=c["w"],
            h=c["h"],
            column_id=c.get("column_id") or f"{page_id}_col_{c['idx']}",
            idx=c["idx"],
        )
        for c in data.get("columns", [])
    ]
    chars = [
        Char(
            x=c["x"],
            y=c["y"],
            w=c["w"],
            h=c["h"],
            txt=c.get("txt") or "",
            cid=c["cid"],
            char_id=c.get("char_id") or f"{page_id}_{c['idx']}",
            idx=c["idx"],
            grade=_normalize_grade(c.get("grade")),
        )
        for c in data.get("chars", [])
    ]
    return PageLabel(image_id=page_id, columns=columns, chars=chars)


def find_image(images_dir: str | Path, stem: str) -> Optional[Path]:
    """Locate images/<stem>.{jpg,jpeg,png}, or None if no match exists."""
    images_dir = Path(images_dir)
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_dataset(images_dir: str | Path, labels_dir: str | Path) -> Iterator[tuple[Path, PageLabel]]:
    """Yield (image_path, PageLabel) for every labels/*.json that has a matching image.

    Raises FileNotFoundError on the first label with no matching image - run
    ``autohdr_data.validate`` first if you want a full report instead of a first-error stop.
    """
    labels_dir = Path(labels_dir)
    for label_path in sorted(labels_dir.glob("*.json")):
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No image found for label {label_path.name} in {images_dir}")
        yield image_path, load_label(label_path)
