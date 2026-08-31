"""Lightweight, pure-Python tooling for the FPHDR-style dataset used by AutoHDR.

Submodules:
    schema        - dataclasses + loaders for the images/ + labels/ annotation format
    validate      - dataset sanity checks (CLI: `python -m autohdr_data.validate`)
    convert_coco  - labels -> COCO json for the AutoHDR damage detector (mmdet)
    convert_ocr   - labels -> OCR ground truth (char bbox + text)
    evaluate      - score restoration predictions vs. ground-truth characters
    download      - fetch + unpack the dataset archive from Google Drive

None of this package touches torch/mmdet/etc. It only needs Pillow, so it runs directly
on Windows without the WSL/GPU environment set up in setup/.
"""
