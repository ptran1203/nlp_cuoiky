"""Download and unpack the dataset archive shared via Google Drive (see note.txt).

Uses ``gdown`` so it works from plain Windows Python - no WSL/GPU environment required.
Handles Google Drive's "file too large to scan for viruses" confirmation page automatically.

Usage:
    python -m autohdr_data.download --out data\\raw
    python -m autohdr_data.download --file-id 1hAXcWcbUUgQ2EySdWdr0--VIhp-iQ_1X --out data\\raw
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

# From the link in note.txt:
#   https://drive.google.com/file/d/1hAXcWcbUUgQ2EySdWdr0--VIhp-iQ_1X/view?usp=sharing
DEFAULT_FILE_ID = "1hAXcWcbUUgQ2EySdWdr0--VIhp-iQ_1X"


def download(file_id: str, out_dir: str | Path) -> Path:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "gdown is required: pip install -r requirements-local.txt"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{file_id}.download"

    print(f"Downloading Drive file {file_id} -> {archive_path}")
    gdown.download(id=file_id, output=str(archive_path), quiet=False)

    if zipfile.is_zipfile(archive_path):
        print(f"Extracting zip archive into {out_dir}")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)
        archive_path.unlink()
    else:
        # Not a zip (e.g. already a single file) - leave it under a stable name for
        # the caller to inspect; report its actual size so odd downloads are obvious.
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"Downloaded file is not a zip ({size_mb:.1f} MB) - left as-is at {archive_path}")

    return out_dir


def _find_dataset_root(out_dir: Path) -> Path | None:
    """After extraction, locate the directory that directly contains images/ and labels/."""
    if (out_dir / "images").is_dir() and (out_dir / "labels").is_dir():
        return out_dir
    for child in out_dir.iterdir():
        if child.is_dir() and (child / "images").is_dir() and (child / "labels").is_dir():
            return child
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID, help="Google Drive file id")
    parser.add_argument("--out", default="data/raw", help="Directory to download/extract into")
    args = parser.parse_args()

    out_dir = download(args.file_id, args.out)
    root = _find_dataset_root(out_dir)
    if root is None:
        print(
            f"Could not find images/ + labels/ under {out_dir} after extraction - "
            "check the archive layout and pass the right subfolder to the other autohdr_data tools."
        )
        return

    if root != out_dir:
        print(f"Dataset found at {root} (nested inside {out_dir})")
    print(f"images/: {root / 'images'}")
    print(f"labels/: {root / 'labels'}")
    print("\nNext: python -m autohdr_data.validate --images "
          f"{root / 'images'} --labels {root / 'labels'}")


if __name__ == "__main__":
    main()
