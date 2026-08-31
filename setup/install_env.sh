#!/usr/bin/env bash
# Set up AutoHDR inside WSL2 Ubuntu: Python 3.10 venv, clone the official repo, install deps.
# Run from inside WSL (see setup/wsl_setup.md). Safe to re-run.
set -euo pipefail

AUTOHDR_HOME="${AUTOHDR_HOME:-$HOME/autohdr}"   # kept on the Linux fs, not /mnt/d/..., for I/O speed
REPO_URL="https://github.com/SCUT-DLVCLab/AutoHDR.git"
REPO_DIR="$AUTOHDR_HOME/AutoHDR"
VENV_DIR="$AUTOHDR_HOME/venv"

echo "== 1/5: GPU check =="
nvidia-smi || { echo "nvidia-smi failed - fix the Windows NVIDIA driver / WSL GPU passthrough first (see setup/wsl_setup.md)"; exit 1; }

echo "== 2/5: Python 3.10 venv at $VENV_DIR =="
mkdir -p "$AUTOHDR_HOME"
if [ ! -d "$VENV_DIR" ]; then
    python3.10 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "== 3/5: Clone AutoHDR into $REPO_DIR =="
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull
fi

echo "== 4/5: Install pinned deps from the repo's requirements.txt =="
pip install -r "$REPO_DIR/requirements.txt"

echo "== 5/5: Install mmcv/mmdet/mmpretrain via openmim (prebuilt CUDA-matched wheels) =="
pip install -U openmim
mim install mmengine==0.10.5
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"
mim install mmpretrain

echo
echo "Done. Repo at: $REPO_DIR"
echo "Next: put checkpoint files under $REPO_DIR/ckpt/ (see docs/checkpoints.md), then run:"
echo "  AUTOHDR_HOME=$AUTOHDR_HOME setup/run_smoke_test.sh"
