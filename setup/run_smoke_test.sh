#!/usr/bin/env bash
# Smoke test: run the official infer_pipeline.py, unmodified, on the repo's own bundled
# example image, using the AutoHDR-Qwen2-1.5B checkpoint. Logs peak GPU memory used so we
# can confirm it fits the 12GB budget. This is the "installed correctly per source + data"
# deliverable (80% bar in note.txt) - no dataset required yet.
set -euo pipefail

AUTOHDR_HOME="${AUTOHDR_HOME:-$HOME/autohdr}"
REPO_DIR="$AUTOHDR_HOME/AutoHDR"
VENV_DIR="$AUTOHDR_HOME/venv"
OUT_DIR="$AUTOHDR_HOME/smoke_test_out"
VRAM_LOG="$AUTOHDR_HOME/smoke_test_vram.csv"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

# The official README documents no CLI flags for infer_pipeline.py - it's invoked bare
# (`CUDA_VISIBLE_DEVICES=<gpu_id> python infer_pipeline.py`), so its input image, output
# path, and which LLM checkpoint (1.5B vs 7B) to load are almost certainly constants near
# the top of the script or in a small config it imports. Before the first run, open
# infer_pipeline.py and point those constants at:
#   - input: the repo's own example.jpg (bundled at the repo root)
#   - output: $OUT_DIR
#   - LLM checkpoint: the AutoHDR-Qwen2-1.5B path under ckpt/ (see docs/checkpoints.md)
# then re-run this script.

# Poll GPU memory in the background while the pipeline runs.
nvidia-smi --query-gpu=memory.used --format=csv -l 2 > "$VRAM_LOG" &
NVSMI_PID=$!
trap 'kill "$NVSMI_PID" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0 python infer_pipeline.py

kill "$NVSMI_PID" 2>/dev/null || true
trap - EXIT

echo
echo "Output written to: $OUT_DIR"
echo "Peak GPU memory used during the run:"
tail -n +2 "$VRAM_LOG" | sort -t' ' -k1 -n | tail -1
