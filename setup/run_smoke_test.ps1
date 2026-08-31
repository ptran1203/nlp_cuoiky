# Smoke test: run the official infer_pipeline.py, unmodified, on the repo's own bundled
# example image. Logs peak GPU memory used so we can confirm it fits the 12GB budget. This is
# the "installed correctly per source + data" deliverable (80% bar in note.txt) - no dataset
# required yet.
# NOT run automatically - review and run this yourself when ready:
#   powershell -ExecutionPolicy Bypass -File setup\run_smoke_test.ps1
$ErrorActionPreference = "Stop"

# Defaults to this project's own root, since AutoHDR is already cloned at <project>\AutoHDR.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AutohdrHome = if ($env:AUTOHDR_HOME) { $env:AUTOHDR_HOME } else { $ProjectRoot }
$RepoDir = Join-Path $AutohdrHome "AutoHDR"
$VenvDir = Join-Path $AutohdrHome "venv"
$VramLog = Join-Path $AutohdrHome "smoke_test_vram.csv"

& "$VenvDir\Scripts\Activate.ps1"
Set-Location $RepoDir

# Confirmed by reading infer_pipeline.py directly (see docs/checkpoints.md):
#   - it always processes the single hardcoded ./example.jpg (bundled in the repo) - no
#     --input flag exists for this
#   - output goes to ./results/img/example.jpg and ./results/combined/example.jpg
#   - --model_name_or_path IS a real flag, and must be set explicitly - the script's default
#     points at the 7B checkpoint, which won't fit this machine's 12GB GPU

# Poll GPU memory in the background while the pipeline runs.
$vramJob = Start-Job -ScriptBlock {
    param($log)
    nvidia-smi --query-gpu=memory.used --format=csv -l 2 | Out-File -FilePath $log -Encoding utf8
} -ArgumentList $VramLog

$env:CUDA_VISIBLE_DEVICES = "0"
try {
    python infer_pipeline.py --model_name_or_path ./ckpt/AutoHDR-Qwen2-1.5B
} finally {
    Stop-Job $vramJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $vramJob -ErrorAction SilentlyContinue | Out-Null
}

Write-Host ""
Write-Host "Output written to: $RepoDir\results\img\example.jpg (and results\combined\example.jpg)"
Write-Host "Peak GPU memory used during the run:"
Get-Content $VramLog | Select-Object -Skip 1 | Sort-Object { [int]($_ -replace '[^\d]', '') } | Select-Object -Last 1
