# Set up AutoHDR natively on Windows: venv, clone the official repo, install deps.
# See setup/windows_setup.md for why this is native Windows instead of WSL2.
# NOT run automatically - review and run this yourself when ready:
#   powershell -ExecutionPolicy Bypass -File setup\install_env.ps1
$ErrorActionPreference = "Stop"

function Assert-LastExitCode($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$step failed (exit code $LASTEXITCODE) - stopping."
        exit 1
    }
}

# Defaults to this project's own root, since AutoHDR is already cloned at <project>\AutoHDR.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AutohdrHome = if ($env:AUTOHDR_HOME) { $env:AUTOHDR_HOME } else { $ProjectRoot }
$RepoUrl = "https://github.com/SCUT-DLVCLab/AutoHDR.git"
$RepoDir = Join-Path $AutohdrHome "AutoHDR"
$VenvDir = Join-Path $AutohdrHome "venv"

Write-Host "== 1/6: GPU check =="
nvidia-smi
Assert-LastExitCode "nvidia-smi"

Write-Host "== 2/6: Python 3.10 venv at $VenvDir =="
New-Item -ItemType Directory -Force -Path $AutohdrHome | Out-Null
if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
    Assert-LastExitCode "venv creation"
}
& "$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip
Assert-LastExitCode "pip upgrade"

Write-Host "== 3/6: Clone AutoHDR into $RepoDir =="
if (-not (Test-Path $RepoDir)) {
    git clone $RepoUrl $RepoDir
} else {
    git -C $RepoDir pull
}
Assert-LastExitCode "git clone/pull"

Write-Host "== 4/6: Install torch/torchvision/torchaudio (CUDA build) + the rest of requirements.txt =="
# Deviates from the repo's pinned torch==2.3.0 - deliberate, not an oversight. Chain of
# constraints that forces this on Windows:
#   - mmdet==3.3.0 hard-requires mmcv < 2.2.0 (checked mmdet's own source: mmcv_maximum_version
#     = '2.2.0', strict less-than).
#   - OpenMMLab only publishes Windows wheels for mmcv against torch2.3.0 starting at mmcv
#     2.2.0 - nothing in mmdet's compatible [2.0, 2.2) range has a torch2.3.0 Windows wheel, so
#     `mim install mmcv==2.1.0` falls back to a from-source build every time, which fails on a
#     pkg_resources/setuptools mismatch (would need MSVC Build Tools + a CUDA dev toolchain to
#     fix properly - much bigger install for no real benefit here).
#   - mmcv==2.1.0 *does* have Windows wheels, but only up through torch2.1.0.
# So: pin the torch2.1.0/torchvision0.16.0/torchaudio2.1.0 triplet (the matched release set)
# instead of 2.3.0/0.18.0/2.3.0. transformers/diffusers/accelerate don't hard-require 2.3.0.
# Plain `pip install torch==X` from the default PyPI index installs the CPU-ONLY Windows build
# (confirmed on 2.3.0: torch.cuda.is_available() was False) - the CUDA build needs PyTorch's
# own index explicitly. cu121 chosen since the driver here reports CUDA 12.2 (backward compat).
# --force-reinstall --no-deps: pip treats a bare "torch==X" requirement as already satisfied by
# an existing +cpu build of the same public version (local version segments like +cpu/+cu121
# are ignored when matching a specifier that doesn't itself pin one) and silently skips
# reinstalling - confirmed this happens - so force it explicitly; --no-deps avoids redundantly
# reinstalling every dependency that's already satisfied.
pip install --force-reinstall --no-deps torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
Assert-LastExitCode "torch CUDA install"
$cudaOk = python -c "import torch; print(torch.cuda.is_available())"
if ($cudaOk -ne "True") {
    Write-Error "torch installed but torch.cuda.is_available() is False - stopping before wasting time on the rest of the install."
    exit 1
}

# The rest of requirements.txt, minus what's handled separately:
#   - torch/torchvision/torchaudio: already installed above from the CUDA index; letting the
#     default-index requirements.txt line reinstall them would silently swap back to +cpu.
#   - triton==2.3.0 has no Windows wheel (Linux-only at that pinned version), and pip aborts
#     the WHOLE requirements.txt install if any one line can't resolve - installed best-effort,
#     separately, below.
#   - mmcv/mmengine/mmdet/mmpretrain: plain `pip install mmcv==2.1.0` tries to build mmcv from
#     source (no plain PyPI wheel for it) and fails on a pkg_resources/setuptools mismatch.
#     They need OpenMMLab's own prebuilt-wheel index, which only `mim install` (step 5) uses.
$reqSrc = Join-Path $RepoDir "requirements.txt"
$reqWin = Join-Path $RepoDir "requirements-windows.generated.txt"
Get-Content $reqSrc | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|triton|mmcv|mmengine|mmdet|mmpretrain)\s*(==?.*)?$' } | Set-Content $reqWin -Encoding utf8
pip install -r $reqWin
Assert-LastExitCode "pip install -r requirements.txt (torch/triton/mmcv family excluded)"

Write-Host "-- best-effort: triton (optional; only some torch.compile/fused-kernel paths need it, not plain inference) --"
pip install triton==2.3.0
if ($LASTEXITCODE -ne 0) {
    Write-Warning "triton==2.3.0 has no Windows wheel - skipped. Only a problem if infer_pipeline.py actually hits a code path that imports it; deal with that if/when it happens."
}

Write-Host "== 5/6: Install mmcv/mmdet/mmpretrain via openmim (prebuilt Windows wheels) =="
pip install -U openmim
Assert-LastExitCode "openmim install"
mim install mmengine==0.10.5
Assert-LastExitCode "mim install mmengine"
mim install "mmcv==2.1.0"
Assert-LastExitCode "mim install mmcv"
mim install "mmdet==3.3.0"
Assert-LastExitCode "mim install mmdet"
mim install mmpretrain
Assert-LastExitCode "mim install mmpretrain"

Write-Host "== 6/6: Install opencc/zhconv - imported by infer_pipeline.py but missing from requirements.txt =="
pip install opencc-python-reimplemented zhconv
Assert-LastExitCode "opencc/zhconv install"

Write-Host "-- huggingface_hub pin: requirements.txt doesn't pin it, so pip grabbed the latest, which"
Write-Host "   removed cached_download() that diffusers==0.22.0 imports at module load time --"
# transformers==4.45.0 needs huggingface_hub>=0.23.2; diffusers==0.22.0 needs cached_download
# (removed in huggingface_hub 0.26.0). 0.25.2 is the newest version satisfying both.
pip install "huggingface_hub==0.25.2"
Assert-LastExitCode "huggingface_hub pin"
$importsOk = python -c "from diffusers import UNet2DModel; from mmdet.apis import init_detector; from opencc import OpenCC; from zhconv import convert; print('OK')"
if ($importsOk -ne "OK") {
    Write-Error "Post-install import check failed - see output above."
    exit 1
}

Write-Host "-- fairscale: needed by the repo's OWN vendored AutoHDR/mmdet/ (not the pip-installed"
Write-Host "   mmdet package - Python resolves the local AutoHDR/mmdet/ dir first when running"
Write-Host "   from inside the repo), specifically models/necks/sfp.py's checkpoint_wrapper --"
pip install fairscale
Assert-LastExitCode "fairscale install"

Write-Host ""
Write-Host "Done. Repo at: $RepoDir"
Write-Host "Next: download checkpoints per docs\checkpoints.md into $RepoDir\ckpt\ (+ dist\ for the OCR model), then run:"
Write-Host "  `$env:AUTOHDR_HOME='$AutohdrHome'; setup\run_smoke_test.ps1"
