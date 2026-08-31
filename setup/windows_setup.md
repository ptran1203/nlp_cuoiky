# Native Windows setup (no WSL)

WSL2 GPU passthrough was tried and found blocked on this machine: Windows build 19041
(version 2004) predates the 19044 (21H2) minimum WSL2 needs for CUDA passthrough — confirmed
by `nvidia-smi` failing inside WSL2 with `GPU access blocked by the operating system` even
after converting the distro to WSL2 and updating the WSL kernel. That requirement is a hard
OS-version floor, not something a driver/kernel update alone fixes.

Since the NVIDIA driver already works natively on Windows (`nvidia-smi` succeeds directly,
RTX 3060 12GB, driver 536.23, CUDA 12.2), the plan runs **directly on native Windows Python**
instead — no WSL, no Docker (Docker Desktop's GPU passthrough uses the same WSL2 mechanism
and would hit the identical wall).

`setup/wsl_setup.md` / `install_env.sh` / `run_smoke_test.sh` are kept as-is in case this
machine's Windows is updated to 21H2+ later and WSL2 becomes viable — but the active path is
the scripts below.

## What's already confirmed on this machine

- NVIDIA driver 536.23, CUDA 12.2, RTX 3060 12GB — works natively (`nvidia-smi`).
- Python 3.10.6 already installed and on PATH — matches the repo's pinned Python 3.10.

## Package version deviation from the repo's requirements.txt

The repo pins `torch==2.3.0`/`torchvision==0.18.0`/`torchaudio==2.3.0`, but `install_env.ps1`
installs the `torch==2.1.0`/`torchvision==0.16.0`/`torchaudio==2.1.0` triplet instead - not an
oversight, forced by a real constraint chain discovered while running this on Windows:

- `mmdet==3.3.0` hard-requires `mmcv < 2.2.0` (checked mmdet's own source directly).
- OpenMMLab only publishes **Windows** wheels for `mmcv` against **torch 2.3.0** starting at
  `mmcv 2.2.0` - nothing in mmdet's compatible `[2.0, 2.2)` range has a torch2.3.0 Windows
  wheel, so `mim install mmcv==2.1.0` fell back to building from source every time, which
  fails on a `pkg_resources`/`setuptools` incompatibility (fixable, but only by installing
  MSVC Build Tools + a full CUDA dev toolchain - a much bigger install for no real benefit).
- `mmcv==2.1.0` *does* have prebuilt Windows wheels, but only up through `torch 2.1.0`.

So torch is pinned down to `2.1.0` to keep `mmcv`/`mmdet` on their intended pinned versions
via prebuilt wheels. `transformers`/`diffusers`/`accelerate` don't hard-require `torch 2.3.0`,
so this shouldn't affect the LLM or diffusion stages - only the damage-detector stage
(mmdet/mmcv) actually needed this constraint satisfied.

## Scripts

- `setup/install_env.ps1` — creates a venv at `<project>\venv`, clones AutoHDR to
  `<project>\AutoHDR` (already done), installs deps (with the torch-version fix above and the
  `opencc`/`zhconv` gap filled in), and installs `mmcv`/`mmdet`/`mmpretrain` via `mim`.
- `setup/run_smoke_test.ps1` — runs the official `infer_pipeline.py` unmodified on the
  repo's bundled `example.jpg`, logging peak GPU memory.

Status: `install_env.ps1` has been run (with fixes applied along the way as each Windows-
specific issue surfaced - triton excluded, mmcv-family excluded from the plain pip pass,
torch pinned to the 2.1.0 triplet per above). Checkpoints still need manual download per
`docs/checkpoints.md` before `run_smoke_test.ps1` can produce a real result.
