# WSL2 setup for running AutoHDR locally (12GB GPU)

AutoHDR's own stack (Ubuntu 20.04, Python 3.10, CUDA 11.8, `mmcv`/`mmdet`) builds reliably
on Linux and is painful on native Windows. WSL2 gives a real Linux userspace with CUDA
passthrough to your existing NVIDIA driver, so this is the path we're using.

## 1. Install WSL2 + Ubuntu

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted, then open the "Ubuntu" app once to finish first-time account setup.

## 2. GPU driver

Do **not** install an NVIDIA driver inside WSL. WSL2 uses the driver already installed on
Windows (make sure it's a recent Game Ready / Studio driver that lists "WSL" support, which
all current drivers do). Verify from inside WSL:

```bash
nvidia-smi
```

You should see your GPU (12GB) and a CUDA version listed. If this fails, update the Windows
NVIDIA driver first (not a WSL-side package) and restart WSL (`wsl --shutdown` from
PowerShell, then reopen Ubuntu).

## 3. Base packages

```bash
sudo apt update
sudo apt install -y build-essential git wget curl python3.10 python3.10-venv python3-pip
```

## 4. Project checkout

Run `setup/install_env.sh` (from inside WSL, in this project directory reached via
`/mnt/d/Master/hk2/NLP_cuoi_ky`) - it creates the Python env, clones AutoHDR, and installs
all pinned deps. See that script for details.

## Notes

- Keep the AutoHDR checkout and any large dataset/checkpoint files on the Linux filesystem
  (e.g. `~/autohdr/`), not under `/mnt/d/...` - cross-filesystem I/O through `/mnt` is much
  slower and can bottleneck data loading.
- `docs/checkpoints.md` lists the checkpoint files to download manually and where to place
  them once the repo is cloned.
