"""Launch the OCR det_model/model_exe binaries via WSL2 when running on native Windows.

Those two binaries are precompiled Linux ELF executables (PyInstaller-frozen Python apps) with
no Windows build anywhere in the official download - `subprocess.Popen` can't launch a Linux
binary directly on Windows. Confirmed on this machine:
  - det_model starts and binds its socket fine inside WSL2 with no GPU access (falls back to
    CPU - it does not hard-require CUDA at startup).
  - WSL2's automatic localhost port-forwarding lets a Windows-side process reach a TCP server
    bound inside WSL2 via plain `localhost:<port>` - no extra networking setup needed.
  - `wsl.exe <cmd>` invoked as a Windows subprocess transparently proxies stdin/stdout/stderr
    between the Windows parent and the Linux child process, so the stdin/stdout-pipe-based
    model_exe wrapper (reg_wrapper.py) works the same way as det_model's socket-based one.

Cold start through this bridge is slow (WSL2 process start + a full torch import inside the
frozen binary) - observed ~20-25s before det_model's socket was actually listening. Callers
that retry-connect need a generous budget accordingly (see det_wrapper.py's max_retries).
"""
from __future__ import annotations

import os
import platform

WSL_DISTRO = os.environ.get('AUTOHDR_WSL_DISTRO', 'Ubuntu-20.04')


def is_windows() -> bool:
    return platform.system() == 'Windows'


def to_wsl_path(win_path: str) -> str:
    """Convert an absolute Windows path to its WSL /mnt/<drive>/... equivalent."""
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    drive_letter = drive.rstrip(':').lower()
    rest = rest.replace('\\', '/').lstrip('/')
    return f'/mnt/{drive_letter}/{rest}'


def wsl_command(executable_rel_path: str, *args: str) -> list:
    """Build a `wsl.exe` command list that runs `executable_rel_path` (relative to the current
    working directory) inside WSL2, with its WSL-side working directory set to match - so
    relative paths in callers behave the same as a native (non-WSL) launch would."""
    wsl_cwd = to_wsl_path(os.getcwd())
    return ['wsl.exe', '-d', WSL_DISTRO, '--cd', wsl_cwd, '--', executable_rel_path, *args]
