# NLP_cuoi_ky — AutoHDR reproduction

Reproduction of [SCUT-DLVCLab/AutoHDR](https://github.com/SCUT-DLVCLab/AutoHDR) (ACL 2025,
"Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document
Restoration") for the final assignment described in [`note.txt`](note.txt).

## Grading criteria (from `note.txt`)

> Loại 1: cài đặt đúng theo mã nguồn + data có sẵn = 80%. Cải tiến thêm có thể >100%.

- **80% bar**: get the official AutoHDR code running end-to-end on this machine, using the
  authors' released checkpoints → `setup/` + `docs/checkpoints.md`.
- **Beyond 80%**: once the real dataset (linked in `note.txt`) is validated, batch-run the
  pipeline over it and score restoration accuracy → `autohdr_data/evaluate.py`; later,
  retrain individual components (see "Future work" below).

Deadline: **3/9**.

## What AutoHDR does

A modular pipeline that mirrors an expert historian's restoration workflow:
1. **Damage detection** — DINO detector (mmdet, Swin-L backbone) locates damaged regions.
2. **OCR** — detects + recognizes legible characters.
3. **AutoHDR-Qwen2 (1.5B / 7B)** — an LLM predicts the missing/damaged characters from context.
4. **DiffHDR** — a diffusion UNet (`diffusers`) inpaints the visual appearance of the fix.

## This machine

12GB GPU (RTX 3060, driver 536.23, CUDA 12.2 — confirmed working). AutoHDR-Qwen2-7B alone
needs ~15GB in fp16 (won't fit); the **1.5B** variant needs ~3GB, and the full pipeline
(detector + OCR + diffusion + 1.5B LLM) totals an estimated ~5-6GB — comfortable. A
4-bit-quantized 7B run (~7-8GB estimated) is a documented stretch goal, not the initial target.

**Environment: Google Colab, not native Windows or WSL2/Docker.** Three things ruled out the
local machine, in order:
1. WSL2 GPU passthrough is blocked: Windows build 19041 (2004) predates the 19044 (21H2)
   minimum WSL2 needs for CUDA passthrough — confirmed by `nvidia-smi` failing inside WSL2
   even after converting the distro to WSL2 and updating its kernel.
2. Native Windows got much further (full env installed, every import verified, `infer_pipeline.py`
   ran until only the checkpoint files were missing — see `setup/windows_setup.md` for that
   whole story) but hit a wall that isn't fixable locally at all: the OCR detector/recognizer
   (`dist/det_model`, `dist/model_exe`) ship as **precompiled Linux ELF executables** with no
   Windows build anywhere in the download (confirmed via `file`) — `subprocess.Popen` can't
   launch a Linux binary on Windows, full stop.
3. Tried a hybrid fix for #2: launch just those two binaries inside the machine's existing WSL2
   Ubuntu-20.04 distro (general Linux execution works fine there even without GPU passthrough -
   confirmed both binaries start, bind their socket/pipes, and run a real inference call on CPU,
   with the Windows-side process reaching them via WSL2's automatic localhost port-forwarding).
   The bridge itself worked (patches kept - see below) but this machine only has **15.8GB total
   system RAM**, and stacking multiple heavy WSL2-hosted torch processes during testing hung the
   entire machine, not just WSL2. The real pipeline run would need Windows-side torch (LLM +
   diffusion + detector) and the WSL2-bridged OCR processes running *simultaneously* - too tight
   a margin on this much RAM to be worth the risk versus just using Colab.

`AutoHDR/utils/wsl_bridge.py` and the WSL-launch patches in `det_wrapper.py`/`reg_wrapper.py`
are kept in the repo, not reverted - they're gated behind `platform.system() == 'Windows'`, so
they're completely inert on Colab's Linux runtime (falls through to the original unmodified
behavior) and only matter again if this is revisited on a Windows machine with more RAM headroom.

Colab is real Linux with a GPU already driver-configured, so neither blocker applies — see
`notebooks/01_autohdr_colab_setup.ipynb`, which reuses everything learned from the Windows
attempt (`requirements.txt` gaps: `opencc`/`zhconv`/`fairscale`/unpinned `huggingface_hub`; the
torch-2.1.0-triplet downgrade for the `mmdet<mmcv2.2.0`-vs-Windows-wheels-only-≥2.2.0 conflict).
That torch/mmcv version conflict turned out to be **not** Windows-specific after all — checked
OpenMMLab's wheel index directly and `mmcv==2.1.0` has no Linux wheel past `torch2.1.0` either —
so the same downgrade is needed on Colab too. Compounding that: Colab's current default runtime
is Python 3.13, and this whole 2023/2024-era pinned stack (torch 2.1-2.3, mmcv 2.1.0) has no
`cp313` wheels at all, and `mim`/`openmim` (pulls in `openxlab`, which hard-pins
`setuptools~=60.2.0`) kept clobbering setuptools back to a version incompatible with Python
3.13's removed `pkgutil.ImpImporter`. Fix: the notebook builds a dedicated Python 3.10 venv
inside Colab and installs everything into that (plain `pip install mmcv -f <wheel-index>`
instead of `mim`, sidestepping `openxlab` entirely) rather than Colab's system Python.

The 3 checkpoint files already downloaded locally (LLM, DiffHDR, OCR) need uploading
to Google Drive once so Colab can reuse them instead of re-fighting BaiduYun's CAPTCHA — see the
notebook's first cell for the expected Drive folder layout. `setup/windows_setup.md` and
`setup/wsl_setup.md` are kept as a record of what was tried and why, not as the active path.

## Repo layout

```
notebooks/01_autohdr_colab_setup.ipynb   Active path: Colab env setup + smoke test (see above)
setup/                  Native-Windows / WSL2 attempts - kept as a record, not the active path:
  windows_setup.md         what was tried, and why it ultimately doesn't work here (Linux-only
                            OCR binaries) - install/import steps up to that point are still valid
  install_env.ps1           creates venv, clones AutoHDR, installs deps incl. mmcv/mmdet
  run_smoke_test.ps1        runs infer_pipeline.py on the repo's bundled example.jpg
  wsl_setup.md / *.sh      WSL2 path, blocked by this machine's Windows build (see above)
docs/checkpoints.md      Where to get + place the official pretrained checkpoints
autohdr_data/            Pure-Python (no torch) dataset tooling — runs directly on Windows:
  schema.py                dataclasses + loaders for the images/+labels/ annotation format
  validate.py               dataset sanity checks (CLI)
  convert_coco.py           labels -> COCO json (damage detector training format)
  convert_ocr.py            labels -> OCR ground truth (char bbox + text)
  evaluate.py               score restoration predictions vs. ground-truth characters
  download.py               fetch + unpack the dataset from the Google Drive link in note.txt
tests/                    pytest suite for autohdr_data, using a synthetic fixture
```

## Quick start

**1. Dataset tooling (already run against the real dataset — see below):**

```powershell
pip install -r requirements-local.txt
pytest tests/
python -m autohdr_data.validate --images data\FPHDR\FPHDR\images --labels data\FPHDR\FPHDR\labels
```

**2. Official pipeline (Colab, GPU):**

1. Upload the already-downloaded checkpoint files (`ckpt/AutoHDR-Qwen2-1.5B/`, `ckpt/unet/`,
   `dist/det_model/`, `dist/model_exe` under `AutoHDR/` locally) to a Google Drive folder — see
   `notebooks/01_autohdr_colab_setup.ipynb`'s first cell for the exact expected layout.
   `ckpt/damage_detect.pth` (Damage Localization Model) is still missing locally — download it
   per `docs/checkpoints.md` before or alongside this.
2. Open `notebooks/01_autohdr_colab_setup.ipynb` in Colab, set Runtime → GPU, run top to bottom.

## Dataset format

Downloaded (`FPHDR_FPHDR_syn.zip`, from the Google Drive link in `note.txt`) and extracted
to `data/`. It unpacks into two subsets, each `images/`+`labels/` as documented:

```
data/FPHDR/FPHDR/{images,labels}          1,664 real scanned pages
data/FPHDR_syn/FPHDR_syn/{images,labels}  7,138 synthetic pages (+ my_samples/, 9 files)
```

The two subsets' `labels/*.json` do **not** share exactly the same shape - confirmed by
inspecting real files, not assumed from the spec alone:

| | `FPHDR/` (real) | `FPHDR_syn/` (synthetic) |
|---|---|---|
| `columns` key | present (matches note.txt) | **absent** |
| char `char_id` | present | **absent** |
| char `grade` values | JSON `null` / `"light"` / `"medium"` / `"severe"` | literal string `"None"` / `"damaged"` (binary, no severity) |
| known gaps | 279 columns missing `column_id`, 4 chars missing `txt` | none found (full scan, 7,138 files) |

`autohdr_data/schema.py` normalizes both into one shape: missing `columns` → `[]`, missing
`column_id`/`char_id` → synthesized as `"<page>_col_<idx>"` / `"<page>_<idx>"`, missing/`"None"`
`txt`/`grade` → `""`/`None`. `GRADES` includes `"damaged"` as its own value alongside
`light`/`medium`/`severe` since the synthetic subset only records damaged-or-not, not severity.
`autohdr_data/validate.py` confirms both subsets parse cleanly under this normalization
(only the 4 pre-existing empty-`txt` real-data gaps are flagged - not a bug, the source data
itself has them).

## Future work (beyond the 80% bar)

Once the dataset is validated end-to-end, retrain individual components using the repo's
own scripts/configs: `ckpt/damage_detect.py` (mmdet DINO config, targets `HDR_Dataset/` —
what `convert_coco.py` is designed to produce), `document/tools/build_HDR.py` +
`document/scheduler/HDR_pipeline.py` for DiffHDR, and LoRA/SFT fine-tuning of Qwen2 on the
masked-character-prediction task for AutoHDR-Qwen2.
