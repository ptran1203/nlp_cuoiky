# AutoHDR checkpoints

Straight from the official repo's README (`SCUT-DLVCLab/AutoHDR`). All links are BaiduYun
(pan.baidu.com) - not scriptable (login/CAPTCHA-gated), so download these manually via
browser once, then place the files as described.

| Model | Link | Extraction code |
|---|---|---|
| AutoHDR-Qwen2-1.5B *(target for the 12GB smoke test)* | https://pan.baidu.com/s/1j_HmyNDG0dOD6TyBHvqYwQ?pwd=W2wq | `W2wq` |
| AutoHDR-Qwen2-7B *(stretch goal, needs 4-bit quant)* | https://pan.baidu.com/s/1CUREGQIBoed1BgHjELguTQ?pwd=6o84 | `6o84` |
| DiffHDR (diffusion inpainting UNet) | https://pan.baidu.com/s/1fSKd5uQsiKp2uPQBdKtC3Q?pwd=63a3 | `63a3` |
| Damage Localization Model (DINO detector) | https://pan.baidu.com/s/1wGcT6Ktzqg_bOyc8NsV4Ig?pwd=2QC7 | `2QC7` |
| OCR Model | https://pan.baidu.com/s/1GfNQKIJ17Yf6QSv-dCaPEQ?pwd=1X88 | `1X88` |

The repo's official FPHDR dataset (for reference - **we're using the instructor's Google
Drive link from `note.txt` instead**, see `autohdr_data/download.py`):

| Data | Link | Extraction code |
|---|---|---|
| Real data | https://pan.baidu.com/s/1zpS4B3E0eZJ9Hza-HxpX4w?pwd=ryk3 | `ryk3` |
| Synthetic data | https://pan.baidu.com/s/1Lrd51vChv72f2WZSNx8R2w?pwd=m8yn | `m8yn` |

## Where to put them (confirmed from reading `infer_pipeline.py` directly, not just the README)

The script hardcodes these exact paths (relative to the repo root, which is now cloned at
`d:\Master\hk2\NLP_cuoi_ky\AutoHDR\`):

| Download | Extract/place at | Already present in the clone? |
|---|---|---|
| Damage Localization Model | `ckpt/damage_detect.pth` | no — `ckpt/damage_detect.py` (the mmdet config) is already there, just the `.pth` weights are missing |
| AutoHDR-Qwen2-1.5B | `ckpt/AutoHDR-Qwen2-1.5B/` | no |
| AutoHDR-Qwen2-7B | `ckpt/AutoHDR-Qwen2-7B/` (this is the script's *default* `--model_name_or_path` — pass `--model_name_or_path ./ckpt/AutoHDR-Qwen2-1.5B` to use the 1.5B one instead) | no |
| DiffHDR | `ckpt/unet/` (loaded via `UNet2DModel.from_pretrained('ckpt/unet')`) | no |
| OCR Model | unzip, then `dist/det_model/` + `dist/reg_model/` (hardcoded literal paths in `infer_pipeline.py`'s `detect()`/`main()`, not configurable via CLI flags) | no — `dist/` doesn't exist yet |

`ckpt/dic_31524.txt` (the OCR character dictionary) is already bundled in the repo - no
download needed for it.

## For this project (12GB GPU)

Only download **AutoHDR-Qwen2-1.5B** + DiffHDR + Damage Localization Model + OCR Model for
the initial smoke test. Skip the 7B checkpoint unless/until you attempt the 4-bit-quantized
stretch goal - and remember to pass `--model_name_or_path ./ckpt/AutoHDR-Qwen2-1.5B` when
running, since the script's default points at the 7B path.

## Confirmed by actually running it

Ran `infer_pipeline.py` against the fully-installed environment (see `setup/windows_setup.md`
for the version fixes that took): every import succeeds, the damage-detector model builds
successfully from `ckpt/damage_detect.py`, and it stops exactly at
`FileNotFoundError: ./ckpt/damage_detect.pth can not be found.` - i.e. the environment is
100% ready; the only remaining blocker is the checkpoint files themselves, which need a
manual browser download (BaiduYun share links are CAPTCHA-gated - confirmed not scriptable).
One more repo-local gap found this way: the repo's *own* vendored `AutoHDR/mmdet/` package
(shadows the pip-installed `mmdet` when running from inside the repo) needs `fairscale`,
also missing from `requirements.txt` - now in `setup/install_env.ps1`.

## Missing from `requirements.txt`

`infer_pipeline.py` imports `opencc` (`from opencc import OpenCC`) and `zhconv`
(`from zhconv import convert`) for Traditional/Simplified Chinese conversion, but neither is
listed in the repo's `requirements.txt`. Install them too:

```
pip install opencc-python-reimplemented zhconv
```

## Single-image only, not a dataset loop

`infer_pipeline.py`'s `if __name__ == '__main__':` block hardcodes `img_path = 'example.jpg'`
and calls `main(data=img_path, opt=opt)` once. The `--data_dir`/`--batch_infer` CLI args exist
in the parser but are never read anywhere in the script - they're unused. To run it over a
folder of images (e.g. our real dataset), either edit `img_path` per image, or use a small
wrapper that imports `main()` from the script and loops it - not yet written here.
