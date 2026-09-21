# External assets and access

> Local handoff: [dataset locations and download policy](../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

## SoccerNet-v3D release assets

Stage 6 uses the public SoccerNet-v3D release assets such as `SNv3D.csv` and `yolo-sn-ball-opt.pt` through `download_open_assets.py`. They are not redistributed in this package.

## SoccerNet_raw_HQ frames

The real 2D detector benchmark needs image frames from the gated Hugging Face repository:

```text
repo:     SoccerNet/SoccerNet_raw_HQ
revision: frames-v3
layout:   <league>/<season>/<match>/Frames-v3.zip
```

After access is granted, authenticate locally:

```powershell
hf auth login
```

Then use the Stage-6 selective downloader:

```powershell
python download_soccernet_v3.py `
  --csv ".\data\SNv3D.csv" `
  --output-root "D:\SoccerNet" `
  --split test
```

The helper derives unique matches from `SNv3D.csv` and sends those exact archive paths as `allow_patterns` to `huggingface_hub.snapshot_download`. It does not require `Labels-v3.json` for this benchmark and does not extract `Frames-v3.zip`.

Do not put Hugging Face tokens or NDA passwords in code, notebooks, JSON reports, or command history. The normal path is cached authentication from `hf auth login`; an environment-variable token can be used only when explicitly requested through `--token-env`.

Review third-party dataset/model/dependency licenses and access terms before redistribution or non-academic deployment.

## v0.3.6 annotation note

The SoccerNet-v3D release README describes `optimized_d` as the optimized ball diameter in image space and `yolo-sn-ball-opt.pt` as fine-tuned on SoccerNet-v3D optimized boxes. Stage 6 does not redistribute these assets. Because the public CSV exposes only the optimized diameter, Stage 6 v0.3.6 explicitly reconstructs a centered square optimized box for 2D evaluation.

## v0.4.0 asset note

v0.4.0 adds no new third-party model or dataset dependency. Temporal refinement operates on the existing Stage-1 camera states and Stage-6 detector/tracker output. The cached-state refiner can therefore be exercised without downloading or rerunning YOLO.
