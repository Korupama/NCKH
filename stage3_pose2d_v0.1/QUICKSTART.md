# Quickstart — Stage 3 v0.1

> Local handoff: [dataset locations and download policy](../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

## 1. Install and test

```powershell
cd D:\NCKH\stage3_pose2d_v0.1
python -m pip install -e ".[all]"
pytest -q
```

## 2. Check the Stage-2 handoff only

```powershell
python run_stage3.py `
  --stage2-dir "D:\NCKH\stage2_sst_rtmw_v1.3.1\runs\stage2" `
  --output-dir ".\runs\stage3_v01" `
  --preflight-only
```

Expected preflight conditions:

```text
EntityTrackState readable
RAW_DISTORTED_PIXEL coordinate space
candidate IDs resolve to Stage-2 tracks
unique candidate bbox at t0
RTMW cache schema is stage2-raw-rtmw-track-cache-1.0
raw RTMW score semantics are declared
```

A missing RTMW pose at `t0` is a warning rather than structural failure, because Stage 3 can explicitly return `MISSING` or use controlled fallback.

## 3. Run Stage 3 from the existing RTMW cache

```powershell
python run_stage3.py `
  --stage2-dir "D:\NCKH\stage2_sst_rtmw_v1.3.1\runs\stage2" `
  --output-dir ".\runs\stage3_v01"
```

Inspect:

```text
runs\stage3_v01\tracked_pose_2d_state.json
runs\stage3_v01\stage3_downstream_handoff.json
runs\stage3_v01\selected_frame_pose_qa.png   # when source video path is valid
```

The important Stage-3 coverage metrics are conditional on Stage-2 candidates:

```text
PoseCoverageAtT0_given_stage2_candidate
AcceptedPoseCoverageAtT0_given_stage2_candidate
ValidPoseCoverageAtT0_given_stage2_candidate
FootPoseCoverageAtT0_given_stage2_candidate
```

Do not reinterpret them as end-to-end footballer recall.

## 4. Optional temporal estimates

```powershell
python run_stage3.py ... --emit-temporal-estimates
```

This only fills `temporal_estimate_xy` for a missing point when adjacent frames support an estimate. The raw `x/y` remain missing.

## 5. 3DSP benchmark

```powershell
# Inspect the existing dataset first; download only if missing (see linked policy).
python benchmark_stage3.py inspect-3dsp --root ".\data\3dsp"
python benchmark_stage3.py run-3dsp `
  --root ".\data\3dsp" `
  --split train `
  --rtmw-model ".\weights\rtmw_l_384x288.onnx" `
  --device cpu `
  --output-dir ".\benchmark_results\3dsp_rtmw_l"
```

The public 3DSP `train` split contains 2D pose labels. Its public `test` portion is primarily a tracklet-only release, so pose-accuracy evaluation should use the annotated portion unless additional labels are supplied.

## 6. COCO-WholeBody sanity benchmark

Install the official extended COCO evaluator separately if needed:

```powershell
pip install xtcocotools
```

Then:

```powershell
python benchmark_stage3.py run-coco-wholebody `
  --images-root "D:\Datasets\coco\val2017" `
  --annotations "D:\Datasets\coco\annotations\coco_wholebody_val_v1.0.json" `
  --rtmw-model ".\weights\rtmw_l_384x288.onnx" `
  --output-dir ".\benchmark_results\coco_wholebody"
```

This is a top-down pose diagnostic using GT person boxes. It does not measure Stage-2 human detection.
