# Stage 3 — Tracked 2D Whole-Body Pose QA v0.1

Stage 3 consumes the frozen Stage-2 handoff and converts the shared raw RTMW WholeBody cache into a traceable, quality-labelled `TrackedPose2DState` for later ground/team and 3D-body stages.

## Scope

Stage 3 **does**:

- preserve the Stage-2 persistent `track_id` and selected frame `t0`;
- validate the Stage2→Stage3 interface;
- keep COCO-WholeBody **133 keypoints** as the canonical pose representation;
- distinguish raw RTMW SimCC evidence from calibrated probability;
- attach anatomy/geometry/temporal QA states without silently changing raw coordinates;
- optionally re-run the same RTMW-L model on controlled re-crops when a cached pose is missing/degraded;
- export a downstream handoff and benchmark adapters for 3DSP, COCO-WholeBody and future SoccerNet-Pose23.

Stage 3 **does not** assign teams, project body parts onto the pitch, reconstruct 3D, decide IFAB legal body parts, or compute offside geometry.

## Primary inputs

```text
stage2_entity_tracks.json
stage2_rtmw_track_cache.json
stage3_handoff.json
```

The expected Stage-2 coordinate space is `RAW_DISTORTED_PIXEL`. RTMW scores are expected to use `RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY` semantics.

## Primary outputs

```text
tracked_pose_2d_state.json
stage3_downstream_handoff.json
selected_frame_pose_qa.png      # only when replay video is still accessible
```

The source of truth remains COCO-WholeBody133:

```text
17 body + 6 feet + 68 face + 21 left hand + 21 right hand = 133
```

The 17-point H36M view is an adapter only; it never replaces the canonical 133-point pose.

## Install

```powershell
cd D:\NCKH\stage3_pose2d_v0.1
python -m pip install -e ".[all]"
pytest -q
```

## Production/cache-only run

```powershell
python run_stage3.py `
  --stage2-dir "D:\NCKH\stage2_sst_rtmw_v1.3.1\runs\stage2" `
  --output-dir ".\runs\stage3_v01"
```

This path does **not** run RTMW again.

## Controlled fallback re-inference

Only enable this when the original replay path is still accessible and the same public RTMW-L ONNX is available:

```powershell
python run_stage3.py `
  --stage2-dir "D:\...\runs\stage2" `
  --output-dir ".\runs\stage3_v01_fallback" `
  --fallback-reinfer `
  --fallback-crop-scales "1.0,1.1,1.2" `
  --rtmw-model ".\weights\rtmw_l_384x288.onnx"
```

If fallback wins, the original Stage-2 pose is preserved under `upstream_raw_pose`. Replacement is explicit in provenance.

## Public benchmark assets

Download the public RTMW-L ONNX archive:

```powershell
python download_open_assets.py rtmw-l-384x288 --output-dir .\weights
```

Download the public 3D Shot Posture Dataset (3DSP):

```powershell
python download_open_assets.py 3dsp --output-dir .\data
```

3DSP is used as the primary football-domain 2D pose benchmark. COCO-WholeBody remains the generic implementation sanity benchmark. A project-specific 23-keypoint football dataset is reserved for later heel/toe evaluation.

See `QUICKSTART.md`, `docs/STAGE3_SPEC.md`, `docs/BENCHMARK.md`, and `docs/MODEL_PROVENANCE.md`.
