# Stage 4 WorldPose Benchmark Report

Date: 2026-10-04  
Scope: Stage 4 model-only lane  
Status: `EVALUATED`

## Summary

Stage 4 was benchmarked independently on the official WorldPose test split
using the frozen KASportsFormer WorldPose detected-2D pretrained checkpoint.
The benchmark does not import or regenerate Stage 1 or Stage 3 artifacts.

| Metric | Official upstream macro-by-action result |
|---|---:|
| MPJPE | **34.276 mm** |
| P-MPJPE | **22.008 mm** |
| Acceleration error | **3.022 mm** |

The official result is the macro-average over 26 WorldPose game/action
buckets, matching the upstream evaluator. The micro-average over all evaluated
frames is MPJPE 34.047 mm, P-MPJPE 21.865 mm and acceleration error 2.882 mm.

## Dataset and Provenance

- Dataset: `WorldPose`.
- Artifact: `benchmark_data/worldpose/wp_hr_conf_cam_source_final.pkl`.
- SHA256: `c324f402501c123f4ef3908089a23ebd7d348e26f12e2d0d11aefbd2bd186e31`.
- Size: 2,304,872,667 bytes.
- Split: official sequence/game-disjoint `test` split.
- Test buckets: 26 games/actions.
- Evaluation windows: 23,466.
- Evaluation frames: 633,582.
- Window length and stride: 27 frames / 27 frames.
- Input: WorldPose detected 2D joints plus confidence.

WorldPose is used here as a labeled metric-3D benchmark. The evaluated
coordinate scope is root-relative metric pose, in millimetres.

## Model and Provenance

- Model: `KASportsFormer`.
- Variant: WorldPose detected-2D checkpoint.
- Checkpoint: `weights/kasportsformer/kasportsformer-wp-det.pth`.
- SHA256: `a4e0b9018e4755676a8421edbf8063375d7d22eb2f423f6aa4ba3883ee90a767`.
- Size: 355,907,646 bytes.
- Upstream source snapshot: `.tmp_kasportsformer`.
- Device: CPU, `torch 2.14.0+cpu` in `nckh-env`.
- Flip test-time augmentation: enabled.
- Seed: `114514`.

The upstream model imports `timm` only for `DropPath` in this inference path.
Because `nckh-env` did not have `timm`, the benchmark runner supplies an
equivalent local inference-only implementation without modifying the upstream
source or installing packages.

## Metrics

### Frame-level distribution

| Metric | Mean | Median | P90 | P95 | Count |
|---|---:|---:|---:|---:|---:|
| MPJPE | 34.047 mm | 28.974 mm | 51.576 mm | 64.314 mm | 633,582 |
| P-MPJPE | 21.865 mm | 18.332 mm | 33.784 mm | 43.755 mm | 633,582 |
| Acceleration error | 2.882 mm | 1.919 mm | 6.152 mm | 8.408 mm | 586,650 |

### Per-joint MPJPE

| Joint | MPJPE |
|---|---:|
| Pelvis | 0.000 mm |
| Right hip | 6.711 mm |
| Right knee | 35.325 mm |
| Right ankle | 60.039 mm |
| Left hip | 6.746 mm |
| Left knee | 34.245 mm |
| Left ankle | 59.168 mm |
| Spine | 14.233 mm |
| Thorax | 29.802 mm |
| Neck | 32.654 mm |
| Head | 36.443 mm |
| Left shoulder | 33.485 mm |
| Left elbow | 41.702 mm |
| Left wrist | 55.238 mm |
| Right shoulder | 33.903 mm |
| Right elbow | 42.272 mm |
| Right wrist | 56.830 mm |

Upper-body MPJPE is 37.656 mm and lower-body MPJPE is 33.706 mm.

### By-game range

The lowest and highest game-level MPJPE values are:

- Lowest: `CRO_MOR_181141`, 28.179 mm.
- Highest: `ARG_CRO_223805`, 54.343 mm.

The complete per-game result is preserved in
`benchmark_results/worldpose_kasportsformer_test.json`.

## Acceptance and Limitations

- WorldPose root-relative 3D accuracy: `EVALUATED`.
- Pitch-world global placement accuracy: `NOT_EVALUATED`.
- Stage 1/Stage 3 integration accuracy: `NOT_EVALUATED`.
- Accuracy claim for the full Stage 4 pitch-world product: `false`.

This benchmark measures the pretrained Stage 4 pose model under the WorldPose
protocol. It does not establish camera calibration quality, pitch-ground
placement, global player coordinates, offside-line accuracy, or downstream
offside decisions. The WorldPose result must therefore not be presented as a
global pitch-world accuracy claim.

## Reproduction

From `stage4_metric3d_v0.2.0`, with `nckh-env`:

```powershell
$env:PYTHONPATH = "D:\GitHub\NCKH\.tmp_kasportsformer"
C:\Users\Admin\anaconda3\envs\nckh-env\python.exe benchmark_worldpose.py `
  --data benchmark_data\worldpose\wp_hr_conf_cam_source_final.pkl `
  --checkpoint weights\kasportsformer\kasportsformer-wp-det.pth `
  --kasportsformer-root D:\GitHub\NCKH\.tmp_kasportsformer `
  --config D:\GitHub\NCKH\.tmp_kasportsformer\configs\worldpose-det-kasportsformer.yaml `
  --report benchmark_results\worldpose_kasportsformer_test.json `
  --batch-size 8 `
  --device cpu
```

The machine-readable result is
`benchmark_results/worldpose_kasportsformer_test.json`. The runner is
`benchmark_worldpose.py`.
