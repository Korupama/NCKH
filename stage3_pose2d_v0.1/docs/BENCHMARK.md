# Stage 3 evaluation protocol

> Local handoff: [dataset locations and download policy](../../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

## Principle

Pose accuracy and upstream candidate coverage are reported separately.

```text
Stage3PoseCoverage | Stage2 candidate
```

must never be presented as end-to-end footballer recall. Stage 2 can miss/exclude a footballer before Stage 3 receives the track.

## B0 — Structural and provenance tests

Required before any research benchmark:

- 100% schema/order validity for accepted WholeBody133 records;
- no ambiguous `(track_id, frame_index)` mapping;
- no silent coordinate transform;
- no silent temporal imputation;
- raw score semantics preserved as non-probabilistic RTMW SimCC evidence.

## B1 — 3D Shot Posture Dataset (3DSP)

Primary football-domain benchmark. The public dataset comes from SoccerNet broadcast footage and its annotated training portion contains 200 shots × 20 cropped images with 2D/3D posture labels.

Stage 3 evaluates RTMW zero-shot on the provided cropped player images. COCO-WholeBody133 predictions are mapped to a derived H36M17 view; the canonical 133 points are not discarded.

Metrics:

```text
PDJ@0.5
AUC over PDJ threshold [0, 0.5]
mean/median normalized joint error
per-group PDJ/AUC:
  Head, Shoulder, Elbow, Wrist, Body, Hip, Knee, Ankle
```

Normalization follows AutoSoccerPose: predicted-to-GT joint distance is divided by the GT distance between shoulder centre and hip centre.

Published zero-shot references from AutoSoccerPose Table 5:

| Model | PDJ | AUC |
|---|---:|---:|
| HRNet | 56.08% | 47.38% |
| DWPose | 67.94% | 55.80% |
| RTMPose | 89.51% | 73.56% |

Published RTMPose group PDJ: Head 96.81%, Shoulder 95.08%, Elbow 86.22%, Wrist 76.01%, Body 96.98%, Hip 95.62%, Knee 88.30%, Ankle 79.00%.

The overall PDJ/AUC are the safest directly comparable figures. `benchmark/dsp3_adapter.py` documents this project's explicit H36M group indices for reproducibility.

## B2 — COCO-WholeBody sanity benchmark

This benchmark verifies that the RTMW wrapper/keypoint ordering behaves as expected on the generic WholeBody task.

The harness uses **GT person boxes**, so the result isolates top-down pose estimation and is not a detector benchmark.

With `xtcocotools` installed, the official extended COCO evaluator is called for:

```text
keypoints_body
keypoints_foot
keypoints_face
keypoints_lefthand
keypoints_righthand
keypoints_wholebody
```

The official COCO-WholeBody sigma arrays are included in the harness. Report Body AP, Foot AP and WholeBody AP as the main checks.

## B3 — SoccerNet-Pose23 task-aware Pose23@t0 benchmark

The Phase-2 contract and annotation rules are defined in
[`POSE23_ANNOTATION_GUIDELINE.md`](POSE23_ANNOTATION_GUIDELINE.md). The local
template is
`data/task_pose23_t0/pose23_t0_manifest.json`; it is intentionally empty until
an authorized broadcast/offside annotation source is selected.

Planned annotation layout:

```text
17 COCO body joints + 6 foot joints = 23 points
```

Sampling should deliberately include small/far players, side/back views, blur, crowding, partial occlusion and frame-border truncation. Split by match/sequence rather than adjacent frames.

Metrics:

```text
PCK@0.05
PCK@0.10
median normalized error
Ankle PCK
Heel PCK
Toe PCK
```

The Stage-3 harness also reports per-keypoint metrics, anatomy evidence
coverage, visibility slices and difficulty slices for scale, occlusion, blur,
view, border truncation and crowding. The primary metric source is explicitly
`RAW_OBSERVED_ONLY`; `TEMPORAL_ESTIMATE_ONLY` points are excluded from primary
PCK/OKS and may be requested as a separate diagnostic.

Optional OKS uses pixel error divided by bbox area with the frozen 23-point
sigma vector in `benchmark/pose23_task.py`. The six foot sigmas are project
engineering defaults, so this is not an official COCO-WholeBody score.

Run the independent Stage-3 evaluator with:

```bash
python benchmark_stage3.py eval-pose23-task \\
  --ground-truth data/task_pose23_t0/pose23_t0_manifest.json \\
  --predictions <pose23_predictions.json> \\
  --output benchmark_results/task_pose23_t0/report.json \\
  --include-oks
```

The current versioned manifest intentionally has zero samples, so this command
must wait until an authorized Pose23 annotation source is populated. Synthetic
fixtures in `tests/test_pose23_task.py` validate the metric protocol now.

This benchmark becomes the main task-specific gate before Stage 5/8 because 3DSP does not provide heel/toe ground truth.

## B4 — Production coverage/QA diagnostics

`tracked_pose_2d_state.json` reports:

```text
PoseCoverageAtT0_given_stage2_candidate
AcceptedPoseCoverageAtT0_given_stage2_candidate
ValidPoseCoverageAtT0_given_stage2_candidate
FootPoseCoverageAtT0_given_stage2_candidate
```

Per-track temporal diagnostics report normalized jitter, temporal outliers and suspected left/right swaps.

## Numerical gate policy

v0.1 freezes structural gates, not football-specific accuracy thresholds. Initial engineering target:

```text
PoseCoverageAtT0_given_stage2_candidate >= 0.98
```

Phase 4 freezes the current RTMW-L 3DSP and COCO-WholeBody diagnostics in
`validation_reports/PHASE4_BASELINE.md` and
`benchmark_results/phase4_baseline_run_manifest.json`. These are reference
baselines, not automatic pass/fail requirements and not task-specific offside
accuracy. Pose23 numerical gates remain pending until the authorized manifest
has samples and a locked test split. Published RTMPose 3DSP numbers are
comparison targets, not automatic pass/fail requirements for RTMW-L.
