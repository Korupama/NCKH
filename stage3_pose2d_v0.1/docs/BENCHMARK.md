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

## B3 — SoccerNet-Pose23 (future project-specific benchmark)

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

3DSP/COCO/Pose23 numerical gates should be frozen only after the real RTMW-L baselines are collected on the user's environment. Published RTMPose 3DSP numbers are comparison targets, not automatic pass/fail requirements for RTMW-L.
