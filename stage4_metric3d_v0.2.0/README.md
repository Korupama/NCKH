# Stage 4 v0.5.1 — Ground-First SAM3D Root Refinement

Current local handoff: read [v0.5.1 report](V051_GROUND_FIRST_REPORT.md) first.
Ground consensus now initializes refinement independently of SAM depth disagreement.
The CPU frame-104 run and 48-test result are recorded there. Older v0.5 architecture
notes below remain background, not the latest ground-gating specification.
Workspace-wide navigation: [handoff guide](../README_BAN_GIAO.md).

Stage 4 v0.5 replaces the v0.4 Field Converter backend as the default production candidate. The new path uses the official SAM 3D Body metric relative skeleton and camera translation, then refines **one global translation per player/frame** with Stage-1 pitch geometry, Stage-3 RTMW 2D observations, and optional temporal smoothness.

```text
Stage 1 CameraState (K, R, C, pitch Z=0)
              +
Stage 3 track/bbox + RTMW WholeBody133
              +
       SAM 3D Body MHR70
     J_rel_cam + pred_cam_t
              ↓
     root-only translation refinement
       - RTMW reprojection
       - foot/pitch ground anchor
       - SAM3D translation prior
       - world-space temporal term
              ↓
    coherent world-grounded skeleton
              ↓
 Stage-1 metric pitch coordinates XYZ
```

## Backend status

- **`sam3d-pitch-refined` — v0.5 production candidate.** No Field Converter normalization statistics are required.
- **`sam3d-direct` — v0.5 ablation.** Uses `pred_cam_t` directly, with no root refinement.
- **`field-converter-tcn` — v0.4 optional backend.** Retained for future ablation, but external inference remains blocked unless the exact upstream `normalization_stats.npz` used with the released checkpoint is available.
- **`fixed-height-v03` — v0.3 baseline.** Retained only for comparison.
- **`run_stage4_legacy3d.py` — v0.2 legacy full-3D optimizer.**

`research_accuracy_frozen` remains `false` until real SAM3D inference is benchmarked and metric/offside-oriented evaluation is completed.

## Why v0.5 avoids the v0.3 failure mode

v0.3 localized each body part independently by intersecting its image ray with a fixed-height plane. Elevated joints could therefore spread several metres over the pitch.

v0.5 keeps the SAM3D relative skeleton coherent and optimizes only a translation vector:

```text
J_cam(j) = J_rel(j) + translation_cam
```

A root correction moves the complete body together. Stage 4 itself cannot separate the head from the feet by several metres unless the native SAM3D relative pose is already invalid.

## Official SAM3D schema

The worker now stores the official **MHR70** keypoints instead of forcing an undocumented 25-joint representation. Stage 4 contains a verified semantic adapter from MHR70 to the 23 canonical body/foot points used by RTMW/Stage 8.

Notable indices:

```text
MHR70 left_wrist  = 62
MHR70 right_wrist = 41
left/right toe-tip landmarks are mapped to the corresponding RTMW toe semantics
```

The mapping is recorded in `schemas/sam3d_mhr70_to_rtmw_wholebody_v1.json`.

## Primary outputs

```text
world_grounded_pose_state.json       schema: world-grounded-pose-state-1.1
stage4_downstream_handoff.json       schema: stage4-downstream-handoff-2.1
stage4_quality_report.json
stage4_preflight.json
selected_frame_world_pose_topdown.png
selected_frame_overlay.png           when source video/frames are locally accessible
```

All downstream coordinates use the canonical Stage-1 frame:

```text
origin = pitch centre
X = goal-to-goal
Y = touchline-to-touchline
Z = up
units = metres
```

Stage 4 does **not** assign teams, infer attack direction, localize the ball, choose Law-11 legal body points, identify the second-last opponent, or emit `ONSIDE/OFFSIDE_POSITION`.

See `QUICKSTART.md`, `docs/STAGE4_SPEC.md`, and `QA_v0.5.0.md`.
