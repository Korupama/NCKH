# Stage 4 Implementation Status

## Current: v0.5.1 ground-first hotfix

See `V051_GROUND_FIRST_REPORT.md` for the implemented contract and actual frame-104 results.
The historical v0.5 package summary below predates local CPU inference and is retained for provenance.

## Historical v0.5 package summary

## Implemented

- official SAM3D MHR70 native cache;
- Stage-1 intrinsics passed into the isolated SAM3D worker;
- official MHR70 semantic verification and MHR70 ↔ RTMW Pose23 mapping;
- `pred_cam_t` direct baseline;
- pitch-ground root candidates from RTMW foot pixels;
- root-only robust least-squares refinement;
- world-space temporal regularization;
- Stage-1 camera/world reconstruction;
- v0.5 preflight and four quality-gate structure;
- full state, compact downstream handoff, QA report and top-down visualization;
- v0.4 Field Converter backend retained as optional;
- v0.3 fixed-height and v0.2 legacy full-3D paths retained.

## Package validation

- `39 passed` across legacy, v0.4 and v0.5 tests.
- deterministic v0.5 synthetic validation: PASS.
- synthetic SAM translation error improved from about `0.829 m` direct to about `0.010 m` after pitch-constrained refinement in the controlled test.
- synthetic camera-convention P95 is approximately numerical zero.

## Supplied real-artifact structural check

The uploaded Stage-3/Stage-1 artifacts are compatible with the v0.5 worker manifest:

- selected frame: 86;
- source FPS: 30;
- image size: 1920×1080;
- Stage-3 window: frames 56–116 (61 frames);
- player/goalkeeper tracks: 13;
- finite bbox coverage: 695/793 ≈ 87.64%;
- Stage-1 optimized camera states available for the full 213-frame shot.

## Not executed in this build

- real official SAM3D checkpoint inference, because the gated model/runtime is not mounted in this environment;
- real frame-86 v0.5 world pose;
- metric GT evaluation;
- Stage-8/9 offside-oriented evaluation.

`research_accuracy_frozen = false`.
