# Stage 4 Implementation Status

## Current: v0.5.2 model-only ground-consensus + temporal safeguards; acceptance deferred

See `V051_GROUND_FIRST_REPORT.md` for the implemented contract and actual frame-104 results.
The historical v0.5 package summary below predates local CPU inference and is retained for provenance.

The independent model-only lane now applies the Phase-4 ground/contact gate in
`stage4_metric3d/model_only_refinement.py`. A single foot or a minority cluster
cannot activate ground-first refinement. Ambiguous/unavailable evidence keeps
the SAM3D prior and is recorded with explicit status, source, candidate names,
rejection reasons and correction-bound diagnostics. This does not change the
Stage-1/3 integration solver.

Phase 5 also hardens the `sam3d-pitch-refined` solver: temporal terms require
exact consecutive frames and valid cameras, optimizer failures fall back to
initial roots, and reprojection/temporal/bound diagnostics are retained.
Camera covariance remains explicitly unavailable and is not treated as
calibrated uncertainty.

Phases 6–8 add model-only semantic safety and cache integrity: output tensors
must agree with `valid_mask`, uncertainty is explicitly uncalibrated, and
resume is allowed only with a matching content-addressed provenance sidecar.

## Phase 0 audit status

The Stage-4 Phase-0 provenance audit is implemented in
`tools/stage4_phase0_audit.py`. It hashes Stage-4 source/config/mapping files,
records upstream input and output availability, and fails closed when the real
Stage-1 camera state, Stage-3 state or SAM3D cache is missing. It never creates
or regenerates upstream artifacts.

The current workspace has no real frame-104 Stage-1/Stage-3/SAM3D baseline
artifacts at the paths recorded by the historical rerun script, so the audit
reports `integration_frame104=NOT_AVAILABLE_INTEGRATION_INPUTS`. This optional
integration status does not gate Stage-4-only model development. Independent
model accuracy remains `NOT_EVALUATED` until an authorized labeled benchmark is
available. Synthetic tests remain implementation evidence only.

Phase 0 now accepts an explicit `stage4-phase0-input-manifest-1.0` so paths are
portable across machines. It reports model-only readiness separately from the
optional integration-cache readiness and records the exact missing artifact
keys. The current workspace still lacks the Stage-4 SAM3D checkpoint/MHR
assets and an independent benchmark dataset, so real model-only inference and
accuracy are not yet evaluated. It also lacks the optional Stage-1/3 frame-104
integration artifacts; that does not block Stage-4-only improvement.

The current v0.5 SAM3D worker/pipeline is still integration-oriented because
its input contract derives boxes and observations from Stage 3 and camera
states from Stage 1. The model-only lane is therefore a Stage-4-owned runner
and dataset-adapter task. Phase 1 now includes the strict
`stage4-model-only-benchmark-manifest-1.0` validator and
`run_sam3d_model_only.py` direct SAM3D baseline. These accept benchmark RGB,
bbox, benchmark camera and metric GT by `record_id`; they do not import or
require Stage 1/3. The runner is ready for real inference only when the
licensed benchmark release and Stage-4 SAM3D checkpoint/MHR assets are
mounted. This status does not claim that the existing integration CLI runs
without upstream inputs.

## Phase 1 model-only intake status

Implemented:

- strict manifest schema with dataset release/license, sequence split and
  canonical `PITCH_WORLD_METRIC` coordinate declaration;
- image decoding, bbox bounds, camera dimensions/intrinsics/extrinsics,
  Pose23 GT shape/visibility and placeholder/missing-file checks;
- deterministic manifest and per-input SHA256 provenance;
- explicit rejection of Stage-1/Stage-3 paths in the model-only manifest;
- official SAM3D MHR70 direct runner and Pose23 world-coordinate conversion;
- model-only ground-first ablation using SAM3D MHR70 foot consensus, bounded
  root correction and explicit fallback when no ground anchor is usable;
- independent fail-closed evaluator with provenance checks, common-cohort
  metrics, coverage and failure-attribution slices;
- per-record/global MPJPE, root-aligned MPJPE and PA-MPJPE report skeleton.

Not yet available:

- licensed real benchmark release converted to the contract;
- verified SAM3D checkpoint, MHR model and official repository/runtime;
- real model-only inference and accuracy numbers.

Therefore `model_only_development=ALLOWED`, while pretrained inference and
research accuracy remain blocked/not evaluated by missing real Stage-4 assets.

Phase 2 adds `evaluate_model_only.py`, which re-evaluates saved outputs without
rerunning SAM3D. It checks all provenance and schema invariants, then reports
direct/ground-first common-cohort metrics and failure attribution. It cannot
turn development metrics or synthetic fixtures into a final accuracy claim.

Phase 3 now includes `compare_model_only_systems.py` and a systems manifest for
fair direct/ground-first or backend A/B comparisons on one exact model-only
cohort. It verifies benchmark/output/checkpoint provenance before comparison;
no backend is promoted without real locked-holdout evidence.

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
- Current Phase-9 verification: `91 passed` across `tests`, `tests_v04` and
  `tests_v05` in `nckh-env`, including provenance and headless visualization
  tests.
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
