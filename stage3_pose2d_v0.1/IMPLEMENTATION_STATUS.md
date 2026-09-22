# Stage 3 v0.1 implementation status

## Implemented

- Stage2→Stage3 three-file adapter and strict structural preflight.
- Canonical COCO-WholeBody133 naming/order.
- Derived H36M17 adapter without changing the source-of-truth pose.
- Non-probabilistic RTMW raw-score semantics.
- Per-pose body/core/feet completeness checks.
- Bbox-relative geometric and broad bone sanity checks.
- Per-keypoint evidence states.
- Bbox-normalized temporal jitter/outlier diagnostic.
- Adjacent-frame left/right swap suspicion.
- Optional temporal estimate stored separately from raw x/y.
- Optional same-RTMW controlled re-crop fallback with preserved upstream pose.
- `TrackedPose2DState 1.0` and downstream handoff.
- Optional selected-frame QA visualization when source video is accessible.
- 3DSP reader + PDJ/AUC harness.
- COCO-WholeBody top-down export + optional official `xtcocotools` evaluation.
- Task-aware Pose23@t0 evaluator with visibility-aware PCK@0.05/@0.10,
  per-keypoint/per-group metrics, optional OKS, difficulty slices, anatomy
  evidence coverage and raw-vs-temporal provenance accounting.
- CLI, download helper, notebook skeleton and automated tests.

## Structural validation performed during packaging

A real legacy Stage-2 migration output was regenerated from the supplied historical SST+RTMW frames 85–87, then passed through Stage 3 cache-only mode. It produced 10/10 available and accepted candidate poses at `t0=86`. This is a plumbing/contract smoke test, **not a pose-accuracy benchmark**.

The historical RTMW raw scores in that smoke run have a median around 3 for a typical pose, which directly demonstrates why Stage 3 does not interpret the score as a probability.

## Still requires later work

- project-specific SoccerNet-Pose23 annotation and foot PCK;
- production Stage-3 coverage distribution on authorized replay cases;
- later error-driven decision on whether same-model fallback, RTMW-X, another
  pose model or fine-tuning is justified.

## Phase 3 validation

Phase 3 is complete within Stage 3 only. The evaluator is deterministic-tested
with synthetic fixtures and does not modify any other stage package. Primary
metrics score only raw observed coordinates at `t0`; temporal estimates are
reported separately and never silently substituted. The task manifest remains
empty until an authorized Pose23 annotation source is available, so no
task-specific accuracy claim is made yet.

Stage 3 package tests: `33 passed`.

## Phase 4 validation

Phase 4 baseline freeze is complete within Stage 3 only. The reproducibility
manifest and report are:

- `benchmark_results/phase4_baseline_run_manifest.json`;
- `validation_reports/PHASE4_BASELINE.md`.

The 3DSP and COCO-WholeBody lanes are frozen as diagnostics. The historical
cache-only artifact is frozen as a contract smoke test. Pose23@t0 remains
`BLOCKED_DATA_EMPTY` because its authorized task-specific manifest has zero
samples and its test split is not locked.
