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
- Future SoccerNet-Pose23 PCK evaluator.
- CLI, download helper, notebook skeleton and automated tests.

## Model backend ablation decision

Comparative ablation is complete, but RTMW-X is **not promoted**. The
workspace has a reproducible RTMW-L baseline and a fully evaluated RTMW-X
backend; RTMW-X did not improve the full 3DSP result. See
[`docs/PHASE7_MODEL_BACKEND_ABLATION.md`](docs/PHASE7_MODEL_BACKEND_ABLATION.md)
for provenance, metrics and re-open criteria. RTMW-L remains the production
default; RTMW-X is offline-only and no Stage-2 input or contract was changed.

## Structural validation performed during packaging

A real legacy Stage-2 migration output was regenerated from the supplied historical SST+RTMW frames 85–87, then passed through Stage 3 cache-only mode. It produced 10/10 available and accepted candidate poses at `t0=86`. This is a plumbing/contract smoke test, **not a pose-accuracy benchmark**.

The historical RTMW raw scores in that smoke run have a median around 3 for a typical pose, which directly demonstrates why Stage 3 does not interpret the score as a probability.

## Still requires user's real benchmark run

- full 3DSP RTMW-L PDJ/AUC;
- COCO-WholeBody Body/Foot/Whole AP sanity result;
- project-specific SoccerNet-Pose23 annotation and foot PCK;
- production Stage-3 coverage distribution on replay cases;
- later error-driven decision on whether same-model fallback, RTMW-X, another pose model or fine-tuning is justified.
