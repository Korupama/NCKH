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

## Phase 8 decision

Phase 8 is **ANNOTATION_READY, not TRAIN_READY**. A sequence-disjoint
SoccerNet-GSR task manifest and WholeBody133 validator are now available.
Training remains gated on human-verified keypoints, reviewer metadata and
confirmed data rights. See [`docs/MODEL_FINE_TUNING.md`](docs/MODEL_FINE_TUNING.md).

Phase 8A pseudo-label self-training preparation is complete: 2,936/2,986 train
tasks passed RTMW-L QA and were exported as train-only pseudo-labels. No
validation/test pseudo-labels were created; no student checkpoint or final
accuracy claim exists yet.

## Phase 9 decision

Final Stage-3 integration is complete with explicit limits: shot-level
internal holdout, immutable Stage-2 contract smoke and full regression pass.
Production fallback remains deferred and COCO official evaluation is
unavailable on the current Windows environment. See
[`docs/FINAL_INTEGRATION_REPORT.md`](docs/FINAL_INTEGRATION_REPORT.md).

## Structural validation performed during packaging

A real legacy Stage-2 migration output was regenerated from the supplied historical SST+RTMW frames 85–87, then passed through Stage 3 cache-only mode. It produced 10/10 available and accepted candidate poses at `t0=86`. This is a plumbing/contract smoke test, **not a pose-accuracy benchmark**.

The historical RTMW raw scores in that smoke run have a median around 3 for a typical pose, which directly demonstrates why Stage 3 does not interpret the score as a probability.

## Still requires user's real benchmark run

- full 3DSP RTMW-L PDJ/AUC;
- COCO-WholeBody Body/Foot/Whole AP sanity result;
- project-specific SoccerNet-Pose23 annotation and foot PCK;
- production Stage-3 coverage distribution on replay cases;
- later error-driven decision on whether same-model fallback, RTMW-X, another pose model or fine-tuning is justified.
