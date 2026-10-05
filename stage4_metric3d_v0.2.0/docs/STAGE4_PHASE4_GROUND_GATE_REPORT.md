# Stage 4 Phase 4 — model-only ground/contact gate

Date: 2026-10-04  
Scope: Stage 4 model-only lane only

## Completed engineering work

`stage4_metric3d/model_only_refinement.py` now refuses to apply a pitch-ground
translation correction unless the foot evidence forms a robust consensus:

- cluster radius: `0.75 m` in camera-root space;
- minimum consensus count: `2` candidates;
- required support: strictly greater than `50%` of available candidates;
- distal foot landmarks are preferred;
- both ankles are an explicit degraded fallback when no distal foot candidate
  is usable;
- invalid camera state, pitch geometry, evidence and correction bounds fail
  closed;
- correction remains bounded by `max_correction_m`.

The fallback result remains the frozen SAM3D prior. A single-frame pipeline does
not claim `AIRBORNE_SUPPORTED`; insufficient evidence is reported as
`CONTACT_AMBIGUOUS` or `GROUND_UNAVAILABLE`.

## Artifact diagnostics

Each model-only refinement record now includes:

- `status`, `reason` and `policy`;
- `source`, candidate names and consensus names;
- candidate count, consensus count and fraction;
- candidate spread;
- candidate rejection reasons;
- whether root correction was bounded;
- whether refinement was actually applied.

The evaluator aggregates status/reason/source counts and bounded-correction
counts in `ground_first_diagnostics_summary`.

## Verification

```text
model-only refinement/evaluation/system tests: 15 passed
full Stage-4 suite: 79 passed
```

The output contract is versioned as
`stage4-model-only-sam3d-output-1.1` and the report as
`stage4-model-only-evaluation-report-1.1`. The evaluator accepts old output
schema `1.0` for compatibility.

## Limitations and next work

No real pretrained inference or independent metric-GT benchmark is mounted in
the workspace, so this report does not contain accuracy numbers. The remaining
Phase-4 work is real-data ablation and holdout evaluation for confidence
weighting, temporal contact support, airborne/occluded slices and non-regression
of global position metrics.
