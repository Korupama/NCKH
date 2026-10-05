# Stage 4 Phase 9 — acceptance decision

Date: 2026-10-04  
Scope: Stage 4 only

## Decision

`DEFER_ACCURACY_CLAIM`

## Evidence

- implementation gate: `PASS_IMPLEMENTATION`;
- full Stage-4 suite: `91 passed`;
- synthetic validator: `PASS`;
- model-only pretrained inference: `BLOCKED_MISSING_STAGE4_MODEL_ASSETS`;
- independent model-only metric accuracy: `NOT_EVALUATED_NO_INDEPENDENT_BENCHMARK_RECORDED`;
- accuracy claim allowed: `false`;
- research accuracy frozen: `false`.

## Required before an accuracy claim

1. Verified SAM3D checkpoint, MHR model and official runtime/repository.
2. Authorized independent benchmark with metric 3D GT.
3. Sequence-disjoint development, calibration and locked holdout manifests.
4. Paired direct-vs-refined metrics plus difficult-slice non-regression.
5. Calibration evidence if uncertainty is presented as calibrated.

Missing Stage 1/3 integration artifacts remain a separate optional integration
blocker; they do not invalidate the independent Stage-4 model-only contract.
