# Stage 4 Phase 5 — temporal and reprojection safeguards

Date: 2026-10-04  
Scope: Stage 4 `sam3d-pitch-refined` implementation lane

## Implemented

- Temporal regularization uses only exact consecutive frame triplets with
  `frame[t+1] - frame[t] == 1` and valid camera states.
- Preflight reports temporal support by track and lists the exact triplets.
- Per-track output distinguishes `APPLIED`, `AVAILABLE_NOT_APPLIED`,
  `TEMPORAL_NOT_AVAILABLE` and `DISABLED`.
- Optimizer exceptions or non-finite/unsuccessful solutions fall back to the
  initial root sequence and expose `fallback_to_initial=true`.
- Optimizer diagnostics include active bound count and reprojection candidate,
  valid and invalid counts.
- Config validation is fail-closed for non-finite uncertainty scales,
  thresholds, unsupported solver losses and malformed XYZ sigma vectors.
- Camera covariance is explicitly unavailable in the current contract; the
  camera is treated as fixed for this lane.

## Tests

```text
Phase-5 temporal/refiner + v0.5 regression tests: 11 passed
```

The tests cover exact consecutive triplets, gapped sequences, invalid camera
states, optimizer exception fallback, and existing synthetic ground-first
behavior.

## Limitations

No real SAM3D checkpoint, licensed model-only benchmark or independent metric
ground truth is available in the workspace. Therefore no Phase-5 accuracy
number or camera-uncertainty calibration claim is made. The planned parameter
sweeps and held-out non-regression gate remain pending real data.
