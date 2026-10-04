# Stage 4 Phases 6–7 — uncertainty and semantic safety

Date: 2026-10-04  
Scope: Stage 4 model-only evaluation/output contract

## Implemented

- `valid_mask=true` requires all native and Pose23 output tensors to be finite.
- `valid_mask=false` must contain no partial finite tensor output.
- Per-record status distinguishes `VALID`, `DEGRADED`, `REJECTED`, `MISSING`
  and `NOT_EVALUATED` with a reason.
- Ground ambiguity/unavailability is retained as finite SAM-prior output but is
  classified as `DEGRADED`, not as supported ground truth.
- Uncertainty is explicitly reported as `NOT_CALIBRATED` with scope
  `SENSITIVITY_ONLY`; unavailable components are not filled with assumptions.

## Verification

```text
model-only evaluation tests: 6 passed
```

## Limitations

The workspace still lacks the authorized real model-only dataset, checkpoint
and independent metric GT. Calibration, subgroup coverage and final accuracy
remain `NOT_EVALUATED`. Synthetic fixtures verify semantics only.
