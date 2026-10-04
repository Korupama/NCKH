# QA — Stage 4 v0.5.0

## Test suite

```text
PYTHONPATH=. pytest -q tests tests_v04 tests_v05
Historical package check: 39 passed
```

## Synthetic validator

`validate_stage4_v050.py` returns PASS.

Key synthetic checks:

- MHR70 cache/schema round-trip;
- exact 23-joint semantic mapping;
- SAM3D `J_rel + pred_cam_t` camera convention;
- foot/pitch root initialization;
- root-only temporal optimizer;
- full state and downstream handoff generation;
- geometric sanity gate.

Observed controlled-case results:

```text
camera-convention P95       ≈ 2.66e-05 px
SAM direct translation err  ≈ 0.829 m
refined translation err     ≈ 0.00985 m
geometric quality gate      PASS_SANITY
```

These are deterministic synthetic implementation tests, not claims of real-world model accuracy.

Current Phase-9 suite: `91 passed` across `tests`, `tests_v04` and `tests_v05`
in `nckh-env`. The provenance audit, headless visualization test and
model-only ground-consensus regression tests are included in this count.

Phase-4 model-only ground/contact checks include single-foot rejection,
mismatched-foot rejection, multi-candidate consensus, explicit ankle fallback,
invalid pitch/camera fail-closed behavior and bounded root correction. These
tests establish implementation semantics only; they are not metric accuracy.

Phase-5 checks additionally cover exact consecutive-frame triplets, gap and
invalid-camera rejection, optimizer exception fallback, non-finite config
rejection and explicit active-bound/reprojection diagnostics.

Phase-6–8 checks cover tensor/mask consistency, explicit uncertainty
semantics, cache provenance mismatch, output tampering and cache round-trip.

## Real supplied inputs

A worker manifest was generated from the actual Stage-3 artifact and Stage-1 camera timeline:

```text
selected frame       86
fps                  30
resolution           1920×1080
window               56..116
tracks               13
bbox coverage        695 / 793 = 0.8764
camera states        213
```

No real SAM3D inference was executed in this environment, so real geometry/metric accuracy remains unevaluated.

## Phase 0 provenance audit

Run:

```powershell
$env:PYTHONPATH = "."
python tools/stage4_phase0_audit.py --tests-passed 50 --tests-failed 0
```

The audit writes `docs/STAGE4_PHASE0_BASELINE.json` and
`docs/STAGE4_PHASE0_BASELINE_REPORT.md`. Missing real inputs are reported as
`NOT_AVAILABLE_MISSING_INPUTS`; the audit never fabricates input, reruns Stage
1/3, or treats self-consistency residuals as metric ground truth.
