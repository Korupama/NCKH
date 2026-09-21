# QA — Stage 4 v0.5.0

## Test suite

```text
PYTHONPATH=. pytest -q tests tests_v04 tests_v05
39 passed
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
