# Stage 4 Phase 8 — model-only cache integrity

Date: 2026-10-04  
Scope: Stage 4 model-only inference lane

## Implemented

The model-only runner now creates a content-addressed cache contract covering:

- benchmark manifest SHA256 and every record's image/camera/GT hashes;
- dataset and split;
- SAM3D checkpoint and MHR model hashes;
- SAM3D repository revision when available;
- requested/resolved device and inference type;
- output schema and Stage-4 source-file hashes.

The sidecar is written atomically after the output `.npz` is finalized. With
`--resume`, the runner verifies the sidecar and output hash, then re-evaluates
the output before returning a cache hit. Any mismatch fails closed.

## Tests

```text
cache/evaluator/systems tests: 12 passed
```

## Limitations

No real model throughput/memory profile is recorded because the pretrained
checkpoint/runtime is not mounted. Cache integrity does not establish model
accuracy.
