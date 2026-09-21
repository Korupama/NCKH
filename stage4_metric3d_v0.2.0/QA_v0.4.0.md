# Stage 4 v0.4 QA

## Build status

Package version: `0.4.0`

Production candidate: `field-converter-tcn`

Research accuracy frozen: **NO**.

## Automated tests

```text
python -m pytest -q
35 passed
```

The suite includes all retained legacy Stage-4 tests plus new v0.4 tests for:

- Stage-1 → Field Converter camera conversion;
- full-distortion vs k1/k2 projection compatibility;
- strict SAM3D 25-joint cache validation;
- official Field Converter raw inference export;
- source-world prediction import and source-frame remapping;
- downstream handoff;
- v0.4 preflight;
- retained fixed-height v0.3 baseline.

## Adapter validator

```text
python validate_stage4_v040.py
status = PASS
```

Synthetic adapter path validated:

- Stage-3/Stage-1 contracts;
- `(T,N,4)` boxes;
- `(T,N,25,2)` SAM2D;
- `(T,N,25,3)` SAM relative 3D;
- `(K,R,t,k)` camera archive;
- source-world Field Converter prediction import;
- `world-grounded-pose-state-1.0`;
- `stage4-downstream-handoff-2.0`.

## Real supplied-artifact preflight

Inputs used:

- supplied Stage-3 `stage3_real_1920/tracked_pose_2d_state.json`;
- supplied Stage-1 optimized camera timeline;
- selected frame 86;
- source FPS 30;
- image 1920×1080.

Results:

```text
adapter_ready                         true
camera_projection_compatibility       PASS
projection compatibility P95          4.07e-12 px
camera_domain                         OUT_OF_DOMAIN
median camera center                  (-2.221, -68.163, 9.662) m
max |training z-score| after align    19.44
```

The near-zero camera projection difference means the supplied Stage-1 camera currently maps cleanly to Field Converter's `K,R,t,k1,k2` camera implementation; it is not a pose-accuracy metric.

The `OUT_OF_DOMAIN` result is important. The user's camera is physically outside the camera-center distribution documented for the pretrained Field Converter model, mainly along the X coordinate and also camera height. No 180-degree origin-preserving alignment fixes that placement. Inference is still allowed, but paper-level pretrained accuracy must not be assumed.

## Not validated yet

The following require external pretrained assets that were not included in the supplied Stage-4 artifacts:

- official SAM 3D Body inference;
- exact 25-joint SAM3D exporter/semantic mapping used by Field Converter;
- official Field Converter-TCN checkpoint/config/stats inference;
- metric GT accuracy;
- downstream Law-11 body extent / second-last-opponent / final offside-position accuracy.

The implementation therefore distinguishes:

```text
implementation_gate       adapter-tested
geometric_quality_gate    evaluated only after real pretrained predictions
metric_accuracy_gate      NOT_EVALUATED
downstream_offside_gate   NOT_EVALUATED
research_accuracy_frozen  false
```
