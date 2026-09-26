# Phase 9 — Final Stage-3 integration and evaluation

## Decision

**PASS with explicit research limits.** RTMW-L remains the production model;
the Stage-2 contract smoke passes, the shot-level internal holdout is frozen,
and the full regression suite passes. No production fallback re-inference was
claimed because the legacy handoff has no accessible source video.

## Regression and contract

```text
pytest: 35 passed
selected frame: 86
candidate tracks: 10
pose coverage at t0: 1.0
accepted pose coverage at t0: 1.0
valid pose coverage at t0: 1.0
foot pose coverage at t0: 1.0
production fallback validation: DEFERRED
```

Final contract output:

```text
runs/stage3_phase9_final_legacy/tracked_pose_2d_state.json
runs/stage3_phase9_final_legacy/stage3_downstream_handoff.json
```

The three Stage-2 handoff SHA256 values were recorded before the run; no
Stage-2 file was modified.

## Internal shot holdout

Manifest:

```text
splits/3dsp_internal_holdout_seed20260926.json
```

It uses a deterministic sequence-level split of the 200 labeled 3DSP train
shots:

```text
development: 160 shots
holdout: 40 shots / 800 images
seed: 20260926
```

RTMW-L holdout result, same full-image bbox/CPU/metric protocol:

```text
PDJ: 0.903125
AUC: 0.669442
mean normalized error: 0.214667
median normalized error: 0.115142
valid joint observations: 11200
```

This is an **internal shot holdout**, not a fully untouched public test:
aggregate train results were observed during earlier phases. It is not a
generalization claim.

## Remaining limits

- 3DSP public test images have no posture JSON in the local inventory.
- COCO-WholeBody official `xtcocotools` evaluation was not run in the Windows
  environment; no substitute evaluator was used.
- Production fallback was not validated because the source video is not
  accessible from the immutable legacy handoff.
- Phase 8 pseudo-label self-training remains exploratory and has no human-
  verified validation/test accuracy claim.

## Production decision

Keep the current RTMW-L Stage-2 cache reuse and Stage-3 QA path. Do not promote
RTMW-X or pseudo-label training based on the available evidence.
