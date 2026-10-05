# Phase 9 — Final Stage-3 integration and evaluation

## Decision

**PASS with explicit research limits.** RTMW-L remains the production model;
the Stage-2 contract smoke passes, the shot-level internal holdout is frozen,
and the full regression suite passes. No production fallback re-inference was
claimed because the legacy handoff has no accessible source video.

## Regression and contract

```text
pytest: 59 passed
selected frame: 86
candidate tracks: 10
pose coverage at t0: 1.0
accepted pose coverage at t0: 1.0
valid pose coverage at t0: 1.0
foot pose coverage at t0: 1.0
production fallback validation: DEFERRED
```

The final acceptance artifact is
`runs/phase8_final_acceptance.json`. It records `PASS_WITH_RESEARCH_LIMITS`:
the Stage-2 read-only contract, crop/ownership QA, temporal raw-coordinate
invariant, sequence-disjoint split, 3DSP non-regression, frozen RTMW-L
rollback and full preannotation/checkpoint integrity gates pass. The
pretrained 3DSP accuracy benchmark, frozen RTMW-L rollback and full
preannotation/checkpoint integrity gates pass. Human-review/fine-tuning is not
part of the pretrained-only release path. Broadcast wrong-person and
hallucinated-`VALID` precision remain unmeasured because the available audit
has no independent WholeBody133 target.

See [`docs/STAGE3_BENCHMARK_REPORT.md`](STAGE3_BENCHMARK_REPORT.md) for the
development/holdout results.

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
PDJ: 0.903214
AUC: 0.669479
mean normalized error: 0.214582
median normalized error: 0.115142
valid joint observations: 11200
```

Current final holdout artifact:

```text
benchmark_results/3dsp_phase_final_holdout_rtmw_l/3dsp_benchmark_summary.json
```

The rerun uses the same 40-shot manifest and frozen RTMW-L model as the
reference. It is non-regressed versus the prior artifact (PDJ `0.903125`, AUC
`0.669442`, mean error `0.214667`).

This is an **internal shot holdout**, not a fully untouched public test:
aggregate train results were observed during earlier phases. It is not a
generalization claim.

The matching development split contains 160 shots / 3,200 images and reaches
PDJ `0.910022`, AUC `0.665081`, mean normalized error `0.214363` and median
normalized error `0.118684`.

## Remaining limits

- 3DSP public test images have no posture JSON in the local inventory.
- COCO-WholeBody official `xtcocotools` evaluation was not run in the Windows
  environment; no substitute evaluator was used.
- Production fallback was not validated because the source video is not
  accessible from the immutable legacy handoff.
- No broadcast wrong-person/hallucination precision benchmark is available in
  the current artifacts.

## Production decision

Keep the current RTMW-L Stage-2 cache reuse and Stage-3 QA path. Do not promote
RTMW-X or pseudo-label training based on the available evidence.
