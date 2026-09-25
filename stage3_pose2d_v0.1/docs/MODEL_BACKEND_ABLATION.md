# RTMW-X backend ablation

## Decision

**KEEP RTMW-L as production default; DEFER RTMW-X promotion.**

RTMW-X was successfully exported, validated and evaluated on the same 3DSP
protocol as RTMW-L. It did not improve the full-split result, so it remains an
offline comparative backend. Stage 2, the Stage-3 cache contract and the
production RTMW-L path are unchanged.

## Review record

| Field | Result |
|---|---|
| phase_id | `phase7_model_backend_ablation` |
| review_date | `2026-09-26` |
| decision | `KEEP RTMW-L; DEFER RTMW-X promotion` |
| Stage-3 regression | `32 passed` |
| evaluation environment | `nckh-env`, Python `3.12.14`, PyTorch `2.14.0+cpu`, OpenCV `5.0.0`, NumPy `2.5.3` |
| production default | unchanged: Stage-2 RTMW cache; optional RTMW-L re-inference remains opt-in |
| Stage 2 | read-only; no Stage-2 files changed |

## RTMW-X provenance

Official MMPose model/config:

```text
Model: RTMW-X cocktail14 / u-coco, 384x288
Config: rtmw-x_8xb320-270e_cocktail14-384x288.py
Checkpoint: rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth
```

Checkpoint SHA256:

```text
f840f2044fe46cb3821b7cea86be83e1f6cba406ccd28f5475ac010412dcda95
```

Exported ONNX SHA256:

```text
b06f61538c14ead710799aff8254becbbfadc2732d7b1384dac3edd82212e346
```

The exported graph was checked with ONNX, OpenCV DNN and ONNX Runtime. It has
one input `[batch, 3, 384, 288]` and SimCC outputs:

```text
simcc_x: [batch, 133, 576]
simcc_y: [batch, 133, 768]
```

The temporary export harness used the official MMPose model definition and
exposed the tensor-only forward path. It was experiment-only and is not kept
in the Stage-3 production source tree.

## Same-protocol 3DSP results

All runs use the train split, full-image bbox, bbox padding `1.25`, input
`288x384`, CPU OpenCV DNN, the same H36M17 adapter and the same PDJ/AUC code.

| Run | Samples | Backend | PDJ | AUC | Mean normalized error |
|---|---:|---|---:|---:|---:|
| `3dsp_baseline_full` | 4,000 | RTMW-L | 0.908607 | 0.665931 | 0.214465 |
| `3dsp_phase7_rtmw_l_control_50` | 50 | RTMW-L | 0.954286 | 0.690200 | 0.173213 |
| `3dsp_phase7_rtmw_x_50` | 50 | RTMW-X | 0.904286 | 0.640707 | 0.226569 |
| `3dsp_phase7_rtmw_x_full` | 4,000 | RTMW-X | 0.902571 | 0.655666 | 0.222620 |

Full-split delta versus RTMW-L:

```text
PDJ:  -0.006036
AUC:  -0.010265
Mean error: +0.008155
```

RTMW-X is close but not better on this benchmark. The result does not justify
switching the production model or adding a model-selection branch.

## Implementation scope

- RTMW-X was evaluated temporarily through the existing RTMW SimCC OpenCV
  runner because it has the same WholeBody133/SimCC tensor contract.
- The RTMW-X selector, export harness and export-only dependencies were
  removed after the comparison; they are not part of the production code.
- No Stage-2 files, handoff fixtures or production defaults were changed.

## Dependency note

The temporary export environment used MMPose 1.3.2, MMEngine 0.10.7,
MMCV-lite 2.1.0, MMDetection 3.3.0, MMDeploy 1.3.1, ONNX 1.17.0,
ONNX Runtime 1.30.0 and onnxscript 0.7.2. These export-only dependencies
are not required by the restored RTMW-L Stage-3 runtime. `xtcocotools`
remains optional because its legacy Windows/Python 3.12 build failed.

## Re-open criteria

RTMW-X may be reconsidered only with a new failure slice or deployment reason
that is not contradicted by the full 3DSP result. Any promotion would require
an independent hold-out comparison and production coverage/latency validation.
