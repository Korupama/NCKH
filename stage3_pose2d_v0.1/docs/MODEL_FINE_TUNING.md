# Phase 8 — Football WholeBody133 fine-tuning readiness

## Current status

**ANNOTATION_READY — not TRAIN_READY.**

## Phase 8A — Pseudo-label self-training

The frozen RTMW-L teacher produced a train-only pseudo-label artifact:

| Field | Result |
|---|---:|
| train tasks processed | 2,986 |
| pseudo-labels accepted by Stage-3 QA | 2,936 |
| pseudo-labels rejected | 50 |
| validation/test pseudo-labels | 0 |
| teacher | RTMW-L |
| teacher SHA256 | `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054` |

The COCO-WholeBody-formatted artifact is
`runs/phase8_pseudo_coco_rtmw_l_train.json`. It is explicitly marked
`PSEUDO_LABEL`, `training_only=true` and `is_ground_truth=false`.

This is ready as input for an exploratory student-training job. It does not
complete Phase 8: no student checkpoint has been trained and no independent
human-ground-truth validation/test accuracy claim exists.

The workspace now has a reproducible annotation-task workflow using the
authorized local SoccerNet-GSR validation images and sequence-level splits.
Training must wait until tasks are human-annotated and independently reviewed
as `COCO_WHOLEBODY_133`.

## Prepared task set

The task generator created 4,286 player/goalkeeper annotation tasks from 58
SoccerNet-GSR sequences:

| Split | Sequences | Tasks |
|---|---:|---:|
| train | 42 | 2,986 |
| validation | 9 | 682 |
| test | 8 | 618 |

The fixed sequence split is stored in
[`splits/phase8_gsr_valid_seed20260926.json`](../splits/phase8_gsr_valid_seed20260926.json).

The generated task manifest is local under `runs/` and is intentionally not
committed with image data. Each task contains the source image, GSR bbox,
sequence/track identity, 133 empty keypoint slots and:

```text
annotation_source = UNANNOTATED
review_status = PENDING
```

RTMW-L may assist an annotator as a pre-annotation, but its predictions must
not be promoted automatically to ground truth.

Generate a separate pre-annotation artifact without modifying the empty ground
truth fields:

```powershell
python tools/preannotate_tasks.py `
  --manifest ".\runs\phase8_annotation_tasks_gsr_valid.json" `
  --rtmw-model "D:\GitHub\NCKH\datasets\stage3_assets\rtmw_l_384x288.onnx" `
  --output ".\runs\phase8_preannotated_full.json" `
  --checkpoint ".\runs\phase8_preannotation.checkpoint" `
  --chunk-size 32
```

The output is explicitly marked `MODEL_PREANNOTATION_ONLY` and remains
ineligible for training until human review replaces the empty ground-truth
fields. The checkpoint stores manifest/model/config hashes and immutable task
chunks. If a long CPU run is interrupted, continue it with the same inputs:

```powershell
python tools/preannotate_tasks.py `
  --manifest ".\runs\phase8_annotation_tasks_gsr_valid.json" `
  --rtmw-model "D:\GitHub\NCKH\datasets\stage3_assets\rtmw_l_384x288.onnx" `
  --output ".\runs\phase8_preannotated_full.json" `
  --checkpoint ".\runs\phase8_preannotation.checkpoint" `
  --resume
```

Resume is rejected if the source manifest, model bytes, Stage-3 config, task
order or ground-truth fields differ from the checkpoint provenance. Completed
tasks are not inferred from the output JSON; they are recovered only from
validated checkpoint chunks.

The real RTMW-L smoke artifact `runs/phase8_preannotated_checkpoint_smoke2.json`
processed two tasks, resumed for two additional tasks, and preserved empty
human-label fields for all four tasks. It is only a checkpoint/reproducibility
check, not evidence of reduced hallucination.

## Human review gate

Before training, every selected task must contain:

- exactly 133 points in canonical COCO-WholeBody order;
- human-reviewed coordinates and visibility values;
- `annotation_source = HUMAN_VERIFIED`;
- `review_status = APPROVED`;
- reviewer ID and review timestamp;
- source/rights record for the SoccerNet-GSR images and annotations.

Start the local reviewer against a separate output manifest. It preserves the
source manifest and keeps model predictions in `model_preannotation_133`:

```powershell
python tools/annotation_reviewer.py `
  --manifest ".\runs\phase8_annotation_tasks_gsr_valid.json" `
  --preannotations ".\runs\phase8_preannotated_full.json" `
  --output ".\runs\phase8_reviewed_annotations.json"
```

Run the validator after annotation:

```powershell
python tools/validate_annotations.py `
  --annotations ".\runs\phase8_reviewed_annotations.json" `
  --output ".\runs\phase8_human_validation.json"
```

The validator must return `PASS` before any training command is added.
`--allow-pending` only checks task structure and cannot authorize training.

After validation, export each split independently. The exporter refuses
pending, pseudo-label, incomplete or non-approved tasks:

```powershell
python tools/export_human_coco.py --manifest ".\runs\phase8_reviewed_annotations.json" --split train --output ".\runs\phase8_human_coco_train.json"
python tools/export_human_coco.py --manifest ".\runs\phase8_reviewed_annotations.json" --split validation --output ".\runs\phase8_human_coco_validation.json"
python tools/export_human_coco.py --manifest ".\runs\phase8_reviewed_annotations.json" --split test --output ".\runs\phase8_human_coco_test.json"
```

Only the exported human-reviewed files may become fine-tuning inputs. The
reviewer and exporter do not train or promote a model.

## Existing data that is not ground truth

- 3DSP supplies H36M17-style pose annotations, not WholeBody133.
- SoccerNet-GSR supplies boxes, roles, tracks and pitch metadata, not pose
  keypoints.
- RTMW predictions are model outputs, not annotation ground truth.

No pseudo-label or 17-to-133 silent mapping is permitted.

Pseudo-labels are permitted only for Phase 8A's train split, with teacher hash,
QA filter and `is_ground_truth=false` provenance. They must never be used as
the final validation/test target.
