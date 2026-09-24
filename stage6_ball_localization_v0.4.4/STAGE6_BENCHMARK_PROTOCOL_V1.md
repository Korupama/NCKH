# Stage 6 Benchmark Protocol v1.3 — Reduced Offside Pipeline

This protocol evaluates the current Stage 6 implementation as four distinct scientific tasks instead of treating random replay frames as a benchmark.

## 1. Evaluation layers

| Layer | Dataset | Purpose | Claim level |
|---|---|---|---|
| **Stage 6A — 2D ball detection** | SoccerNet-v3D fixed **test** split | Detect the correct ball observation used by geometry | Primary |
| **Stage 6B — longitudinal ball localization** | SoccerNet-v3D fixed **test** split | Estimate ball **X** in canonical pitch coordinates | **Primary** |
| **Stage 6C — temporal / hybrid geometry** | ISSIA-3D | Evaluate consecutive-frame geometry independently of SoccerNet-v3D row ordering | Secondary / ablation |
| **Stage 6D — contact association** | FOOTPASS validation-derived frozen manifest, or a separately frozen project set | Evaluate `contact_track_id` and `contact_body_region` at `t0` | Primary for contact |
| Internal QA | user-selected project replay frames | Debugging, regression and visualization only | **No accuracy claim** |

The current folder name remains `stage6_ball_localization_v0.4.4`, but runtime code reports Stage 6 v0.5.1. Benchmark reports embed runtime provenance to avoid stale-package confusion.

---

## 2. Why SoccerNet-v3D is the primary dataset

The current repository already has a native `SoccerNetV3DCSV` adapter and benchmark code for:

- official 2D ball boxes;
- optimized ball diameter/box geometry;
- per-frame camera calibration;
- triangulated `ball_3D` in metres;
- fixed `train` / `test` flags;
- canonical conversion to Stage-6 `X=goal-to-goal`, `Y=touchline-to-touchline`, `Z=up`.

Therefore no new dataset adapter is required for the primary benchmark.

**Important:** SoccerNet-v3D action/replay rows are synchronized multi-view observations. They are **not consecutive broadcast frames** and must never be fed to Viterbi/temporal geometry in CSV order.

---

## 3. Primary metrics

### Stage 6A — 2D ball detection

| Metric | Role |
|---|---|
| `AP50` | Primary detection quality |
| `mAP50_95` | Primary detection quality |
| `Recall@IoU0.5` | Primary; missed balls directly block Stage 6 |
| `CandidateRecall@5` | Measures whether the correct ball survives top-K candidate generation |
| `CenterErrorPx median/P90` | Important because the image center defines the camera ray |
| Precision | Secondary |

The benchmark uses **optimized GT boxes** with the official optimized SoccerNet-v3D detector weight `yolo-sn-ball-opt.pt`.

### Stage 6B — longitudinal ball localization

The reduced offside pipeline primarily consumes the ball coordinate along the goal-to-goal axis:

\[
E_X = |\hat X_b-X_b^{GT}|.
\]

Report:

| Metric | Role |
|---|---|
| `Ball-X MAE` | **Primary** |
| `Ball-X median` | Primary |
| `Ball-X P90` | Tail error |
| `Ball-X P95` | Tail error |
| localization coverage | Primary |
| 3D MAE / P95 | Secondary diagnostic only |
| Y/Z errors | Secondary diagnostic only |

The existing code names these values `BLE_X_*`; protocol v1 reports them semantically as **Ball-X** without breaking the historical evaluator.

---

## 4. Required oracle vs end-to-end decomposition

Run both:

| Experiment | Ball box | Camera | Purpose |
|---|---|---|---|
| **Oracle geometry** | GT optimized box | SoccerNet-v3D calibration | Geometry/size-prior ceiling; detector error excluded |
| **End-to-end top1** | predicted top-1 box | SoccerNet-v3D calibration | Detector → metric-X performance |

The difference

\[
\Delta E_X = E_X^{E2E}-E_X^{Oracle}
\]

is reported as an error-budget diagnostic.

`best-iou` selection is not a production benchmark because it uses GT to choose the detector candidate. Protocol v1 freezes **top1** for end-to-end evaluation.

---


## 4.1 Center-vs-diameter geometry diagnostics (v1.1)

After the primary benchmark has produced a complete `01_detector_2d` cache, run the
diagnostic decomposition without rerunning YOLO.  Four observations are synthesized for
each SoccerNet-v3D row:

| Variant | Center | Diameter | Interpretation |
|---|---|---|---|
| `GT_CENTER_GT_DIAMETER` | GT | GT optimized | single-frame geometry oracle |
| `PRED_CENTER_GT_DIAMETER` | predicted top-1 | GT optimized | center/selection contribution |
| `GT_CENTER_PRED_DIAMETER` | GT | predicted top-1 | apparent-size contribution |
| `PRED_CENTER_PRED_DIAMETER` | predicted top-1 | predicted top-1 | normal top-1 E2E geometry |

The report contains three views: all evaluable rows, the IoU>=0.5 top-1 matched subset,
and the intersection of frames where all four variants yield valid geometry.  The
**common-frame** view is the preferred error-attribution comparison because the sample
set is identical across variants.

MAE differences are diagnostic and are **not** assumed to be additive causal effects.
A non-additive interaction residual is reported explicitly.

The same command also runs the existing `best-iou` E2E mode.  This is a GT-assisted
upper bound on candidate ranking and remains `production_metric=false`.

## 5. Temporal benchmark — ISSIA-3D only

The repository already provides ISSIA-3D temporal evaluation and enforces the published camera split:

| Split | Cameras | Usage |
|---|---|---|
| development/tuning | **3, 4, 5, 6** | hybrid selector calibration |
| held-out test | **1, 2** | final temporal/cross-domain evaluation |

Never tune hybrid thresholds on cameras 1–2.

Protocol v1.3 exposes this as two explicit commands: `issia-calibrate` then `issia-test`.

---

## 6. Contact benchmark

The Stage 6 v0.5.1 contact code is a conservative 2D-pose heuristic, not a learned classifier, and its thresholds are currently uncalibrated. A real benchmark must therefore use independent contact GT.

FOOTPASS is suitable for producing contact cases because validation events contain frame/team/jersey/action identities and player track information. However, Stage 6 outputs its own `track_id`, so a deterministic dataset-specific mapping is required before scoring.

To avoid coupling Stage 6 to one private local directory layout, protocol v1.1 uses a canonical frozen manifest:

```json
{
  "metadata": {
    "name": "stage6-contact-eval",
    "dataset": "FOOTPASS",
    "split": "validation",
    "frozen": true
  },
  "cases": [
    {
      "case_id": "game18_H1_001234",
      "action_class": "Pass",
      "state_json": "states/game18_H1_001234/ball_trajectory_state.json",
      "gt_track_id": "track_006",
      "gt_region": "FOOT",
      "gt_x_m": null
    }
  ]
}
```

Report:

- contact-track accuracy;
- contact-assignment coverage;
- contact-region accuracy when region GT exists;
- optional Ball-X MAE/median/P90/P95 when metric GT exists;
- per-action contact accuracy.

Missing Stage-6 assignment counts as incorrect, not as an ignored case.

The FOOTPASS challenge/test GT is hidden, so local contact scoring should use the validation split or a separately frozen project test set. The official FOOTPASS ±12-frame action-spotting F1 is **not** reused as the Stage-6 contact metric because the project receives a manually selected `t0`.

---

## 7. Commands

### Print protocol

```powershell
python benchmark_stage6_protocol.py protocol
```

### Primary full SoccerNet-v3D benchmark

```powershell
python benchmark_stage6_protocol.py primary `
  --csv "D:\datasets\SoccerNet-v3D\SNv3D.csv" `
  --image-root "D:\SoccerNet" `
  --weights ".\weights\yolo-sn-ball-opt.pt" `
  --output-dir ".\outputs\benchmark_stage6_primary" `
  --split test `
  --device 0
```

Outputs:

```text
benchmark_stage6_primary/
├── 01_detector_2d/
├── 02_oracle_geometry/
├── 03_e2e_top1/
└── 04_protocol_report/
    ├── stage6_primary_benchmark.json
    └── stage6_primary_benchmark.md
```

### Smoke test only

```powershell
python benchmark_stage6_protocol.py primary ... --max-images 30 --max-rows 30
```

Any truncated run is labeled `SMOKE` and `scientific_claim_ready=false`.


### Center/diameter decomposition + best-IoU diagnostic

Reuse the completed detector cache from the primary run:

```powershell
python benchmark_stage6_protocol.py geometry-diagnostics `
  --csv "D:\datasets\SoccerNet-v3D\SNv3D.csv" `
  --benchmark-2d-dir ".\outputs\benchmark_stage6_primary\01_detector_2d" `
  --primary-report ".\outputs\benchmark_stage6_primary\04_protocol_report\stage6_primary_benchmark.json" `
  --output-dir ".\outputs\benchmark_stage6_geometry_diagnostics" `
  --split test
```

Outputs:

```text
benchmark_stage6_geometry_diagnostics/
├── 01_center_diameter_decomposition/
│   ├── benchmark_3d_decomposition.json
│   ├── benchmark_3d_decomposition.md
│   └── frame_metrics.csv
├── 02_e2e_best_iou/
└── 03_report/
    ├── stage6_geometry_diagnostics.json
    └── stage6_geometry_diagnostics.md
```

The optional `--primary-report` check verifies that `GT+GT` and `Pred+Pred` reproduce
the already-published primary oracle/top-1 MAE under identical geometry settings.

### ISSIA-3D development and held-out test

```powershell
python benchmark_stage6_protocol.py issia-calibrate `
  --csv ".\assets\ISSIA-3D.csv" `
  --calibration ".\assets\issia_calibration.json" `
  --output-dir ".\outputs\issia_dev"

python benchmark_stage6_protocol.py issia-test `
  --csv ".\assets\ISSIA-3D.csv" `
  --calibration ".\assets\issia_calibration.json" `
  --hybrid-config-json ".\outputs\issia_dev\hybrid_calibration.json" `
  --output-dir ".\outputs\issia_test"
```

### Contact manifest

```powershell
python benchmark_stage6_protocol.py contact-template `
  --output ".\contact_eval\manifest.json"

python benchmark_stage6_protocol.py contact `
  --manifest ".\contact_eval\manifest.json" `
  --output-dir ".\outputs\contact_benchmark"
```

---

## 8. What does not count as benchmark evidence

The following remain useful but must not be converted into Stage-6 accuracy claims:

- a hand-picked frame such as frame 86/95/104;
- the existing 61-frame replay without independent ball GT;
- visual minimap plausibility;
- synthetic regression alone;
- self-reprojection error of a point constructed on its own camera ray;
- ISSIA cameras 1–2 after tuning on those same cameras;
- `best-iou` detector candidate selection.

They are integration, failure-analysis or engineering QA evidence only.

---

## 9. Acceptance state

Protocol v1.1 intentionally does **not** invent a centimetre threshold for PASS/FAIL. A metric threshold can be frozen only after the full benchmark is run and compared with an established baseline or a justified downstream offside error budget.

Until then, reports distinguish:

- implementation completeness;
- full-vs-smoke dataset coverage;
- scientific-claim readiness;
- measured metrics.


## v1.2 diagnostic correction

The v1.1 center/diameter decomposition reconstructed mixed detector observations as square boxes.
That was acceptable for a rough size sensitivity study, but it changed the exact detector box consumed by
the canonical E2E localizer. In the 810-row test run, Oracle reproduced exactly while Pred+Pred did not
reproduce the primary E2E MAE/coverage. v1.2 therefore freezes a stricter decomposition:

- GT+GT: exact optimized GT box.
- Pred-center + GT-size: translate the optimized GT box to the predicted center.
- GT-center + Pred-size: recenter the exact predicted box on the GT center, preserving width/height/aspect.
- Pred+Pred: use the exact cached top-1 detector box and diameter, unchanged.

The Pred+Pred MAE **and coverage** must match the primary `run_3d_e2e --selection top1` result.
The report also stratifies E2E Ball-X error by absolute relative diameter error.


## v1.3 ground-plane diagnostic for the production FOOT branch

The v1.2 decomposition established that predicted ball box size dominates the additional
single-frame Ball-X error, while predicted center contributes comparatively little.  The
next diagnostic therefore evaluates the geometry used by Stage 6 v0.5.1 when contact is
classified as `FOOT`:

\[
\text{camera ray through ball center} \cap Z=r_{ball}.
\]

SoccerNet-v3D does not label physical contact, so v1.3 does **not** pretend to be a
contact benchmark.  Instead, GT ball height is used only to create near-ground proxy
subsets:

- `|GT_Z - r_ball| <= 0.05 m`;
- `|GT_Z - r_ball| <= 0.10 m` (**primary proxy**);
- `|GT_Z - r_ball| <= 0.20 m`.

Inference never consumes GT XYZ.  On each subset the evaluator compares:

| Variant | Geometry | 2D observation |
|---|---|---|
| `GROUND_PLANE_GT_CENTER` | ray ∩ `Z=r_ball` | optimized GT center |
| `GROUND_PLANE_PRED_CENTER` | ray ∩ `Z=r_ball` | predicted top-1 center |
| `SIZE_PRIOR_GT_BBOX` | current monocular size prior | optimized GT bbox |
| `SIZE_PRIOR_TOP1` | current monocular size prior | exact predicted top-1 bbox |

The common-valid-frame report exposes:

- Ball-X MAE/median/P90/P95 and coverage;
- gain of predicted-center ground-plane geometry over size-prior E2E;
- gain of GT-center ground-plane geometry over size-prior oracle;
- the residual penalty from using predicted rather than GT center on the ground plane.

This remains `production_metric=false` because GT height selects the subset and contact
association is bypassed.  A separate FOOTPASS/frozen-manifest experiment is still required
to measure `contact_track_id` / `contact_body_region` accuracy.

### Ground-plane diagnostic command

```powershell
python benchmark_stage6_protocol.py ground-plane-diagnostics --csv ".\\data\\SNv3D.csv" --benchmark-2d-dir ".\\outputs\\benchmark_stage6_primary\\01_detector_2d" --output-dir ".\\outputs\\benchmark_stage6_ground_plane" --split test
```

No YOLO inference is rerun.  The key output is:

```text
outputs/benchmark_stage6_ground_plane/
├── stage6_ground_plane_diagnostic.json
├── stage6_ground_plane_diagnostic.md
└── frame_metrics.csv
```

The primary result to inspect is:

```text
primary_near_ground_proxy.common_valid_frames
```

If `GROUND_PLANE_PRED_CENTER` materially outperforms `SIZE_PRIOR_TOP1` on the near-ground
proxy while retaining high coverage, the evidence supports demoting the size prior to a
fallback for the project's initial ground-ball scope.  That conclusion still does not
validate contact association itself.

## v1.4 — Remaining TBD evaluators

The remaining Stage-6 TBD rows are now executable without silently manufacturing GT:

- **Ground-plane Ball-X**: `ground-plane-diagnostics` (SoccerNet-v3D near-ground proxy; already introduced in v1.3).
- **Temporal/hybrid Ball-X**: `issia-calibrate` on cameras 3-6, then `issia-test` on held-out cameras 1-2.
- **Contact actor/track**: `contact-production` reports assignment coverage, assigned precision and overall accuracy.
- **FOOT vs NON_FOOT**: `contact-production` reports binary accuracy, FOOT precision, recall and F1; exact region remains secondary.
- **Production Ball-X @ t0**: `contact-production` reports MAE/median/P90/P95/coverage only when the frozen manifest contains independent `gt_x_m`/`gt_xyz_m`.
- **Unified readiness**: `tbd-status` aggregates the three reports and labels research/offside project targets. Missing GT remains `NOT_EVALUATED`.

### FOOTPASS bridge

FOOTPASS validates `(game_key, frame, team, jersey, class)`. Use `footpass-bridge-template` then `footpass-build-manifest` to bind selected official validation events to Stage-6 state files. A direct project `gt_track_id` or a deterministic `track_identity_map_json` is required to turn FOOTPASS actor identity into a project-track evaluation. Only **Header→HEAD** and **Throw-in→ARM_HAND** are derived automatically from action semantics; no Pass/Cross/Shot event is assumed to be FOOT.

FOOTPASS does **not** provide metric ball-X GT, so it cannot by itself validate the full contact-aware Ball-X output. That metric requires an additional frozen source of `gt_x_m`; otherwise the evaluator returns null and does not mark the criterion as passed.
