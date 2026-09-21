# Stage 2 benchmark — SoccerNet-GSR v1.3

This harness evaluates the Stage-2 SST–RTMW front-end in **image space** and with the same target-frame-centred semantics used by the offside pipeline.

It deliberately does **not** report GS-HOTA as the primary Stage-2 metric because Stage 2 does not own team, jersey, pitch position, or world geometry.

## 1. Dataset

Expected layout:

```text
D:/Datasets/SoccerNetGS/
  valid/
    SNGS-.../
      Labels-GameState.json
      ... image frames or an mp4 ...
  train/
  test/
  challenge/
```

Use SoccerNet-GSR **v1.3 or later**. The loader checks `info.version` and rejects an older/missing version unless `--allow-old-version` is explicitly used.

Official SoccerNet download (after `pip install SoccerNet` and using your authorized access):

```python
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory=r"D:\Datasets\SoccerNetGS")
d.downloadDataTask(task="gamestate-2024", split=["valid"])
```

Then extract `gamestate-2024/valid.zip` so the final layout is `D:\Datasets\SoccerNetGS\valid\...`. Keep credentials outside source/notebooks; the benchmark harness itself never stores or requests them.

The reader uses only:

- image-space human bounding boxes;
- `track_id`;
- `attributes.role`;
- image/frame metadata.

Team, jersey and pitch coordinates are intentionally ignored.

## 2. Inspect before inference

```powershell
python benchmark_stage2.py inspect --soccernet-root "D:\Datasets\SoccerNetGS" --split valid
```

This checks the split, JSON schema/version, frame count, FPS/resolution, role counts, and whether image/video sources can be resolved.

Do not begin a long GPU benchmark until `inspect` looks correct.

You can also preview the exact deterministic target frames and how many unique frames will require perception inference:

```powershell
python benchmark_stage2.py plan --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick
```

`plan` does not load SST/RTMW. Use it to estimate the frame budget before a GPU run.

## 3. Quick benchmark

Quick protocol:

- first 10 validation sequences unless overridden;
- 3 deterministic target frames per sequence near 25%, 50%, 75% of the clip;
- ±1 second target window;
- each required frame is inferred at most once per sequence;
- target-window tracking is then evaluated on the cached perception.

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --sst-device cuda --rtmw-device cpu --output-dir ".\benchmark_results\quick"
```

To debug one sequence first:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --sequence-id "SNGS-021" --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --output-dir ".\benchmark_results\one_sequence"
```

## 4. Reuse the perception cache

After a first run, the expensive output is stored under:

```text
benchmark_results/<run>/sequences/<sequence>/perception/
```

If the cache contains all frames required by the requested protocol, model paths can be omitted:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --output-dir ".\benchmark_results\quick"
```

Switching from `quick` to `full` normally requires additional frames. In that case pass model paths again; existing frame-perception JSONs are reused and only missing frames are inferred unless `--overwrite-perception` is specified.

## 5. Full validation benchmark

Full protocol:

- every sequence in the selected split;
- 5 deterministic target frames per sequence near 1/6, 1/3, 1/2, 2/3, 5/6;
- ±1 second target window.

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol full --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --sst-device cuda --output-dir ".\benchmark_results\full_valid"
```

Use validation for threshold/model selection. Do not repeatedly tune on test.

## 6. What is evaluated

### Selected-frame entity/detection

- human precision / recall @ IoU 0.5;
- `CandidateRecall` for Player + Goalkeeper;
- `RefereeLeakageRate`;
- cross-class duplicate rate before consolidation;
- post-consolidation duplicate rate;
- track-level role precision/recall/F1 and confusion matrix;
- human `AP50` and `mAP50:95`, plus per-role AP.

### t0-centric tracking

For each window, GT identities are **anchored at t0**. A GT identity that is absent at t0 is not added later as a required track. This matches the Stage-2 decision-frame-anchored tracker rather than conventional full-clip MOT.

Two views are reported:

- `tracking_human`: all Stage-2 human roles;
- `tracking_candidate`: Player + Goalkeeper only (primary downstream metric).

Metrics:

- HOTA;
- DetA;
- AssA;
- LocA;
- IDF1 / IDR / IDP;
- ID-switch and fragmentation diagnostics;
- `TCR@1s`.

The self-contained HOTA and Identity implementations follow the public TrackEval reference equations in image-IoU space. Before publication, cross-check a frozen run with official TrackEval / SoccerNet sn-trackeval.

### TCR@1s

At t0, Hungarian IoU matching establishes GT↔predicted track pairs. Every GT candidate visible at t0 contributes to overall TCR: a missing prediction anchor contributes 0. For matched anchors, per-track TCR is the fraction of visible evaluation frames within the ±1 s window for which the same predicted track still overlaps that GT by IoU >= 0.5. `AnchorCoverage` reports how many t0 GT candidates receive an anchor, while `ConditionalTCR` averages continuity only over matched anchors.

## 7. Built-in ablations

Default (cache-reusable, no extra model inference):

```text
primary
no_pose
oracle_boxes_geom
```

`primary` uses SST + physical-human consolidation + RTMW tracking cue.

`no_pose` uses the **same SST/consolidation cache** but removes RTMW pose cues before association. Comparing `primary` vs `no_pose` measures the value of RTMW for Stage-2 tracking without rerunning SST.

`oracle_boxes_geom` replaces detections with GT image boxes and GT roles but deliberately does not expose GT identity to the tracker and does not provide pose. It estimates how well the current geometric association behaves when detection is perfect. It is an association diagnostic, not an end-to-end result.

`oracle_boxes_pose` is the stronger detector-oracle ablation proposed for the research analysis: it uses GT image boxes + GT roles, runs the same RTMW-L pose model on those GT boxes, and still never exposes GT track identity to association. This isolates the loss caused by SST/consolidation from the loss caused by the RTMW-assisted tracker. Because it requires fresh RTMW inference, it is optional rather than part of the cache-only default.

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --rtmw-model ".\weights\rtmw_l_384x288.onnx" --output-dir ".\benchmark_results\quick" --experiments primary,no_pose,oracle_boxes_geom,oracle_boxes_pose
```

Disable ablations to make tracker evaluation faster:

```powershell
--experiments primary
```

## 8. Outputs

```text
benchmark_results/<run>/
  benchmark_summary.json
  benchmark_summary.csv
  window_metrics.csv
  protocol_windows.json
  detection_ap.json
  role_confusion_matrix.csv
  role_confusion_matrix.png        # if matplotlib is available
  failure_cases.json
  failure_cases/*.png              # up to --failure-images N
  benchmark_manifest.json
  sequences/
    <sequence>/
      frames/                       # only when decoding from video is required
      perception/
        perception_manifest.json
        frame_perception/*.json
```

`benchmark_summary.csv` is the compact table for the thesis/report. `window_metrics.csv` is for distributions/error analysis. `failure_cases.json` records every window with incomplete CandidateRecall, referee leakage, or TCR < 0.90.

## 9. Initial acceptance gate

### Hard Stage-3 handoff gates

| Metric | Minimum |
|---|---:|
| CandidatePrecision | >= 0.95 |
| CandidateRecall | >= 0.95 |
| Referee leakage | <= 0.02 |
| AnyDuplicateRate after consolidation | < 0.02 |
| Candidate HOTA | >= 0.60 |
| Candidate AssA | >= 0.55 |
| Candidate IDF1 | >= 0.60 |
| TCR@1s | >= 0.90 |

### Quality checks (reported, not hard blockers)

| Metric | Target |
|---|---:|
| Player recall | >= 0.95 |
| Goalkeeper recall | >= 0.90 |
| Track-level role macro-F1 | >= 0.90 |
| AnchorCoverage | >= 0.95 |
| ConditionalTCR | >= 0.95 |

These are engineering targets, not claimed benchmark results. `acceptance_gate.passed` is based on the hard downstream-oriented checks; exact Player/GK quality remains visible separately.

## 10. Installation

Core project:

```powershell
pip install -e .
```

With confusion-matrix plotting support:

```powershell
pip install -e ".[benchmark]"
```

Regression tests:

```powershell
pytest -q
```

## 11. Crash-safe checkpoint / resume (v1.2)

Benchmark evaluation is checkpointed by default. You do **not** need a special `--resume`
flag: rerunning the exact same command with the same `--output-dir` resumes automatically.

Two independent checkpoint layers are used:

```text
benchmark_results/<run>/
  sequences/<sequence>/perception/
    perception_checkpoint.json        # updated after every SST+RTMW frame
    perception_manifest.json          # complete-cache manifest
    frame_perception/*.json            # atomic per-frame perception outputs

  checkpoints/
    benchmark_state.json               # global benchmark progress/status
    partial_summary.json               # aggregate of completed windows
    evaluation/
      <sequence>/
        t000001234/
          detection.json               # selected-frame entity checkpoint
          tracking_primary.json        # HOTA/IDF1/TCR checkpoint
          tracking_no_pose.json
          tracking_oracle_boxes_geom.json
          tracking_oracle_boxes_pose.json  # when requested
```

Each JSON checkpoint is written through a temporary file and atomically replaced. If the
terminal, Python process, machine, or GPU job is interrupted, already committed units are
not evaluated again. On restart the runner verifies the deterministic protocol/configuration,
SoccerNet label-file hashes, and available model fingerprints before accepting old checkpoints.

Normal resume:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --output-dir ".\benchmark_results\quick"
```

If this is the same configuration/output directory, the console prints `[RESUME]` for completed
units and continues from the first missing unit.

Inspect saved progress without loading SST/RTMW:

```powershell
python benchmark_stage2.py status --output-dir ".\benchmark_results\quick"
```

Important distinction:

- `--overwrite-perception` deliberately rebuilds expensive SST/RTMW perception caches;
- `--restart-evaluation` deletes **only evaluation checkpoints** and keeps perception caches.

For example, after changing only tracker parameters or fixing evaluator code:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --output-dir ".\benchmark_results\quick" --restart-evaluation
```

If the new tracker parameters differ from the saved run signature, the runner refuses to mix old
metric checkpoints and tells you to use `--restart-evaluation` or a new output directory.

## 12. Live console progress (v1.2)

Evaluation prints one progress line for every committed detection/tracking task. Perception prints
at the first/last frame and every 10 frames by default. Example:

```text
[18:41:03] [SEQ 2/10] SNGS-021 version=1.3 fps=25 windows=3 inference_frames=153
[18:41:25] [PERCEPTION SNGS-021 30/153 19.6%] frame=224 infer humans=18 pose=16 ball=1 | elapsed 00:00:22 ETA 00:01:31
[18:43:11] [EVAL 14/120 11.7%] SNGS-021 t0=375 tracking:primary | HOTA=0.681 AssA=0.712 IDF1=0.735 TCR=0.944 | elapsed 00:02:08 ETA 00:11:34 | CHECKPOINT
[18:43:11] [WINDOW CHECKPOINT] SNGS-021 t0=375 fully committed
```

Change the perception reporting frequency:

```powershell
--progress-every 5
```

Suppress live progress (useful for redirected logs/CI):

```powershell
--quiet
```

A clean `Ctrl+C` marks `benchmark_state.json` as `INTERRUPTED`, writes `partial_summary.json`,
and exits with code 130. A hard process/machine failure can at worst lose the single unit that had
not yet reached its atomic checkpoint.

## 12. v1.3 / M1 evaluator and cache changes

A v1.3 perception run stores `raw_low_score_detections` down to `--raw-human-score-floor` and `rescue_humans`. Because v1.2 caches do not contain this information, they are not accepted as v1.3 M1 production caches.

New selected-frame metrics:

- `CandidatePrecision` and `CandidateRecall` in candidate-only matching space;
- `AnyDuplicateRate_before/after`;
- `CrossClassConflictRate_before/after`.

New tracking diagnostics:

- `AnchorCoverage` — GT footballers at t0 that obtain a prediction anchor;
- `ConditionalTCR` — continuity only among anchors that exist;
- `TCR` — overall continuity including missing anchors.

The v1.3 hard gate is intentionally downstream-oriented: CandidatePrecision/Recall, referee leakage, residual duplicate humans, HOTA, AssA, IDF1 and overall TCR. Exact Player/GK role recall/F1 and AC/cTCR remain reported as quality checks.

### Cache-only threshold sweep

After a v1.3 quick run:

```powershell
python benchmark_stage2.py sweep-thresholds --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --benchmark-dir ".\benchmark_results\quick_v13" --player-thresholds "0.30,0.35,0.40,0.45,0.50" --goalkeeper-thresholds "0.30,0.35,0.40,0.45,0.50"
```

The sweep reuses only cached low-score SST detections at each t0 and therefore does not load SST or RTMW. It is a **selected-frame detector/role sweep**, not a tracking benchmark. After choosing a threshold pair, rerun the quick protocol with `--player-threshold` / `--goalkeeper-threshold` so RTMW and tracking reflect the promoted candidates.

Example:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --player-threshold 0.40 --goalkeeper-threshold 0.40 --raw-human-score-floor 0.20 --output-dir ".\benchmark_results\quick_v13_p040_g040"
```
