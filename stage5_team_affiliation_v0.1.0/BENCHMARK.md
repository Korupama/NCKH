# Stage 5 third-party benchmark — SoccerNet-GSR v1.3

> Local handoff: [dataset locations and download policy](../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

This benchmark evaluates **team affiliation only** on a third-party dataset. It is deliberately separate from the project replay/visual QA case.

## Protocol

- Dataset: SoccerNet-GSR v1.3.
- Recommended development split: `valid`; keep `test` for the final held-out report if labels/evaluation access permits.
- Evaluation unit: **track**, not frame.
- GT inputs allowed during inference: `bbox_image`, `track_id`, `role`, raw RGB frame.
- GT `team` is hidden until evaluation.
- Referees are excluded from team clustering.
- Predicted `TEAM_0/TEAM_1` IDs are aligned to SoccerNet `left/right` **per sequence using outfield tracks only**. The same mapping is then used for goalkeepers.
- `UNKNOWN` counts as wrong in overall accuracy and is excluded from selective accuracy.

## Methods

### B0 — `bbox-color`
Third-party intrinsic baseline that uses SoccerNet GT bbox/track/role and the Stage-5 HSV/Lab colour descriptor with bbox torso/lower-body regions. No pose model is needed.

### B1 — `stage5-color`
The actual pose-guided Stage-5 colour method. This requires a per-sequence pose cache under `--pose-cache-root`. The benchmark refuses to silently fall back to bbox-only mode when a pose cache is missing.

Accepted pose-cache locations for `SNGS-021`:

- `<pose-root>/SNGS-021/tracked_pose_2d_state.json`
- `<pose-root>/SNGS-021/stage5_gsr_pose_cache.json`
- `<pose-root>/SNGS-021.json`

Accepted schemas are `tracked-pose-2d-state-1.0` and `stage5-gsr-pose-cache-1.0`, provided track IDs correspond to the GSR oracle tracks used by the intrinsic protocol.

### B2 — `external`
Imports third-party/official predictions for comparison. Accepted forms:

- COCO-like JSON with `annotations[].track_id` and `annotations[].attributes.team` (`left/right`),
- `{track_id: 0/1/left/right}` mapping,
- `{ "track_team": { track_id: {"team_id": ...} } }`.

This makes the harness usable with SoccerNet official baseline exports without coupling Stage 5 to that baseline's runtime environment.

## Metrics

Primary:

- track-level overall accuracy,
- track-level selective accuracy,
- coverage / UNKNOWN rate,
- macro-F1,
- ARI,
- NMI,
- outfield accuracy,
- goalkeeper team accuracy,
- referee team contamination rate.

Reports contain both micro counts and macro-by-sequence metrics. A deterministic 95% bootstrap CI is computed by resampling **sequences**, not frames.

## Commands

Inspect the local dataset:

```powershell
python benchmark_soccernet_gsr.py inspect `
  --gsr-root "D:\GSR" `
  --split valid
```

Run B0:

```powershell
python benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" `
  --split valid `
  --method bbox-color `
  --output-dir ".\benchmark_results\gsr_valid_bbox_color"
```

Run B1 after generating oracle-track pose caches:

```powershell
python benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" `
  --split valid `
  --method stage5-color `
  --pose-cache-root "D:\GSR_STAGE3_ORACLE" `
  --output-dir ".\benchmark_results\gsr_valid_stage5_color"
```

Evaluate an external/official baseline export:

```powershell
python benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" `
  --split valid `
  --method external `
  --external-predictions-root "D:\GSR_OFFICIAL_PRED" `
  --output-dir ".\benchmark_results\gsr_valid_official"
```

Compare summaries:

```powershell
python benchmark_soccernet_gsr.py compare `
  --summary ".\benchmark_results\gsr_valid_bbox_color\benchmark_summary.json" `
  --summary ".\benchmark_results\gsr_valid_stage5_color\benchmark_summary.json" `
  --summary ".\benchmark_results\gsr_valid_official\benchmark_summary.json" `
  --output ".\benchmark_results\comparison.md"
```

## Output

Each run writes:

- `benchmark_summary.json`
- `benchmark_summary.md`
- `sequence_metrics.jsonl`
- `sequences/<SNGS-id>/prediction.json`
- `sequences/<SNGS-id>/metrics.json`

No metric from the project replay/frame 104 is included in this benchmark.
