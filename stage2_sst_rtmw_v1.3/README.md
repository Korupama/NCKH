# Stage 2 — SST Human Entities, Roles & Target-Window Tracking

This project is the Stage-2 extraction of the previous `sst_rtmw_3d_world_project_v04` perception/tracking front-end.
It intentionally stops before team assignment, world placement, Body3D and offside logic.

## Frozen Stage-2 responsibility

Stage 2 answers:

1. Which physical human entities are visible in the Stage-1 replay window?
2. Where is each entity in original/raw image pixels?
3. Which persistent `track_id` belongs to each entity around the user-selected decision frame?
4. Is each track a `player`, `goalkeeper`, `referee`, or `other`?

It does **not** decide teams, attacking/defending status, 3D pose, world XYZ, second-last opponent or offside.

## Primary pipeline

```text
Stage-1 ReplayContext / target window
        |
        v
SST 6-class detection
  Ball / Player / Goalkeeper / Main referee / Side referee / Staff
        |
        v
class-local threshold + NMS
        |
        v
CROSS-CLASS PHYSICAL-HUMAN CONSOLIDATION   <-- new, before RTMW
        |
        +--> Player / Goalkeeper candidate hint
        +--> Referee / Staff retained internally but excluded from Stage-3 candidate set
        +--> Ball retained as auxiliary evidence only
        |
        v
RTMW WholeBody-133 shared cue cache
        |
        v
decision-frame-anchored bidirectional Hungarian tracking
        |
        v
track-level role aggregation
        |
        +--> stage2_entity_tracks.json
        +--> stage2_rtmw_track_cache.json
        +--> stage3_handoff.json
```

RTMW is executed in Stage 2 only once as a **shared perception cache / tracking cue**. Its raw pose is not declared a Stage-3-quality pose result. Stage 3 receives that cache and performs the formal pose normalization/quality/evaluation step.

## Why cross-class consolidation is mandatory

The older pipeline used class-local NMS. In the supplied real frame 85 it contained, for example:

- `player_009` + `main_referee_001`, IoU ≈ 0.916;
- `player_004` + `goalkeeper_002`, IoU ≈ 0.909.

Those are mutually exclusive labels for nearly the same physical box. Stage 2 v1.3 replaces the strict-IoU-only rule with geometry-aware cross-class consolidation. A direct high-IoU rule (default 0.82) is complemented by a geometry-consistent rule (IoU >= 0.60 plus center/area/intersection checks). Player and Goalkeeper evidence is additionally grouped under the `footballer` superclass so exact Player/GK disagreement does not become a Stage-3 eligibility error.

## Stage-1 compatibility

`--stage1-root` accepts the actual frozen Stage-1 v12 workspace. The adapter validates:

- `stage1_ready_for_stage2=true`;
- Stage-1 package `0.12.0` and `CameraState` schema `1.2`;
- target-frame consistency;
- contiguous target window;
- original video FPS/frame count/resolution;
- Stage-1 `image.pixel_space == original_raw`;
- Stage-1 shot bounds when transition diagnostics are available.

`original_raw` is normalized to Stage-2's name `RAW_DISTORTED_PIXEL` as a semantic alias only. No resize or undistortion is applied.

## Model assets

Model weights are deliberately not bundled.

Required for a fresh production run:

- SST checkpoint (`model.pth` / trusted equivalent);
- RTMW WholeBody ONNX model (the project was previously validated with RTMW-L 384×288).

The supplied v0.4 artifact had those assets omitted, so this package includes a legacy migration path and real JSON samples for contract/regression tests without pretending that a fresh model run occurred.

## Production run

```bash
python run_stage2.py \
  --stage1-root /path/to/stage_1_camera_v12 \
  --sst-checkpoint /path/to/model.pth \
  --rtmw-model /path/to/rtmw_l_384x288.onnx \
  --output-dir runs/stage2
```

The Stage-1 analysis window is decoded losslessly to PNG with global frame names. SST and RTMW load once, then all frames are processed.

## Reuse a perception manifest

```bash
python run_stage2.py \
  --stage1-root /path/to/stage_1_camera_v12 \
  --perception-manifest /path/to/perception_manifest.json \
  --output-dir runs/stage2
```

## Migrate old v0.4 SST+RTMW JSON for testing

```bash
python run_stage2.py \
  --replay-context samples/replay_context_legacy_validation.json \
  --legacy-json-dir samples/legacy_validation \
  --output-dir runs/legacy_validation \
  --no-video
```

Legacy mode reconstructs the new consolidation/tracking contract but is explicitly marked as migration-only because v0.4 ran RTMW **before** cross-class consolidation.

## Stage-3 handoff

Stage 3 consumes:

- `stage2_entity_tracks.json` — entity/identity source of truth;
- `stage2_rtmw_track_cache.json` — raw WholeBody-133 observations re-keyed by `(track_id, frame_index)`;
- `stage3_handoff.json` — candidate track IDs and readiness diagnostics.

Only final track roles `player` and `goalkeeper` are candidates for Stage 3. Referee/staff tracks are retained internally for audit and hidden from the default user visual.

RTMW scores are stored as:

```text
RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY
```

They must not be interpreted as probabilities.

## Tests

```bash
python -m pip install -e .
pytest -q
```

Tests include the supplied real v0.4 frame-85 duplicate cases, the real three-frame 85–87 migration, the Stage-1 v12 adapter when the supplied workspace is mounted, and a mocked production perception test proving consolidation occurs before RTMW.

## Research status

Code/contract status: **IMPLEMENTED**.

Model/metric status: **M1 IMPLEMENTED, NOT YET FROZEN**. The v1.2 quick benchmark identified detection completeness, cross-class grouping and short temporal dropouts as the main refinement targets. v1.3 implements those inference/evaluator changes; a fresh v1.3 quick run is required before full validation.


## M1 additions in v1.3

SST human detections are persisted down to `--raw-human-score-floor` (default 0.20). Normal high-threshold detections still create t0 entities; low-score footballer hypotheses are **rescue-only** and may only fill an existing anchored track. They never create a new t0 track and are not sent to RTMW. The tracker default `max_gap` is 6 frames.

Important benchmark diagnostics now include `CandidatePrecision`, apples-to-apples `AnyDuplicateRate` / `CrossClassConflictRate`, `AnchorCoverage`, and `ConditionalTCR`. Exact Player/GK classification is reported as a quality metric rather than a hard Stage-3 handoff blocker.

After a v1.3 quick run, sweep Player/GK thresholds without rerunning SST/RTMW:

```powershell
python benchmark_stage2.py sweep-thresholds --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --benchmark-dir ".\benchmark_results\quick_v13" --player-thresholds "0.30,0.35,0.40,0.45,0.50" --goalkeeper-thresholds "0.30,0.35,0.40,0.45,0.50"
```

See `MODEL_IMPROVEMENT_M1.md`. Historical changelogs are archived in `../docs_ban_giao/tai_lieu_lich_su_20260920.zip`.

## SoccerNet-GSR benchmark (v1.3 / M1)

Stage 2 v1.3 retains the t0-centric benchmark harness and adds M1 cache/evaluator improvements. Start with:

```powershell
python benchmark_stage2.py inspect --soccernet-root "D:\Datasets\SoccerNetGS" --split valid
```

Preview the deterministic window/frame budget:

```powershell
python benchmark_stage2.py plan --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick
```

Then run the quick protocol:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\Datasets\SoccerNetGS" --split valid --protocol quick --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --output-dir ".\benchmark_results\quick"
```

See `BENCHMARK.md` for protocol details, cache reuse, metrics, ablations and the full-validation command.

## Benchmark checkpoint/resume (v1.3)

Long SoccerNet evaluation is crash-resumable by default. Perception is committed per frame and
benchmark metrics are committed per target-window task/ablation. Re-run the **same command** with
the same `--output-dir` after an interruption; completed units are reused automatically.

Check progress at any time without loading models:

```powershell
python benchmark_stage2.py status --output-dir ".\benchmark_results\quick"
```

`--restart-evaluation` clears only metric/evaluation checkpoints while retaining expensive
SST+RTMW perception caches. `--overwrite-perception` is intentionally separate and rebuilds model
inference caches. The console reports sequence, perception-frame, evaluation-task, elapsed-time,
ETA and checkpoint events. Use `--progress-every N` to control frame-level reporting.
