# Stage 2 SST–RTMW v1.3 — Implementation status

## Implemented

- actual Stage-1 v12 workspace adapter;
- exact lossless target-window extraction with global frame indices;
- SST football-class backend integration;
- strict cross-class physical-human consolidation before RTMW;
- candidate/excluded human semantics;
- RTMW WholeBody-133 shared raw cue cache with explicit non-probability score semantics;
- decision-frame-anchored bidirectional Hungarian tracking;
- track-level role aggregation;
- persistent `EntityTrackState 1.0`;
- RTMW cache re-keyed by `track_id` and global frame index;
- Stage-3 handoff manifest;
- selected-frame and temporal visual QA;
- auxiliary SST ball retention;
- selected-frame evaluation helpers;
- legacy v0.4 migration/regression path.

## Executed in this artifact runtime

Using the real v0.4 SST+RTMW JSON for source frames 85–87:

- frame 85: 13 raw human-class detections -> 11 physical humans -> 10 Player/GK candidates;
- `player_009 + main_referee_001` collapsed to one referee hypothesis;
- `player_004 + goalkeeper_002` collapsed to one player hypothesis;
- three-frame migration creates 10 tracks anchored at frame 86;
- 2 cross-class duplicate groups are reported as collapsed;
- one extra frame-85 border player is retained as an orphan rather than inventing a new decision-frame track;
- RTMW cache is re-keyed to `track_001...track_010`;
- CLI legacy validation completes with Stage-2 structural status `VALID`.

The supplied Stage-1 v12 workspace adapter was also tested against the actual ZIP and resolves:

- 1920x1080;
- 30 FPS;
- 213 frames;
- selected frame 86;
- Stage-1 window 56..116;
- shot 0..212;
- `original_raw` -> `RAW_DISTORTED_PIXEL` semantic alias without image transform.

Automated v1.0 tests in the earlier artifact runtime: **5 passed**.

## Not executed here

A fresh 61-frame SST+RTMW production inference was not run because the user intentionally omitted model weights/third-party assets from the supplied ZIP. The production path is implemented and requires explicit `--sst-checkpoint` and `--rtmw-model` paths.

Research metrics (HOTA/AssA/IDF1/CandidateRecall on annotated validation data) have not yet been measured, so Stage 2 must not be labeled scientifically frozen yet.

## v1.1 benchmark harness

Implemented after the v1.0 runtime demo:

- SoccerNet-GSR COCO-style `Labels-GameState.json` reader with v1.3 gate;
- deterministic quick/full t0-centred protocols;
- union-of-windows perception cache so SST/RTMW does not rerun on overlapping windows;
- selected-frame human precision/recall, CandidateRecall and RefereeLeakageRate;
- GT-referenced cross-class duplicate before/after metrics;
- 101-point AP50 and mAP50:95 in image space;
- track-level role F1 + confusion matrix;
- self-contained TrackEval-equation HOTA/DetA/AssA/LocA and Identity IDF1/IDR/IDP;
- t0-centric `TCR@1s`;
- human and Player/GK-candidate tracking reports;
- `primary`, `no_pose`, and `oracle_boxes_geom` tracking ablations;
- compact CSV/JSON reports plus failure-case diagnostics/overlays;
- model-free cache reuse for evaluator iteration;
- synthetic end-to-end benchmark tests.

The benchmark code is implemented and synthetically verified. No real SoccerNet metric is claimed in this artifact because the user's SoccerNet-GSR validation files are not mounted in this runtime.

v1.1 standalone automated suite: **8 passed, 1 skipped**. The skipped test is the optional actual-Stage-1 workspace integration test because this packaged benchmark validation is intentionally runnable without a mounted Stage-1 directory. The benchmark-specific loader/metrics/runner/cache tests passed. See `validation_reports/BENCHMARK_HARNESS_SMOKE_REPORT.json`.

Additional v1.1 safeguards:

- model SHA256 fingerprints are stored in fresh perception manifests;
- cache/model/stage-version mismatches are rejected instead of silently mixing predictions;
- `plan` command previews deterministic t0 windows and the unique inference-frame budget before GPU work;
- optional `oracle_boxes_pose` ablation: SoccerNet GT boxes/roles + RTMW pose, while withholding GT identity from association; cached per GT frame to avoid repeated RTMW inference across overlapping windows.

## v1.2 resilience/progress update

Added for long benchmark runs:

- atomic per-frame SST+RTMW perception JSON writes;
- `perception_checkpoint.json` written before inference and updated after every completed frame;
- model-fingerprint validation before reusing an interrupted perception cache;
- atomic evaluation checkpoints at detection-window and per-ablation tracking granularity;
- deterministic benchmark run signature includes exact windows and SHA256 of SoccerNet label files;
- `benchmark_state.json` with `RUNNING / INTERRUPTED / FAILED / COMPLETE` status;
- `partial_summary.json` refreshed after every fully committed window;
- automatic resume by rerunning the same command/output directory;
- `benchmark_stage2.py status` for zero-inference progress inspection;
- `--restart-evaluation` to discard only evaluation checkpoints while preserving perception cache;
- live console progress with sequence/frame/task counts, elapsed time, ETA and metric snippets;
- clean Ctrl+C writes interrupt state and preserves completed work.

Automated suite after this update: **10 passed** in the artifact runtime, including a simulated
mid-benchmark interruption followed by resume without rewriting the already-completed detection checkpoint.

## v1.3 M1 update

Implemented:

- low-score SST cache with configuration/model fingerprinting;
- footballer/referee/other hierarchical superclass semantics;
- geometry-aware cross-class physical-human consolidation;
- rescue-only temporal observations and primary-first association;
- default gap budget 6 frames;
- corrected duplicate metrics and candidate-only precision;
- `AnchorCoverage` + `ConditionalTCR` decomposition;
- cache-only Player/GK threshold sweep;
- benchmark CLI support for active detector thresholds;
- regression tests for the M1 invariants.

Status remains **M1 IMPLEMENTED / NEEDS FRESH QUICK BENCHMARK**. Full validation should not be run until the v1.3 quick benchmark and threshold/rescue ablations are reviewed.
