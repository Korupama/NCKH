# Stage 2 specification — SST Human Entities, Roles & Target-Window Tracking

## 1. Mission

```text
Stage-1 target-centred replay window
    -> physical human entities
    -> stable identities
    -> track-level roles
    -> Stage-3 candidate tracks + shared raw RTMW cache
```

Stage 2 is image-space entity perception. It does not own pose quality, teams, world placement or offside geometry.

## 2. Inputs

Required logical input:

- original replay video;
- global `selected_frame=t0` chosen by the user upstream;
- Stage-1 `window_start/window_end` and shot bounds;
- original image width/height and FPS;
- coordinate space equivalent to Stage-1 `original_raw`.

Preferred source is the actual frozen Stage-1 v12 workspace through `stage1_adapter.py`.

## 3. Perception pipeline

### 3.1 SST detection

Classes retained from SST:

- Ball;
- Player;
- Goalkeeper;
- Main referee;
- Side referee;
- Staff.

Default starting thresholds:

| Class | threshold |
|---|---:|
| Player | 0.50 |
| Goalkeeper | 0.50 |
| Ball | 0.35 |
| Main/Side referee | 0.55 |
| Staff | 0.60 |

These are starting inference settings, not frozen research-optimal thresholds.

### 3.2 Physical-human consolidation

Class-local NMS is retained first. Different human classes are then grouped by the v1.3 geometry-aware rule: direct merge at IoU >= `0.82`, or merge at IoU >= `0.60` when center distance, area similarity and intersection-over-min-area are jointly consistent. Exact thresholds are configurable and recorded in the perception manifest.

For each physical hypothesis, Stage 2 preserves:

- all source detection IDs;
- per-class evidence;
- collapsed role evidence (`player/goalkeeper/referee/other`);
- representative bbox;
- role score/margin/status;
- `pose_required` and candidate hint.

The resolved role is evidence-driven. There is no hard rule that Goalkeeper always beats Player or Referee always beats Player.

### 3.3 RTMW cue cache

RTMW WholeBody-133 is executed once on physical hypotheses that require pose evidence. It is deliberately skipped for confidently excluded referee/staff hypotheses.

The cache contains raw 133-point image coordinates and raw SimCC maxima. Stage 2 does not export the old `legal_for_offside_position` flags as legal truth.

### 3.4 Decision-frame-anchored tracking

Every Stage-2 track is anchored to one physical human at `t0`, then associated independently backward and forward.

Association currently combines:

- bbox IoU;
- center displacement;
- RTMW body-pose cue when available;
- conservative role compatibility penalty.

Objects that never correspond to a target-frame anchor are retained as orphan diagnostics rather than becoming Stage-3 tracks.

### 3.5 Track-level role aggregation

Frame evidence is aggregated over the track. Final role ∈:

```text
player | goalkeeper | referee | other
```

Only `player` and `goalkeeper` set `candidate_for_stage3=true`.

## 4. Output contract

Primary `EntityTrackState 1.0`:

```text
schema_version
stage2_version
replay_context
status
tracks[]
  track_id
  role / role_score / role_margin / role_status
  candidate_for_stage3
  identity_confidence
  observations[]
    frame_index
    bbox_xyxy                   # original/raw pixels
    detector_score
    frame_role + role_evidence
    physical_human_id
    source_detection_ids
    pose_cache_key optional
    association_cost optional
selected_frame_entities[]
excluded_detections_by_frame
auxiliary_ball_detections_by_frame
stage3_handoff
backend / diagnostics / artifacts
```

No team ID, attack direction, pitch XY, Body3D or offside result is permitted in this contract.

## 5. Visual contract

Default user visual at `t0`:

- Player and Goalkeeper bbox only;
- persistent track ID;
- track-level role;
- no referee/staff bbox;
- no skeleton;
- no team colours;
- no offside geometry.

Debug visual may additionally show excluded referee/staff.

Temporal QA video shows the same IDs and short trails through the target window. `RTMW` in debug video means raw cue available, not Stage-3 pose acceptance.

## 6. Evaluation

### Detector/entity metrics

- per-class precision/recall and AP where GT supports it;
- `CandidateRecall` for Player/GK;
- `RefereeLeakageRate` into candidate set;
- `CrossClassDuplicateRate` before and after consolidation.

### Tracking metrics

- HOTA;
- DetA;
- AssA;
- IDF1;
- ID switches;
- fragmentation;
- `TCR@1s` around the selected frame.

### Project-specific selected-frame metrics

- Selected-frame entity completeness;
- candidate completeness at `t0`;
- role correctness at track level.

Initial engineering targets (to be frozen only after baseline validation):

| Metric | minimum | target |
|---|---:|---:|
| Player recall @ IoU .5 | .95 | .97 |
| Goalkeeper recall | .90 | .95 |
| Referee leakage | ≤ .02 | ≤ .01 |
| post-consolidation cross-class duplicate rate | < .02 | < .01 |
| track-level role macro-F1 | .90 | .95 |
| HOTA | .60 | .70 |
| AssA | .55 | .70 |
| IDF1 | .60 | .70 |
| TCR@1s | .90 | .95 |

Runtime `VALID/DEGRADED` is an internal quality signal and is not a substitute for these GT metrics.

## 7. Data and model policy

Priority order:

1. open/public source implementations and pretrained weights;
2. the user's already-authorized SoccerNet data for evaluation/fine-tuning;
3. additional training only after failure analysis.

Primary runtime assets:

- SST detector checkpoint;
- RTMW WholeBody ONNX.

SoccerNet GSR remains an external benchmark/reference rather than a mandatory runtime dependency for this primary Stage 2.

Credentials are never stored in source, notebook, JSON, logs or exported ZIPs.

## 8. Definition of Done

Implementation DoD:

- Stage-1 adapter passes;
- exact target window uses global raw-frame coordinates;
- cross-class consolidation is executed **before RTMW**;
- all target-frame humans receive anchored track IDs;
- referee/staff are retained internally but cannot leak into Stage-3 candidates after final role aggregation;
- RTMW cache is re-keyed by track ID without invented pose on missing frames;
- EntityTrackState + Stage3 handoff + visual QA are emitted;
- regression/unit tests pass.

Research DoD additionally requires the GT evaluation metrics above. Until then Stage 2 is implemented but not scientifically frozen.

## 9. Benchmark implementation v1.1

The research DoD is now executable through `benchmark_stage2.py`. The official Stage-2 protocol is t0-centric rather than full-clip MOT: target identities are those present at the selected frame, and tracking is evaluated only for those anchors through ±1 s. Both all-human and Player/GK-candidate tracking are reported, with candidate tracking treated as the downstream-primary view.

The quick protocol uses 10 sequences × 3 deterministic target frames; the full protocol uses every selected-split sequence × 5 deterministic target frames. Overlapping windows share a per-frame SST/RTMW perception cache.

The harness computes image-space HOTA/DetA/AssA and IDF1 from the public TrackEval reference equations and should be cross-checked against the official TrackEval implementation before publication. Full SoccerNet GS-HOTA remains outside Stage 2 because team, jersey and pitch/world state are intentionally downstream responsibilities.

### Detector-oracle association diagnostic

The benchmark harness provides two oracle-box diagnostics. `oracle_boxes_geom` uses GT image boxes/roles without pose; `oracle_boxes_pose` additionally runs the same RTMW-L model on GT candidate boxes. Neither passes SoccerNet GT `track_id` to the tracker. The latter is the preferred diagnostic for separating SST/consolidation error from RTMW-assisted association error when an RTMW model is available.

## 9. Benchmark execution resilience (v1.2)

Research evaluation is resumable and separate from the Stage-2 runtime contract.

- SST+RTMW perception is checkpointed atomically per frame.
- Detection evaluation is checkpointed once per deterministic t0 window.
- Tracking evaluation is checkpointed separately for each experiment/ablation.
- `benchmark_state.json` records global progress and terminal state.
- `partial_summary.json` is refreshed after every fully committed window.
- A run signature covers exact protocol windows, tracker settings, SoccerNet label-file SHA256 values,
  and benchmark evaluator source fingerprint. Incompatible old metric checkpoints are rejected.
- Model SHA256 fingerprints protect perception/oracle-pose caches from accidental mixing.
- Re-running the same command resumes automatically; `--restart-evaluation` intentionally discards only
  evaluator checkpoints, while `--overwrite-perception` intentionally rebuilds model inference.
- Console progress reports sequence/frame/task counts, elapsed time, ETA and checkpoint commits.

A Stage-2 scientific freeze must be based on a `COMPLETE` benchmark state, never a partial summary.

## v1.3 M1 addendum — hierarchical candidates and rescue observations

Stage-3 eligibility is defined by the hierarchical superclass `footballer`, which contains SST Player and Goalkeeper evidence. Fine Player/GK role is retained for audit/quality metrics but does not define Stage-3 eligibility.

Perception exposes two observation tiers:

- `primary`: passes production class threshold; can anchor a t0 entity and receives RTMW when pose is required;
- `rescue`: below the production threshold but above the low-score cache floor; footballer-only, does not receive RTMW, cannot create a t0 entity, and may only associate to an existing unmatched footballer track.

The association order is primary-first then rescue-only. Default `max_gap=6`; default rescue assignment ceiling is `0.78` versus `0.92` for normal observations.

Benchmark decomposition separates selected-frame completeness (`AnchorCoverage`) from conditional temporal continuity (`ConditionalTCR`) while retaining overall TCR as the downstream hard gate.
