# Stage 3 specification — Tracked 2D Whole-Body Pose QA

## Mission

```text
Stage-2 candidate tracks + raw RTMW WholeBody133
    -> structural validation
    -> canonical WholeBody133 representation
    -> per-keypoint evidence state
    -> anatomy/geometry QA
    -> temporal QA
    -> optional controlled same-model re-crop
    -> TrackedPose2DState 1.0
```

Stage 3 owns 2D pose representation and quality semantics. It does not own player identity discovery, team identity, pitch/world coordinates, 3D lifting, IFAB legal-body semantics or offside decisions.

## Input contract

Required files:

```text
stage2_entity_tracks.json
stage2_rtmw_track_cache.json
stage3_handoff.json
```

Hard structural requirements:

- `EntityTrackState` schema family is supported;
- Stage-2 raw-pixel coordinates are `RAW_DISTORTED_PIXEL`;
- every candidate ID resolves to one entity track;
- each candidate has exactly one bbox observation at `t0`;
- RTMW cache schema is `stage2-raw-rtmw-track-cache-1.0`;
- duplicate RTMW observations at `(track_id, frame_index)` are forbidden.

Missing pose evidence is represented explicitly; it is not fabricated during preflight.

## Canonical keypoint schema

`COCO_WHOLEBODY_133` is the public source of truth:

```text
0..16    body17
17..22   feet6
23..90   face68
91..111  left hand21
112..132 right hand21
```

H36M17 or later Stage-5 model layouts are derived views only.

## Raw score semantics

Stage 2 exports RTMW SimCC maxima. These values can be greater than one and are not probabilities. Stage 3 therefore separates:

```text
raw_model_score          # uncalibrated RTMW SimCC evidence
state                    # Stage-3 evidence/QA label
quality_score_calibrated # intentionally absent in v0.1
```

No fixed `0.5 = 50% confidence` assumption is permitted.

## Per-keypoint states

```text
VALID
LOW_MODEL_EVIDENCE
GEOMETRIC_OUTLIER
TEMPORAL_OUTLIER
LEFT_RIGHT_SUSPECT
MISSING
```

`TEMPORAL_IMPUTED` is reserved for downstream representations, but v0.1 deliberately keeps a temporal estimate in the separate `temporal_estimate_xy` field instead of overwriting raw coordinates.

## Per-pose states

```text
VALID
DEGRADED
REJECTED
MISSING
```

The v0.1 gate combines body/feet/core availability, bbox-relative plausibility and broad bone-length sanity. It is an engineering QA gate, not a learned probability calibration.

## Temporal QA

Coordinates are normalized by each observation bbox before temporal diagnostics. Stage 3 reports:

- normalized second-difference jitter;
- temporal-outlier fraction;
- left/right swap suspicion against the previous adjacent frame;
- optional interpolation estimate for missing points.

No temporal operation silently moves raw keypoints, especially at `t0`.

## Optional controlled re-inference

Default production path reuses the Stage-2 RTMW cache. Re-inference is opt-in only.

When enabled, the same RTMW-L graph is run with configured crop scales, e.g. `1.0, 1.1, 1.2`. A candidate wins only if its pose-status rank/engineering score improves. If a fallback is selected, the original cache pose is preserved under `upstream_raw_pose` and all attempts are recorded in provenance.

## Output contract

Primary output: `tracked_pose_2d_state.json`.

Important top-level fields:

```text
schema_version = tracked-pose-2d-state-1.0
stage3_version
source_stage2 + SHA256 provenance
replay_context
coordinate_space
keypoint_schema
score_semantics
configuration
preflight
tracks[]
selected_frame_poses[]
metrics
diagnostics
artifacts
```

`stage3_downstream_handoff.json` groups valid, degraded and rejected/missing tracks at `t0` but explicitly warns downstream stages not to treat Stage-3 anatomy as legal-body semantics.

## Non-goals

The following are forbidden in Stage 3:

- creating new human identities missed by Stage 2;
- team/attacking/defending assignment;
- projecting elevated body points with a pitch homography;
- world XYZ or ground XY;
- removing arms/hands according to IFAB offside rules;
- choosing the offside-defining body point;
- computing the offside plane.
