# Stage 6 Ball Contract — v0.5.0

Contact-aware output uses `ball-trajectory-state-1.2` and preserves legacy selected-frame
XYZ/method/extent fields. New blocks: observation, contact, ground_anchor, localization,
fallback, anchor_sensitivity. `stage6_downstream_handoff.json` supplies Stage7 contact
identity/evidence and Stage8 geometric X/extent. It never supplies a team, attack direction
or offside verdict. `usable_for_offside` means available geometry, NOT validated accuracy.
Stage3 requires matching replay, image size, FPS and RAW_DISTORTED_PIXEL coordinates.
Stage4 accepts the quality-gated selected_frame_ground_anchor contract with VALID
DENSEST_ANKLE_FOOT_CLUSTER_MEDIAN and canonical metre axes. Upper-body proxies are ignored.
The old hybrid/temporal modes retain schema 1.1; new fields are additive in contact-aware mode.

## Scope

Stage 6 consumes Stage-1 camera geometry plus ball detections and produces a metric ball trajectory for the user-selected replay window. It does not detect the frame of pass and it does not decide offside.

## Coordinate contract

- Image observations: `RAW_DISTORTED_PIXEL`.
- World output: Stage-6 canonical pitch coordinates.
- `X`: goal-to-goal.
- `Y`: touchline-to-touchline.
- `Z`: up-positive.
- Stage-1 camera projection/undistortion remains the single camera-geometry authority.

## v0.4 temporal observation model

For a directly observed frame, the detector supplies a centre pixel and apparent diameter. The centre defines a Stage-1 world ray. v0.4 keeps the optimized 3D point on that ray and estimates the scalar range along it.

The size-prior relation remains the observation initializer:

```text
rho_size = ball_radius / sin(apparent_angular_radius)
```

but `rho_size` is no longer automatically treated as the final production range.

## Temporal stages

1. Robust log-diameter refinement.
2. Short-gap centre/diameter interpolation.
3. Per-frame ray construction.
4. Range observations from refined apparent size.
5. Optional ground-compatible range anchor.
6. Joint trajectory optimization with world-position smoothness.
7. Feasibility and measurement guardrails.

Long missing gaps are not bridged.

## Output semantics

`ball_trajectory_state.json` schema is `ball-trajectory-state-1.1` for v0.4 production output. Existing top-level keys remain compatible with v0.3 readers; temporal information is added in diagnostics.

A normal temporally optimized frame uses:

```text
selected_method = TEMPORAL_DIAMETER_RANGE_TRAJECTORY
localization_status = VALID_TEMPORAL_3D
```

An interpolated short-gap frame uses:

```text
localization_status = VALID_TEMPORAL_3D_INTERPOLATED
```

If temporal support is absent at the user-selected `t0` but the old single-frame size prior is valid:

```text
selected_method = MONOCULAR_BALL_SIZE_PRIOR_FALLBACK
localization_status = DEGRADED_TEMPORAL_FALLBACK_...
```

## Offside handoff

At the selected frame `t0`, Stage 6 provides `center_xyz_world_m`, `X_world_m`, and `ball_center_x_extent_m`. Downstream offside logic remains responsible for comparing the ball longitudinal extent with eligible defender/attacker body geometry.
