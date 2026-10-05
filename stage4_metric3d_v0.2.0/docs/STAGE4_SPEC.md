# Stage 4 v0.5 Specification

This is the base v0.5 specification. The [v0.5.1 ground-first report](../V051_GROUND_FIRST_REPORT.md)
supersedes ground-anchor validity, initialization, refinement bounds and ground quality gates.

## Purpose

Estimate a coherent world-grounded player skeleton in the canonical metric pitch frame for each usable tracked player around the manually selected replay frame `t0`.

## Execution lanes

Stage 4 is independently improvable from Stage 1 and Stage 3. It has two
separate lanes:

1. **Model-only lane:** a Stage-4 pretrained backend receives benchmark RGB
   images/person crops, person boxes and the camera/GT metadata supplied by the
   benchmark. This lane is used to compare pretrained models, preprocessing,
   root refinement and temporal settings. Stage 1/3 files are not required.
2. **Integration lane:** the project handoff consumes Stage-1 camera states and
   Stage-3 2D tracks as immutable inputs. This lane verifies deployment
   compatibility and pitch-world placement, but its missing artifacts do not
   block model-only development.

## Integration inputs

1. Stage-1 `CameraState` timeline: `K`, `R_world_to_camera`, camera centre `C`, distortion, pitch plane `Z=0`.
2. Stage-3 `tracked-pose-2d-state-1.0`: track IDs, player/goalkeeper roles, bboxes, RTMW WholeBody133 keypoints and QA states.
3. `stage4-sam3d-native-cache-1.0`: official SAM3D MHR70 2D/relative-3D keypoints, `pred_cam_t`, focal length.

## Pretrained model

Primary provider: official SAM 3D Body DINOv3. Stage 4 passes Stage-1 intrinsics to SAM3D. No Field Converter checkpoint is required by the v0.5 production candidate.

## Geometry

For native SAM3D joint `j`:

```text
J_cam(j) = J_rel(j) + r_cam
```

The optimization variable is only `r_cam`.

### SAM prior

```text
r_cam ≈ pred_cam_t
```

### RTMW reprojection

Verified MHR70 joints are projected with the Stage-1 camera/distortion and compared against Stage-3 raw distorted pixels.

### Ground anchor

For valid toe/heel landmarks, Stage-3 image pixels are intersected with the Stage-1 pitch plane. A translation candidate is:

```text
r_ground = ground_hit_cam - J_rel(foot)
```

The ground candidate requiring the smallest correction from the SAM prior is preferred. If all distal-foot landmarks are unavailable, ankles are an explicit degraded fallback. Ground candidates too far from the SAM prior are recorded but not forced into the objective.

### Temporal term

Translation is transformed to Stage-1 world coordinates before temporal second-difference regularization so camera PTZ motion is not mistaken for player motion.

## Output schemas

- Full state: `world-grounded-pose-state-1.1`
- Downstream handoff: `stage4-downstream-handoff-2.1`

The handoff exposes track ID, role, root/pelvis world position, canonical joint world XYZ, validity and quality only.

## Quality gates

1. `implementation_gate`
2. `geometric_quality_gate`
3. `metric_accuracy_gate`
4. `downstream_offside_gate`

Passing implementation/geometric sanity does not freeze research accuracy.

## Non-ownership

Stage 4 does not own team identity, attacker/defender semantics, attack direction, toucher, ball state, legal-body filtering, second-last opponent selection, or offside-position decisions.
