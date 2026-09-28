# Stage 8 Specification — Offside Reference Geometry v0.1.0

## Responsibility

Stage 8 deterministically converts already-resolved upstream context into the longitudinal reference geometry needed by Stage 9.

### In scope

1. Consume Stage-7 `opponents` and attack direction `s in {-1,+1}`.
2. Consume Stage-4 world-space Pose23 landmarks at the same selected frame.
3. Build a project-scope legal-landmark proxy by excluding elbow/wrist arm landmarks.
4. Compute each opponent's goalward legal-landmark extent.
5. Rank all opponents in the attack direction and identify the second-last opponent reference coordinate.
6. Consume the Stage-6 ball longitudinal extent.
7. Choose the final reference as the goalward-most of the ball and second-last-opponent coordinates.
8. Emit a canonical vertical plane `X = X_ref` for Stage 9.

### Out of scope

- attacker offside-position classification
- own-half test
- excluding the toucher from attacker candidates
- interference / gaining-advantage / offence semantics
- exact anatomical body-surface reconstruction
- correcting upstream team, role, tracking, camera, body-geometry or ball-localization errors

## Coordinate convention

Stage 4 must use `STAGE1_PITCH_WORLD`:

- X: goal-to-goal
- Y: touchline-to-touchline
- Z: up
- metres
- pitch plane Z=0

Stage 7 supplies attack direction `s`.

Define the goalward coordinate:

```text
q = s * X
```

Larger `q` always means closer to the opponent goal line, independent of left/right attack direction.

## Legal-landmark proxy

Included Pose23 landmarks:

- nose, eyes, ears
- shoulders
- hips
- knees
- ankles
- big toes, small toes, heels

Excluded:

- elbows
- wrists

This is explicitly a landmark-based proxy. It is not an exact body surface and does not claim a calibrated armpit/boot/head contour.

For opponent i:

```text
q_i = max_j (s * X_ij)
```

where j runs over finite included legal landmarks at t0.

## Second-last opponent

Sort opponents by descending `q_i`.

- rank 1: last opponent
- rank 2: second-last opponent

A goalkeeper has no special Stage-8 rule; they are ranked exactly like any other opponent supplied by Stage 7.

Numerically tied second-last coordinates preserve the reference `q` while marking identity as ambiguous and emitting all candidate track IDs.

## Ball reference

Stage 6 supplies ordered `[xmin, xmax]` in world X.

```text
q_ball = max(s*xmin, s*xmax)
```

## Final reference

```text
q_ref = max(q_second_last, q_ball)
X_ref = s * q_ref
```

Reference source is one of:

- `SECOND_LAST_OPPONENT`
- `BALL`
- `BALL_AND_SECOND_LAST_OPPONENT_LEVEL`

## Fail-closed policy

Stage 8 refuses to emit a valid reference when any Stage-7 opponent lacks usable Stage-4 legal geometry. Missing one opponent can change which player is first or second-last, so partial defender geometry is diagnostic only.

Other blockers include frame mismatch, unresolved Stage 7, invalid direction, coordinate-frame mismatch, fewer than two opponents, unusable ball longitudinal geometry, and missing ball extent.

## Output

`offside_reference_state.json` schema: `stage8-offside-reference-1.0`.

A `VALID` result contains:

- attack direction
- legal-body policy
- complete opponent ranking
- second-last opponent coordinate/candidates
- ball goalward extent
- final reference source and X
- vertical plane equation
- diagnostics, quality and provenance

## Accuracy semantics

Implementation validity is not metric accuracy. Stage 8 v0.1.0 always preserves the fact that upstream Stage-4 legal extent is a landmark proxy and that independent offside metric accuracy is not established by merely running this deterministic logic.
