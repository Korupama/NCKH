# Stage 7 Specification — Game-State Context

## Responsibility

Resolve the game-state semantics needed by Stage 8 from already-estimated upstream states.

### In scope

1. Resolve Stage 6 `contact_track_id` against Stage 5 tracks.
2. Resolve attacking team from the toucher's Stage 5 team affiliation.
3. Resolve attack direction from Stage 1 centre-ray pitch-half cue.
4. Build attacker and opponent sets.
5. Exclude referees/officials.
6. Emit deterministic validity diagnostics and invariants.

### Out of scope

- second-last opponent
- ball-vs-defender reference maximum
- legal-body extremum
- offside-position classification
- offence semantics

## Invariants

For a `VALID` context:

- toucher ∈ attackers
- attackers ∩ opponents = ∅
- referees ∉ attackers ∪ opponents
- all active known-team non-referees are partitioned into attackers/opponents
- `s ∈ {-1,+1}`
- `attacking_team_id = team(toucher)`

## Evaluation

### Oracle-input logic

Target = 100%. Any failure is an implementation defect.

### End-to-end

Use predicted Stage 1/5/6 artifacts. A miss can be upstream-induced, so report both component outcome and Stage 7 logic diagnostics.


## Integration rule added in v0.1.1

If `centre_ray_pitch_hit_m` is absent but Stage 1 exposes calibrated extrinsics, Stage 7 may derive the optical-axis/pitch intersection from the camera centre and rotation. Explicit Stage 1 view metadata always takes precedence. Derived geometry must be provenance-tagged and camera quality remains diagnostic. Upstream role/team mistakes are not corrected inside Stage 7.
