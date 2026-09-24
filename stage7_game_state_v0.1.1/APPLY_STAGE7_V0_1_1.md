# Apply Stage 7 v0.1.1

Overlay this package on `stage7_game_state_v0.1.0` or replace the old Stage 7 folder.

Main change: when an older Stage 1 `CameraState` does not contain `view.centre_ray_pitch_hit_m`, Stage 7 derives the centre-ray pitch hit from calibrated camera extrinsics (`R` and `C`/`position_meters`, or `R`+`t`). The explicit Stage 1 view field always has priority.

The patch does **not** auto-correct Stage 5 role/team labels. A known referee mislabel remains a blocker and should be fixed upstream.
