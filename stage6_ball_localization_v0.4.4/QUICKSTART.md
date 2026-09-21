# Quickstart — Stage 6 v0.4.0

## 1. Confirm that the correct package is being imported

```powershell
cd D:\NCKH\stage6_ball_localization_v0.4.0
python benchmark_ball.py provenance
```

It should report:

```text
package_version = 0.4.0
stage6_version   = stage6-ball-localization-0.4.0
```

The printed `package_root` should point inside the v0.4.0 folder you are running.

## 2. Fastest next step: refine the existing real replay state

If you already have the previous `ball_trajectory_state.json`, do not run YOLO again:

```powershell
python refine_ball_trajectory.py `
  --state-json "D:\NCKH\previous_stage6_output\ball_trajectory_state.json" `
  --stage1-root "D:\NCKH\stage_1_camera_v12" `
  --output-dir ".\outputs\temporal_v040"
```

Inspect:

```text
outputs\temporal_v040\ball_trajectory_state.json
outputs\temporal_v040\ball_trajectory_minimap.png   (when the source video is reachable)
outputs\temporal_v040\selected_frame_ball.png       (when the source video is reachable)
```

## 3. Full run with detector + tracker + temporal geometry

```powershell
python run_stage6_ball.py `
  --stage1-root "D:\NCKH\stage_1_camera_v12" `
  --output-dir ".\outputs\stage6_v040" `
  --provider yolo `
  --weights ".\weights\yolo-sn-ball-opt.pt" `
  --tracker viterbi `
  --localization-mode temporal-3d `
  --conf-floor 0.05 `
  --top-k 10 `
  --imgsz 1920 `
  --device cpu
```

## 4. Main temporal controls

Defaults are intentionally conservative:

```text
--temporal-diameter-second-diff-weight 8.0
--temporal-diameter-first-diff-weight  0.10
--temporal-max-gap-frames               3
--temporal-range-sigma-fraction         0.08
--temporal-position-second-diff-sigma-m 0.75
--temporal-position-third-diff-sigma-m  0.50
--temporal-pitch-margin-m               6.0
--temporal-max-height-m                 30.0
```

Do not tune these on the SoccerNet-v3D 810-row CSV as if those rows were consecutive frames; they are not a temporal sequence.

## 5. Existing v0.3 benchmarks still work

`run-2d`, `reevaluate-2d`, `run-3d-oracle`, and `run-3d-e2e` are preserved. Their reports now include runtime provenance so a stale imported package is visible immediately.
