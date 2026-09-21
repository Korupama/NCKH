# Quickstart

```powershell
cd D:\NCKH\stage5_team_affiliation_v0.1.0
pip install -e .
python -m pytest -q
python validate_stage5_v010.py
```

Run on project replay:

```powershell
python run_stage5.py `
  --stage3-state "PATH\tracked_pose_2d_state.json" `
  --video "D:\NCKH\stage_1_camera_v12\download.mp4" `
  --output-dir ".\outputs\project_replay"
```

Inspect:
- `team_affiliation_state.json`
- `stage5_downstream_handoff.json`
- `selected_frame_team_affiliation.png`

Do not interpret `TEAM_0/TEAM_1` as attacking/defending. Stage 7 owns that semantic mapping.

## Third-party SoccerNet-GSR benchmark

```powershell
python benchmark_soccernet_gsr.py inspect --gsr-root "D:\GSR" --split valid

python benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" `
  --split valid `
  --method bbox-color `
  --output-dir ".\benchmark_results\gsr_valid_bbox_color"
```

For the pose-guided B1 protocol, provide oracle-track pose caches with `--pose-cache-root`. See `BENCHMARK.md`.
