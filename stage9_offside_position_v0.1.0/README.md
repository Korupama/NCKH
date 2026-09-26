# Stage 9 v0.1.0 — visualization hotfix

**Hotfix policy:** Stage 9 no longer fabricates missing image-space boxes. A player without a Stage-3 bbox or a Stage-4+Stage-1 projected bbox stays in the side panel only. Keypoints are OFF by default; only the last and second-last opponents are drawn by default. The web UI reports the exact background frame source so baked-in upstream overlays can be diagnosed.

For a clean visualization, explicitly pass the raw replay:

```powershell
.\run_project_frame104_demo.ps1 -Video "D:\path\to\raw_replay.mp4"
```

or a clean selected-frame image:

```powershell
.\run_project_frame104_demo.ps1 -Image "D:\path\to\frame_104.jpg"
```

---

# Stage 9 — Offside Position + Local Web Demo v0.1.0

Stage 9 is the final **offside-position** visualization layer for the reduced replay pipeline. It consumes Stage 4 body geometry, Stage 7 game-state sets, and Stage 8 reference geometry, then flags attacking players as `ONSIDE` or `OFFSIDE_POSITION`.

This version is intentionally **demo-first and best-effort**. It can ignore upstream `BLOCKED`, `DEGRADED`, `UNRESOLVED`, and uncertainty statuses when enough numeric geometry is still present. The goal is to make the current pipeline visually inspectable. It does **not** claim a validated referee decision and does not decide offside offence semantics.

## Classification

Use the canonical goalward coordinate:

```text
q = s * X_world_m
```

For each attacker other than the Stage-7 toucher:

```text
OFFSIDE_POSITION if:
    attacker_goalward_q > reference_goalward_q + epsilon
    AND attacker_goalward_q > 0  # some legal landmark is in opponents' half
else ONSIDE
```

The toucher is shown as `PASSER` and excluded from classification. Stage-4 legal Pose23 landmarks are preferred; when no legal landmark is available but `root_world_m` exists, the demo uses that root as a visible fallback. Missing geometry is still rendered as `ON?` rather than blocking the page.

## Install

```powershell
cd D:\NCKH\stage9_offside_position_v0.1.0
pip install -r requirements.txt
python -m pytest -q
```

## Run bundled web demo

```powershell
python web_demo.py --demo
```

Open:

```text
http://127.0.0.1:8099
```

The page supports toggling overlay, reference line, defenders, keypoints, and labels.

## Run Stage 9 JSON classification

```powershell
python run_stage9.py --stage4 ".\examples\stage4.json" --stage7 ".\examples\stage7.json" --stage8 ".\examples\stage8.json" --output ".\outputs\example\offside_position_state.json" --stage1 ".\examples\stage1.json" --stage3 ".\examples\stage3.json" --image ".\examples\frame104_demo.jpg" --overlay ".\outputs\example\stage9_overlay.jpg"
```

## Run web demo with project artifacts

```powershell
python web_demo.py --stage4 "D:\NCKH\stage4_metric3d_v0.2.0\runs\stage4_v051_frame104\sam3d-pitch-refined\stage4_downstream_handoff.json" --stage6 "D:\NCKH\stage6_ball_localization_v0.4.4\outputs\replay_t0_104_contact_v051\stage6_downstream_handoff.json" --stage7 "D:\NCKH\stage7_game_state_v0.1.0\outputs\project_replay_104\game_state_context.json" --stage8 "D:\NCKH\stage8_offside_reference_v0.1.0\outputs\project_replay_104\offside_reference_state.json" --stage1 "D:\NCKH\stage_1_camera_v12\outputs\temporal_v13\batch_video\shot_ptz\optimized_camera_states\camera_state_00000104_view.json" --stage3 "D:\NCKH\stage3_pose2d_v0.1\runs\stage3_real_1920\tracked_pose_2d_state.json" --video "<YOUR_REPLAY_VIDEO.mp4>"
```

If Stage 7/8 are still marked unresolved but contain sets/reference geometry, Stage 9 demo continues. That bypass is deliberate in this version.


### If Stage 8 is still blocked

`--stage8` is optional in best-effort mode. Stage 9 can recompute a demo reference from Stage 4 opponent geometry + Stage 6 ball geometry + Stage 7 opponent set:

```powershell
python web_demo.py --stage4 "<stage4_downstream_handoff.json>" --stage6 "<stage6_downstream_handoff.json>" --stage7 "<game_state_context.json>" --stage1 "<camera_state.json>" --stage3 "<tracked_pose_2d_state.json>"
```

For the frame-104 workspace, `run_project_frame104_demo.ps1` automatically uses Stage 8 when the artifact exists and otherwise activates this fallback.
