# Stage 4 v0.5.1 local quickstart

For this CPU-only machine, reuse the verified frame-104 cache:

```powershell
powershell -ExecutionPolicy Bypass -File "D:\NCKH\stage4_metric3d_v0.2.0\rerun_frame104_v051.ps1"
```

This replaces files in `runs/stage4_v051_frame104`, not older v0.5 outputs.
Read [the current report](V051_GROUND_FIRST_REPORT.md) for requirements and limitations.
The earlier full worker script `run_frame104_cpu.ps1` reruns inference and writes the
older v0.5-named output folder; it is not needed for the cached hotfix experiment.

## Historical v0.5 packaging instructions

The test count and CUDA setup below describe the original package build, not this
CPU-only installation. The subsequent local v0.5.1 test run recorded 48 passing tests.

## 1. Validate the patch without pretrained weights

```powershell
$env:PYTHONPATH = "."
pytest -q tests tests_v04 tests_v05
python validate_stage4_v050.py
```

Expected package-level result for this patch build: `39 passed` and synthetic validator `PASS`.

## 2. SAM 3D Body environment

Keep SAM 3D Body in its own CUDA environment. Stage 4 uses the official DINOv3 checkpoint by default:

```text
pretrained_models/
  sam3d_body/
    model.ckpt
    model_config.yaml
    assets/
      mhr_model.pt
```

The Stage-4 host environment does not need the SAM3D CUDA dependency stack.

## 3. Build a native MHR70 SAM3D cache

Run this command using the Python interpreter from the SAM3D environment:

```powershell
<PATH_TO_SAM3D_PYTHON> run_sam3d_body_worker.py `
  --stage3-state "PATH\TO\tracked_pose_2d_state.json" `
  --camera-dir "PATH\TO\optimized_camera_states" `
  --video "PATH\TO\replay.mp4" `
  --checkpoint "D:\NCKH\pretrained_models\sam3d_body\model.ckpt" `
  --mhr-path "D:\NCKH\pretrained_models\sam3d_body\assets\mhr_model.pt" `
  --output-cache ".\runs\stage4_v05\sam3d_native.npz"
```

The worker supplies Stage-1 `K` to SAM3D and saves:

```text
pred_keypoints_2d      (T,N,70,2)
pred_keypoints_3d      (T,N,70,3)
pred_cam_t             (T,N,3)
focal_length           (T,N)
MHR70 joint semantics
```

It fails instead of silently truncating/reordering if the installed SAM3D release no longer matches official MHR70.

## 4. Preflight v0.5

```powershell
python run_stage4.py sam3d-pitch-refined `
  --stage3-state "PATH\TO\tracked_pose_2d_state.json" `
  --camera-dir "PATH\TO\optimized_camera_states" `
  --sam3d-cache ".\runs\stage4_v05\sam3d_native.npz" `
  --preflight-only
```

Preflight checks:

- Stage-3 track ordering and selected-frame coverage;
- Stage-1 camera timeline coverage;
- native MHR70 cache shape/schema;
- verified SAM3D MHR70 ↔ RTMW Pose23 mapping;
- SAM3D camera-space convention by reprojecting `J_rel + pred_cam_t` with Stage-1 `K`;
- selected-frame ground-anchor availability.

## 5. Run v0.5 production candidate

```powershell
python run_stage4.py sam3d-pitch-refined `
  --stage3-state "PATH\TO\tracked_pose_2d_state.json" `
  --camera-dir "PATH\TO\optimized_camera_states" `
  --sam3d-cache ".\runs\stage4_v05\sam3d_native.npz" `
  --output-dir ".\runs\stage4_v05"
```

The optimizer changes only the global camera-space translation of each SAM3D skeleton. It uses RTMW reprojection, pitch ground anchors, the original SAM3D `pred_cam_t` prior, and a world-space temporal second-difference term.

## 6. Run direct-SAM3D ablation

```powershell
python run_stage4.py sam3d-direct `
  --stage3-state "PATH\TO\tracked_pose_2d_state.json" `
  --camera-dir "PATH\TO\optimized_camera_states" `
  --sam3d-cache ".\runs\stage4_v05\sam3d_native.npz" `
  --output-dir ".\runs\stage4_v05_sam3d_direct"
```

This keeps `pred_cam_t` unchanged, which gives a clean ablation against pitch-constrained refinement.

## 7. Retained baselines

```powershell
python run_stage4.py fixed-height-v03 --help
python run_stage4.py field-converter-tcn --help
python run_stage4_legacy3d.py --help
```

The Field Converter backend remains optional. Do not fabricate its missing training `normalization_stats.npz`; v0.5 does not require it.
