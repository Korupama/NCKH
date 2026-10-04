# Stage 4 v0.5.1 local quickstart

Phase 0 first requires a real artifact manifest. Run the audit from this
directory with the included example (it will remain fail-closed until the
paths point to real artifacts):

```powershell
$env:PYTHONPATH = "."
$python = if ($env:STAGE4_PYTHON) { $env:STAGE4_PYTHON } else { "C:\Users\Admin\anaconda3\envs\nckh-env\python.exe" }
& $python tools\stage4_phase0_audit.py `
  --manifest docs\STAGE4_PHASE0_INPUT_MANIFEST.example.json `
  --tests-passed 72 --tests-failed 0
```

The report separates Stage-4-only model development from optional project
integration. Missing Stage-1/3 artifacts do not block independent Stage-4
backend work. To run a model-only benchmark, use benchmark RGB/crops, boxes,
camera metadata and labels with a licensed dataset adapter; do not feed Stage-1
or Stage-3 outputs as labels. The optional frame-104 integration route remains
fail-closed until its matching real artifacts are available. Its recovery
routes are documented in
`docs/STAGE4_PHASE0_BASELINE_REPORT.md`:

1. replay an existing native cache; or
2. regenerate only the Stage-4 SAM3D cache after the exact Stage-1/Stage-3
   inputs and frozen SAM3D assets are available.

For this CPU-only machine, reuse the verified frame-104 cache once the replay
route is ready:

```powershell
powershell -ExecutionPolicy Bypass -File ".\rerun_frame104_v051.ps1"
```

This replaces files in `runs/stage4_v051_frame104`, not older v0.5 outputs.
Read [the current report](V051_GROUND_FIRST_REPORT.md) for requirements and limitations.
The earlier full worker script `run_frame104_cpu.ps1` reruns inference and writes the
older v0.5-named output folder; it is not needed for the cached hotfix experiment.

## 1. Independent model-only benchmark lane

Prepare a licensed benchmark release and convert it to
`docs/STAGE4_MODEL_ONLY_MANIFEST.example.json`. The manifest must point to
real RGB images/crops, bboxes, benchmark camera states and
`metric-pose23-gt-1.0` labels. Validate it without loading Stage 1/3:

```powershell
$env:PYTHONPATH = "."
& $python -c "from stage4_metric3d.model_only import load_model_only_manifest; m=load_model_only_manifest('path\\to\\manifest.json'); print({'sha256': m.sha256, 'records': len(m.records)})"
```

Run the independent SAM3D direct baseline from the SAM3D runtime environment:

```powershell
& $python run_sam3d_model_only.py `
  --manifest "path\to\manifest.json" `
  --checkpoint "path\to\model.ckpt" `
  --mhr-path "path\to\mhr_model.pt" `
  --output ".\runs\model_only\sam3d_direct.npz" `
  --device auto
```

This writes the native output and a sibling `.metrics.json` report keyed by
benchmark `record_id`. The report contains paired `sam3d-direct` and
model-only `ground-first` ablation metrics; it does not claim the Stage-1/3
integration pipeline. Missing weights,
invalid inputs, or absent licensed benchmark data fail closed.

Re-evaluate an existing output without rerunning the model:

```powershell
& $python evaluate_model_only.py `
  --manifest "path\to\manifest.json" `
  --output ".\runs\model_only\sam3d_direct.npz" `
  --report ".\runs\model_only\sam3d_direct.evaluation.json"
```

The evaluator rechecks the manifest hash, every referenced input hash, record
order, output schema and MHR70 ordering before computing the paired metrics.

Compare multiple independent Stage-4 systems/variants on the same cohort:

```powershell
& $python compare_model_only_systems.py `
  --systems-manifest "path\to\STAGE4_MODEL_ONLY_SYSTEMS.json" `
  --report ".\runs\model_only\systems.evaluation.json"
```

Each system must declare backend, variant, coordinate scope, checkpoint hash,
output and (when available) repository/config provenance. The comparator
rejects cohort, manifest, output-schema and checkpoint mismatches.

## Historical v0.5 packaging instructions

The test count and CUDA setup below describe the original package build, not this
CPU-only installation. The current local Phase-0 suite records 52 passing tests.

## 1. Validate the patch without pretrained weights

```powershell
$env:PYTHONPATH = "."
pytest -q tests tests_v04 tests_v05
python validate_stage4_v050.py
```

Expected package-level result for this patch build: tests pass and synthetic validator `PASS`.

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
  --checkpoint "PATH\TO\sam3d_body\model.ckpt" `
  --mhr-path "PATH\TO\sam3d_body\assets\mhr_model.pt" `
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
