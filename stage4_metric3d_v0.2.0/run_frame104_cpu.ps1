$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:SAM3D_DINOV3_ROOT = Join-Path $repoRoot 'third_party\dinov3'
$workerPython = if ($env:SAM3D_PYTHON) { $env:SAM3D_PYTHON } else { Join-Path $repoRoot '.venv-sam3d-cpu\Scripts\python.exe' }
$stagePython = if ($env:STAGE4_PYTHON) { $env:STAGE4_PYTHON } else { 'C:\Users\Admin\anaconda3\envs\nckh-env\python.exe' }
$stage3 = Join-Path $repoRoot 'stage3_pose2d_v0.1\runs\stage3_real_1920\tracked_pose_2d_state.json'
$cameras = Join-Path $repoRoot 'stage_1_camera_v12\outputs\temporal_v13\batch_video\shot_ptz\optimized_camera_states'
$video = Join-Path $repoRoot 'stage_1_camera_v12\download.mp4'
$sam3dRoot = Join-Path $repoRoot 'third_party\sam-3d-body'
$checkpoint = Join-Path $repoRoot 'third_party\model.ckpt'
$mhrPath = Join-Path $repoRoot 'third_party\mhr_model.pt'
$cache = '.\runs\stage4_v05_frame104\sam3d_native_frame104.npz'
& $workerPython -u .\run_sam3d_body_worker.py --stage3-state $stage3 --camera-dir $cameras --video $video --sam3d-root $sam3dRoot --checkpoint $checkpoint --mhr-path $mhrPath --device cpu --cpu-threads 4 --person-batch-size 1 --inference-type body --frame-index 104 --output-cache $cache
if ($LASTEXITCODE -ne 0) { throw 'SAM3D inference failed' }
& $stagePython .\run_stage4.py sam3d-pitch-refined --stage3-state $stage3 --camera-dir $cameras --sam3d-cache $cache --selected-frame 104 --window-radius 0 --disable-temporal --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'Stage4 preflight failed' }
foreach ($backend in @('sam3d-direct', 'sam3d-pitch-refined')) {
    & $stagePython .\run_stage4.py $backend --stage3-state $stage3 --camera-dir $cameras --sam3d-cache $cache --selected-frame 104 --window-radius 0 --disable-temporal --output-dir ".\runs\stage4_v05_frame104\$backend"
    if ($LASTEXITCODE -ne 0) { throw "Stage4 failed: $backend" }
}
