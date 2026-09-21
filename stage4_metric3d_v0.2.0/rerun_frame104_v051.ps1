$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$stagePython = 'D:\anaconda3\envs\stage4-rtmw3d\python.exe'
$stage3 = 'D:\NCKH\stage3_pose2d_v0.1\runs\stage3_real_1920\tracked_pose_2d_state.json'
$cameras = 'D:\NCKH\stage_1_camera_v12\outputs\temporal_v13\batch_video\shot_ptz\optimized_camera_states'
$cache = '.\runs\stage4_v05_frame104\sam3d_native_frame104.npz'
if (-not (Test-Path -LiteralPath $cache)) { throw 'Existing frame104 SAM3D cache is required; this script does not run inference.' }
& $stagePython .\run_stage4.py sam3d-pitch-refined --stage3-state $stage3 --camera-dir $cameras --sam3d-cache $cache --selected-frame 104 --window-radius 0 --disable-temporal --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'Stage4 preflight failed' }
foreach ($backend in @('sam3d-direct', 'sam3d-pitch-refined')) {
    & $stagePython .\run_stage4.py $backend --stage3-state $stage3 --camera-dir $cameras --sam3d-cache $cache --selected-frame 104 --window-radius 0 --disable-temporal --output-dir ".\runs\stage4_v051_frame104\$backend"
    if ($LASTEXITCODE -ne 0) { throw "Stage4 execution failed: $backend" }
}
# Exit code reports execution, not scientific validity; inspect stage4_quality_report.json.
