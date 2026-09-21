param(
    [string]$EnvName = "stage4-rtmw3d",
    [string]$MMPoseRoot = "D:\NCKH\third_party\mmpose",
    [ValidateSet("cpu", "cu121", "cu118")]
    [string]$TorchBuild = "cpu"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/8] Creating conda environment: $EnvName"
conda create -n $EnvName python=3.10 -y

Write-Host "[2/8] Upgrading pip and pinning NumPy"
conda run -n $EnvName python -m pip install --upgrade pip setuptools wheel
conda run -n $EnvName python -m pip install "numpy==1.26.4"

if ($TorchBuild -eq "cpu") {
    $TorchIndex = "https://download.pytorch.org/whl/cpu"
    $MMCVIndex = "https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html"
} elseif ($TorchBuild -eq "cu121") {
    $TorchIndex = "https://download.pytorch.org/whl/cu121"
    $MMCVIndex = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"
} else {
    $TorchIndex = "https://download.pytorch.org/whl/cu118"
    $MMCVIndex = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html"
}

Write-Host "[3/8] Installing PyTorch 2.1.2 / torchvision 0.16.2 ($TorchBuild)"
conda run -n $EnvName python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url $TorchIndex

Write-Host "[4/8] Installing MMEngine + MMCV 2.1.0"
conda run -n $EnvName python -m pip install "mmengine==0.10.7" "openmim>=0.3.9"
conda run -n $EnvName python -m pip install "mmcv==2.1.0" -f $MMCVIndex

Write-Host "[5/8] Cloning official MMPose v1.3.2"
$Parent = Split-Path -Parent $MMPoseRoot
New-Item -ItemType Directory -Force -Path $Parent | Out-Null
if (-not (Test-Path $MMPoseRoot)) {
    git clone --branch v1.3.2 --depth 1 https://github.com/open-mmlab/mmpose.git $MMPoseRoot
} else {
    Write-Host "MMPose root already exists: $MMPoseRoot"
}

Write-Host "[6/8] Installing MMPose source in editable mode"
conda run -n $EnvName python -m pip install -e $MMPoseRoot

Write-Host "[7/8] Installing Stage 4 package in the RTMW3D environment"
conda run -n $EnvName python -m pip install -e $PSScriptRoot

Write-Host "[8/8] Verifying environment"
$env:MMPOSE_ROOT = $MMPoseRoot
conda run -n $EnvName python "$PSScriptRoot\verify_rtmw3d_env.py" --mmpose-root $MMPoseRoot

Write-Host ""
Write-Host "Environment ready. Activate it with:"
Write-Host "  conda activate $EnvName"
Write-Host "Then run the worker with:"
Write-Host "  python run_rtmw3d_worker.py --mmpose-root `"$MMPoseRoot`" ..."
