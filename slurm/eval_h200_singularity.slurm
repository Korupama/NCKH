#!/usr/bin/env bash
# ==============================================================================
# Slurm Job Script: NCKH Pipeline Evaluation on NVIDIA H200 (Singularity/Apptainer)
# Submit with: sbatch slurm/eval_h200_singularity.slurm
# ==============================================================================

#SBATCH --job-name=nckh_h200_eval
#SBATCH --output=slurm_logs/eval_%j_%x.out
#SBATCH --error=slurm_logs/eval_%j_%x.err
#SBATCH --partition=gpu               # Configured partition for GPU nodes (e.g., gpu, h200, hopper)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # Recommended: 16-32 vCPUs per H200 GPU
#SBATCH --mem=64G                     # Recommended system memory (H200 has 141GB VRAM)
#SBATCH --gres=gpu:1                  # Request 1 H200 GPU (or gpu:h200:1 depending on cluster config)
#SBATCH --time=04:00:00               # Max runtime (hh:mm:ss)

set -euo pipefail

# Create output log directory
mkdir -p slurm_logs

echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-LOCAL}"
echo "Node:          $(hostname)"
echo "Start Time:    $(date -u)"
echo "Directory:     $(pwd)"
echo "========================================================================"

# Purge any conflicting host modules and load container runtime if modularized
if command -v module &>/dev/null; then
    module purge || true
    module load apptainer/latest 2>/dev/null || module load singularity/latest 2>/dev/null || true
fi

# Detect Apptainer or Singularity
if command -v apptainer &>/dev/null; then
    CONTAINER_CLI="apptainer"
elif command -v singularity &>/dev/null; then
    CONTAINER_CLI="singularity"
else
    echo "[ERROR] Neither apptainer nor singularity found on this compute node!"
    exit 1
fi

echo "Using container runtime: $CONTAINER_CLI ($(which $CONTAINER_CLI))"

# Check GPU hardware and Hopper H200 architecture
echo "Checking GPU hardware:"
nvidia-smi

# Set paths
PROJECT_ROOT="$(pwd)"
SIF_FILE="${PROJECT_ROOT}/nckh_h200.sif"

if [ ! -f "$SIF_FILE" ]; then
    echo "[WARNING] $SIF_FILE not found in project root!"
    echo "Checking in ~/containers/ or parent directory..."
    if [ -f "$HOME/containers/nckh_h200.sif" ]; then
        SIF_FILE="$HOME/containers/nckh_h200.sif"
    elif [ -f "$HOME/nckh_h200.sif" ]; then
        SIF_FILE="$HOME/nckh_h200.sif"
    else
        echo "[ERROR] Could not locate nckh_h200.sif. Please build or transfer it first!"
        exit 1
    fi
fi
echo "Using SIF container: $SIF_FILE"

# Optimize PyTorch & CUDA environment inside container
export SINGULARITYENV_TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
export SINGULARITYENV_PYTHONUNBUFFERED=1
export APPTAINERENV_TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export APPTAINERENV_PYTHONUNBUFFERED=1

# Run sanity check inside container: Verify PyTorch detects NVIDIA H200
echo "========================================================================"
echo "Verifying PyTorch CUDA & H200 detection inside container:"
$CONTAINER_CLI exec --nv \
    --bind "${PROJECT_ROOT}:/workspace/NCKH_SoICT" \
    --bind /dev/shm:/dev/shm \
    "$SIF_FILE" \
    python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    print('Device Capability:', torch.cuda.get_device_capability(0))
"
echo "========================================================================"

# Run the accelerated evaluation / cascaded benchmark
echo "Starting Pipeline Evaluation at $(date -u)..."

# Example 1: Run cascaded evaluation (Stage 2 -> Stage 5 -> Stage 7 -> Stage 8)
$CONTAINER_CLI exec --nv \
    --bind "${PROJECT_ROOT}:/workspace/NCKH_SoICT" \
    --bind /dev/shm:/dev/shm \
    "$SIF_FILE" \
    python3 /workspace/NCKH_SoICT/evaluate_cascaded_pipeline.py

# Example 2 (optional): Run bookmarked video frames if mixed.mp4 exists
if [ -f "video_bookmark/mixed.mp4" ]; then
    echo "Running video bookmark evaluation..."
    $CONTAINER_CLI exec --nv \
        --bind "${PROJECT_ROOT}:/workspace/NCKH_SoICT" \
        --bind /dev/shm:/dev/shm \
        "$SIF_FILE" \
        python3 /workspace/NCKH_SoICT/video_bookmark/run_accelerated_pipeline.py \
            --video video_bookmark/mixed.mp4 \
            --frames video_bookmark/mixed.txt
fi

echo "========================================================================"
echo "Evaluation job completed successfully at: $(date -u)"
echo "========================================================================"

