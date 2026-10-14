#!/usr/bin/env bash
# ==============================================================================
# Slurm Job Script: NCKH Pipeline Evaluation on NVIDIA H200 (Pyxis / Enroot)
# Submit with: sbatch slurm/eval_h200_pyxis.slurm
# ==============================================================================

#SBATCH --job-name=nckh_h200_pyxis
#SBATCH --output=slurm_logs/eval_%j_%x.out
#SBATCH --error=slurm_logs/eval_%j_%x.err
#SBATCH --partition=gpu               # Configured partition for GPU nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # Recommended: 16-32 vCPUs per H200 GPU
#SBATCH --mem=64G                     # Recommended system memory
#SBATCH --gres=gpu:1                  # Request 1 H200 GPU
#SBATCH --time=04:00:00               # Max runtime (hh:mm:ss)

# Pyxis/Enroot direct container image:
# Either local sqsh/image or docker registry image
#SBATCH --container-image=nckh-pipeline:h200
#SBATCH --container-mounts=$(pwd):/workspace/NCKH_SoICT,/dev/shm:/dev/shm
#SBATCH --container-workdir=/workspace/NCKH_SoICT

set -euo pipefail

mkdir -p slurm_logs

echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-LOCAL}"
echo "Node:          $(hostname)"
echo "Start Time:    $(date -u)"
echo "========================================================================"

nvidia-smi

echo "Verifying PyTorch CUDA & H200 detection inside container:"
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    print('Device Capability:', torch.cuda.get_device_capability(0))
"

echo "Starting Pipeline Evaluation at $(date -u)..."
python3 evaluate_cascaded_pipeline.py

echo "Evaluation completed at: $(date -u)"

