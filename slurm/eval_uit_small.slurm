#!/usr/bin/env bash
# Slurm Job Script: Evaluation on HPC-UIT partition small (MIG 1g.18gb)
# Usage: sbatch slurm/eval_uit_small.slurm

#SBATCH --job-name=uit_nckh_small
#SBATCH --output=/datastore/%u/slurm_logs/eval_small_%j.out
#SBATCH --error=/datastore/%u/slurm_logs/eval_small_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

USER_NAME="${USER:-hpc}"
DATASTORE_DIR="/datastore/${USER_NAME}"
PROJECT_ROOT="${DATASTORE_DIR}/NCKH_SoICT"
SIF_FILE="${DATASTORE_DIR}/nckh_h200.sif"

export APPTAINER_CACHEDIR="${DATASTORE_DIR}/.apptainer_cache"
export APPTAINER_TMPDIR="${DATASTORE_DIR}/.apptainer_tmp"
export SINGULARITY_CACHEDIR="${DATASTORE_DIR}/.apptainer_cache"
export SINGULARITY_TMPDIR="${DATASTORE_DIR}/.apptainer_tmp"

mkdir -p "${DATASTORE_DIR}/slurm_logs"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "========================================================================"
echo "HPC-UIT Slurm Job (Partition: small - H200 1g.18gb)"
echo "Job ID:        ${SLURM_JOB_ID:-LOCAL}"
echo "Node:          $(hostname)"
echo "User:          ${USER_NAME}"
echo "========================================================================"

if command -v module &>/dev/null; then
    module purge || true
    module load apptainer/latest 2>/dev/null || module load singularity/latest 2>/dev/null || true
fi

CONTAINER_CLI="apptainer"
if ! command -v apptainer &>/dev/null; then
    CONTAINER_CLI="singularity"
fi

if [ ! -f "$SIF_FILE" ]; then
    if [ -f "${PROJECT_ROOT}/nckh_h200.sif" ]; then
        SIF_FILE="${PROJECT_ROOT}/nckh_h200.sif"
    else
        echo "[ERROR] Container $SIF_FILE not found in /datastore/${USER_NAME}/!"
        exit 1
    fi
fi

BIND_OPTS="--bind ${DATASTORE_DIR}:${DATASTORE_DIR} --bind /dev/shm:/dev/shm"
if [ -d "/datastore/shared" ]; then
    BIND_OPTS="${BIND_OPTS} --bind /datastore/shared:/datastore/shared:ro"
fi

cd "${PROJECT_ROOT}"

nvidia-smi

export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# In small partition (16GB usable VRAM), use person batch size 8 for SAM-3D
export SAM3D_PERSON_BATCH_SIZE=8

$CONTAINER_CLI exec --nv ${BIND_OPTS} "${SIF_FILE}" python -u evaluate_cascaded_pipeline.py

echo "Finished at: $(date -u)"

