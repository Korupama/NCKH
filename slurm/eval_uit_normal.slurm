#!/usr/bin/env bash
# Slurm Job Script: Evaluation on HPC-UIT partition normal (MIG 2g.35gb)
# Usage: sbatch slurm/eval_uit_normal.slurm

#SBATCH --job-name=uit_nckh_normal
#SBATCH --output=/datastore/%u/slurm_logs/eval_normal_%j.out
#SBATCH --error=/datastore/%u/slurm_logs/eval_normal_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

USER_NAME="${USER:-hpc}"
DATASTORE_DIR="/datastore/${USER_NAME}"
PROJECT_ROOT="${DATASTORE_DIR}/NCKH_SoICT"
SIF_FILE="${DATASTORE_DIR}/nckh_h200.sif"

# CRITICAL: Relocate Apptainer cache to /datastore to protect the 10GB /home limit!
export APPTAINER_CACHEDIR="${DATASTORE_DIR}/.apptainer_cache"
export APPTAINER_TMPDIR="${DATASTORE_DIR}/.apptainer_tmp"
export SINGULARITY_CACHEDIR="${DATASTORE_DIR}/.apptainer_cache"
export SINGULARITY_TMPDIR="${DATASTORE_DIR}/.apptainer_tmp"

mkdir -p "${DATASTORE_DIR}/slurm_logs"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "========================================================================"
echo "HPC-UIT Slurm Job (Partition: normal - H200 2g.35gb)"
echo "Job ID:        ${SLURM_JOB_ID:-LOCAL}"
echo "Node:          $(hostname)"
echo "User:          ${USER_NAME}"
echo "Start Time:    $(date -u)"
echo "Working Dir:   ${PROJECT_ROOT}"
echo "Container:     ${SIF_FILE}"
echo "========================================================================"

# Purge host modules and load container runtime
if command -v module &>/dev/null; then
    module purge || true
    module load apptainer/latest 2>/dev/null || module load singularity/latest 2>/dev/null || true
fi

if command -v apptainer &>/dev/null; then
    CONTAINER_CLI="apptainer"
elif command -v singularity &>/dev/null; then
    CONTAINER_CLI="singularity"
else
    echo "[ERROR] Neither apptainer nor singularity found on compute node!"
    exit 1
fi

echo "Using container runtime: $CONTAINER_CLI ($(which $CONTAINER_CLI))"
echo "Checking GPU hardware allocated via MIG:"
nvidia-smi

if [ ! -f "$SIF_FILE" ]; then
    if [ -f "${PROJECT_ROOT}/nckh_h200.sif" ]; then
        SIF_FILE="${PROJECT_ROOT}/nckh_h200.sif"
    else
        echo "[ERROR] Container $SIF_FILE not found in /datastore/${USER_NAME}/!"
        echo "Please build or copy nckh_h200.sif to /datastore/${USER_NAME}/ (DO NOT put in /home)."
        exit 1
    fi
fi

# Define BIND mounts (/datastore is required, /dev/shm for PyTorch IPC)
BIND_OPTS="--bind ${DATASTORE_DIR}:${DATASTORE_DIR} --bind /dev/shm:/dev/shm"
if [ -d "/datastore/shared" ]; then
    BIND_OPTS="${BIND_OPTS} --bind /datastore/shared:/datastore/shared:ro"
fi

cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "Executing Accelerated Pipeline inside H200 Container..."
echo "========================================================================"

# Hopper H200 Native Optimization Flags
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAM3D_PERSON_BATCH_SIZE=16

# Run cascaded evaluation or bookmark evaluation
# (Change the command below according to your experiment target)
$CONTAINER_CLI exec --nv ${BIND_OPTS} "${SIF_FILE}" python -u evaluate_cascaded_pipeline.py

echo "========================================================================"
echo "Job ${SLURM_JOB_ID:-LOCAL} finished at: $(date -u)"
echo "========================================================================"

