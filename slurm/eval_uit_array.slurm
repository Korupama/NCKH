#!/usr/bin/env bash
# Slurm Job Array: 3 parallel workers on HPC-UIT partition normal (MIG 2g.35gb)
# Usage: sbatch slurm/eval_uit_array.slurm

#SBATCH --job-name=uit_nckh_array
#SBATCH --output=/datastore/%u/slurm_logs/eval_array_%A_task_%a.out
#SBATCH --error=/datastore/%u/slurm_logs/eval_array_%A_task_%a.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1
#SBATCH --array=0-2%3

set -euo pipefail

USER_NAME="${USER:-hpc}"
DATASTORE_DIR="/datastore/${USER_NAME}"
PROJECT_ROOT="${DATASTORE_DIR}/NCKH_SoICT"
SIF_FILE="${DATASTORE_DIR}/nckh_h200.sif"

# Protect the 10GB /home limit by redirecting Apptainer cache to /datastore
export APPTAINER_CACHEDIR="${DATASTORE_DIR}/.apptainer_cache"
export APPTAINER_TMPDIR="${DATASTORE_DIR}/.apptainer_tmp"
export SINGULARITY_CACHEDIR="${DATASTORE_DIR}/.apptainer_cache"
export SINGULARITY_TMPDIR="${DATASTORE_DIR}/.apptainer_tmp"

mkdir -p "${DATASTORE_DIR}/slurm_logs"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "========================================================================"
echo "HPC-UIT Slurm Job Array Worker"
echo "Array Job ID:     ${SLURM_ARRAY_JOB_ID:-LOCAL}"
echo "Worker Task ID:   ${TASK_ID} (Chunk ${TASK_ID} of 3)"
echo "Node:             $(hostname)"
echo "User:             ${USER_NAME}"
echo "Start Time:       $(date -u)"
echo "========================================================================"

# Purge and load container runtime
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

echo "Checking GPU MIG allocation:"
nvidia-smi

export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAM3D_PERSON_BATCH_SIZE=16
EVAL_MODE="${EVAL_MODE:-bookmarks}"

echo "Evaluation mode: ${EVAL_MODE}"
if [ "${EVAL_MODE}" = "cascaded" ]; then
    echo "Launching Cascaded Benchmark Worker ${TASK_ID} (Chunk ${TASK_ID}/3)..."
    $CONTAINER_CLI exec --nv ${BIND_OPTS} "${SIF_FILE}" python -u evaluate_cascaded_pipeline.py \
        --num-chunks 3 \
        --chunk-index "${TASK_ID}"
else
    echo "Launching Video Bookmark Pipeline Worker ${TASK_ID} (Chunk ${TASK_ID}/3)..."
    $CONTAINER_CLI exec --nv ${BIND_OPTS} "${SIF_FILE}" python -u video_bookmark/run_accelerated_pipeline.py \
        --video video_bookmark/mixed.mp4 \
        --frames video_bookmark/mixed.txt \
        --output-root "${PROJECT_ROOT}/stage9_offside_position_v0.1.0/outputs/worker_${TASK_ID}" \
        --device cuda \
        --num-chunks 3 \
        --chunk-index "${TASK_ID}" \
        --sam3d-batch-size 16
fi

echo "========================================================================"
echo "Worker ${TASK_ID} finished at: $(date -u)"
echo "========================================================================"


