#!/usr/bin/env bash
# Slurm Job Script: Direct Conda execution on HPC-UIT (No Docker/SIF required)
# Partition: small (MIG 1g.18gb) - Change to normal if running full workload
# Usage: sbatch slurm/eval_uit_conda.slurm

#SBATCH --job-name=uit_conda_eval
#SBATCH --output=/datastore/thangcpd/slurm_logs/eval_%j.out
#SBATCH --error=/datastore/thangcpd/slurm_logs/eval_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

mkdir -p /datastore/thangcpd/slurm_logs

# Kich hoat moi truong conda env_ai da cai san tren server lab
if [ -f "/datastore/thangcpd/miniconda3/bin/activate" ]; then
    source "/datastore/thangcpd/miniconda3/bin/activate" /datastore/thangcpd/env_ai 2>/dev/null || \
    source "/datastore/thangcpd/miniconda3/bin/activate" env_ai 2>/dev/null || true
fi

# Tim thu muc code tren datastore
PROJECT_DIR=""
for candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(pwd)" \
    "/datastore/thangcpd/KLTN_HungHuynh" \
    "/datastore/thangcpd/NCKH_HungHuynh" \
    "/datastore/thangcpd/NCKH_TonDaiQuoc" \
    "/datastore/thangcpd/NCKH_SoICT" \
    "/datastore/thangcpd/NCKH_TonDaiQuoc/NCKH_SoICT"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/evaluate_cascaded_pipeline.py" ]; then
        PROJECT_DIR="${candidate}"
        break
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    echo "Khong tim thay thu muc code tren /datastore/thangcpd!"
    exit 1
fi

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/stage_1_camera_v12:${PROJECT_DIR}/stage2_sst_rtmw_v1.3:${PROJECT_DIR}/stage3_pose2d_v0.1:${PROJECT_DIR}/stage5_team_affiliation_v0.1.0:${PROJECT_DIR}/stage6_ball_localization_v0.4.4:${PROJECT_DIR}/stage7_game_state_v0.1.0:${PROJECT_DIR}/stage8_offside_reference_v0.1.0:${PROJECT_DIR}/stage9_offside_position_v0.1.0:${PYTHONPATH:-}"

echo "Working directory: $(pwd)"
echo "Python interpreter: $(which python)"
nvidia-smi

export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAM3D_PERSON_BATCH_SIZE=16

EVAL_MODE="${EVAL_MODE:-bookmarks}"
echo "Evaluation mode: ${EVAL_MODE}"

if [ "${EVAL_MODE}" = "cascaded" ]; then
    echo "Running Cascaded Benchmark (Stages 2 -> 5 -> 7 -> 8)..."
    python -u evaluate_cascaded_pipeline.py
else
    echo "Running Accelerated 9-Stage Video Pipeline..."
    VIDEO_FILE="${VIDEO_FILE:-video_bookmark/mixed.mp4}"
    BOOKMARKS_FILE="${BOOKMARKS_FILE:-video_bookmark/mixed.txt}"
    OUTPUT_DIR="${OUTPUT_DIR:-/datastore/thangcpd/outputs/run_${SLURM_JOB_ID:-manual}}"
    mkdir -p "${OUTPUT_DIR}"

    python -u video_bookmark/run_accelerated_pipeline.py \
        --video "${VIDEO_FILE}" \
        --frames "${BOOKMARKS_FILE}" \
        --output-root "${OUTPUT_DIR}" \
        --device cuda \
        --sam3d-batch-size 16
fi

