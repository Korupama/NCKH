#!/usr/bin/env bash
# ==============================================================================
# Slurm Array Job: Multi-Worker Data-Parallel Evaluation on NVIDIA H200 (141GB)
#
# This script divides the evaluation workload (windows or frames) across
# parallel workers, exploiting the 141GB VRAM of H200 for linear speedup.
#
# Submit with:
#   sbatch slurm/eval_h200_array.slurm
# Or customize array range (e.g. 4 workers):
#   sbatch --array=0-3 slurm/eval_h200_array.slurm
# ==============================================================================

#SBATCH --job-name=nckh_h200_array
#SBATCH --output=slurm_logs/eval_array_%A_task_%a.out
#SBATCH --error=slurm_logs/eval_array_%A_task_%a.err
#SBATCH --partition=gpu               # Configured GPU partition (e.g. gpu, h200)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # 8 vCPUs per worker
#SBATCH --mem=32G                     # 32GB RAM per worker
#SBATCH --gres=gpu:1                  # 1 GPU (or share GPU if multiple array tasks on same node)
#SBATCH --time=02:00:00               # Max runtime per chunk
#SBATCH --array=0-3                   # Default: 4 parallel workers (tasks 0, 1, 2, 3)

set -euo pipefail

mkdir -p slurm_logs

echo "========================================================================"
echo "Array Job ID:     ${SLURM_ARRAY_JOB_ID:-LOCAL}"
echo "Task ID:          ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:             $(hostname)"
echo "Start Time:       $(date -u)"
echo "========================================================================"

# Container runtime detection
if command -v module &>/dev/null; then
    module purge || true
    module load apptainer/latest 2>/dev/null || module load singularity/latest 2>/dev/null || true
fi

CONTAINER_CLI="singularity"
if command -v apptainer &>/dev/null; then
    CONTAINER_CLI="apptainer"
fi

PROJECT_ROOT="$(pwd)"
SIF_FILE="${PROJECT_ROOT}/nckh_h200.sif"

if [ ! -f "$SIF_FILE" ]; then
    if [ -f "$HOME/containers/nckh_h200.sif" ]; then
        SIF_FILE="$HOME/containers/nckh_h200.sif"
    elif [ -f "$HOME/nckh_h200.sif" ]; then
        SIF_FILE="$HOME/nckh_h200.sif"
    else
        echo "[ERROR] Container $SIF_FILE not found! Please build it first."
        exit 1
    fi
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-4}"

echo "Processing chunk: Task $TASK_ID of $NUM_TASKS"

# Run parallel worker on its assigned partition of windows/frames
$CONTAINER_CLI exec --nv \
    --bind "${PROJECT_ROOT}:/workspace/NCKH_SoICT" \
    --bind /dev/shm:/dev/shm \
    "$SIF_FILE" \
    python3 -c "
import os
import sys
from pathlib import Path
import json

task_id = int('$TASK_ID')
num_tasks = int('$NUM_TASKS')

# If protocol_windows.json exists, split windows across tasks
benchmark_root = Path(os.environ.get('BENCHMARK_ROOT', '/home/tondaiquoc/benchmarks/stage2_valid_quick'))
protocol = benchmark_root / 'protocol_windows.json'
if protocol.is_file():
    windows = json.loads(protocol.read_text())
    # Distribute windows evenly across tasks
    chunk = [w for i, w in enumerate(windows) if i % num_tasks == task_id]
    print(f'[Task {task_id}] Assigned {len(chunk)} / {len(windows)} target windows.')
    
    # Save task chunk protocol
    chunk_protocol = benchmark_root / f'protocol_chunk_{task_id}.json'
    chunk_protocol.write_text(json.dumps(chunk, indent=2))
    
    # Execute cascaded evaluation on this chunk
    from evaluate_cascaded_pipeline import run_cascaded_benchmark
    run_cascaded_benchmark(limit_windows=len(chunk))
else:
    print(f'[Task {task_id}] Running benchmark evaluation...')
    from evaluate_cascaded_pipeline import run_cascaded_benchmark
    run_cascaded_benchmark(limit_windows=10)
"

echo "========================================================================"
echo "Task $TASK_ID completed successfully at $(date -u)"
echo "========================================================================"
