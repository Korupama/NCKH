#!/usr/bin/env bash
# ==============================================================================
# Script: Automated Setup for HPC-UIT (hpc.uit.edu.vn)
# Run this ONCE on the login node:
#   bash scripts/setup_uit_environment.sh
# ==============================================================================

set -euo pipefail

USER_NAME="${USER:-hpc}"
DATASTORE_DIR="/datastore/${USER_NAME}"

echo "=== [1/4] Checking Disk Quota in /home/${USER_NAME} ==="
HOME_USAGE=$(du -sh "$HOME" 2>/dev/null | cut -f1 || echo "unknown")
echo "Current /home usage: $HOME_USAGE"
echo "[IMPORTANT] /home has a strict limit of 10GB. If exceeded, job submission is blocked!"

echo "=== [2/4] Initializing /datastore/${USER_NAME} Storage ==="
if [ ! -d "$DATASTORE_DIR" ]; then
    echo "Creating datastore directory: $DATASTORE_DIR"
    mkdir -p "$DATASTORE_DIR"
fi

# Create dedicated cache folders on datastore
mkdir -p "${DATASTORE_DIR}/.apptainer_cache"
mkdir -p "${DATASTORE_DIR}/.apptainer_tmp"
mkdir -p "${DATASTORE_DIR}/slurm_logs"
mkdir -p "${DATASTORE_DIR}/weights"
mkdir -p "${DATASTORE_DIR}/datasets"

echo "[SUCCESS] Datastore directories ready at: $DATASTORE_DIR"

echo "=== [3/4] Configuring Environment Variables (~/.bashrc) ==="
# Ensure Apptainer never writes cache into /home
BASHRC="$HOME/.bashrc"
ENV_SNIPPET="
# >>> HPC-UIT Apptainer / Singularity Datastore Redirect >>>
export APPTAINER_CACHEDIR=\"${DATASTORE_DIR}/.apptainer_cache\"
export APPTAINER_TMPDIR=\"${DATASTORE_DIR}/.apptainer_tmp\"
export SINGULARITY_CACHEDIR=\"${DATASTORE_DIR}/.apptainer_cache\"
export SINGULARITY_TMPDIR=\"${DATASTORE_DIR}/.apptainer_tmp\"
# <<< HPC-UIT Apptainer / Singularity Datastore Redirect <<<
"

if ! grep -q "HPC-UIT Apptainer" "$BASHRC" 2>/dev/null; then
    echo "$ENV_SNIPPET" >> "$BASHRC"
    echo "[SUCCESS] Appended datastore cache paths to ~/.bashrc"
else
    echo "[INFO] ~/.bashrc already configured with datastore cache paths."
fi

echo "=== [4/4] Verifying Slurm Partitions and Accounts ==="
if command -v sinfo &>/dev/null; then
    sinfo -s || true
else
    echo "[INFO] sinfo not available on local machine (will run on cluster)."
fi

echo "Setup complete. Working directory: ${DATASTORE_DIR}/NCKH_SoICT"

