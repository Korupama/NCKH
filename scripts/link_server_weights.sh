#!/usr/bin/env bash
# ==============================================================================
# Script: Auto-discover and symlink pretrained weights on HPC-UIT
# Usage:
#   bash scripts/link_server_weights.sh
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASTORE_BASE="/datastore/${USER:-thangcpd}"

echo "=== Auto-linking Pretrained Weights from ${DATASTORE_BASE} ==="

mkdir -p "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage1_pnlcalib"
mkdir -p "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage2_sst_rtmw"
mkdir -p "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage4_sam3d"
mkdir -p "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage6_ball"
mkdir -p "${PROJECT_ROOT}/stage_1_camera_v12/weights"
mkdir -p "${PROJECT_ROOT}/stage2_sst_rtmw_v1.3/weights"
mkdir -p "${PROJECT_ROOT}/stage6_ball_localization_v0.4.4/weights"

link_file() {
    local target_filename="$1"
    local dest_dir="$2"

    if [ -f "${dest_dir}/${target_filename}" ]; then
        echo "[EXISTS] ${target_filename} -> ${dest_dir}"
        return 0
    fi

    echo "Searching for ${target_filename}..."
    local found_file=""
    found_file=$(find "${DATASTORE_BASE}" -not -path "${PROJECT_ROOT}/*" -name "${target_filename}" -type f 2>/dev/null | head -n 1 || true)

    if [ -n "${found_file}" ] && [ -f "${found_file}" ]; then
        ln -sf "${found_file}" "${dest_dir}/${target_filename}"
        echo "[LINKED] ${found_file} -> ${dest_dir}/${target_filename}"
    else
        echo "[NOT FOUND] ${target_filename}"
    fi
}

echo "--- Stage 1: PnLCalib ---"
link_file "SV_kp" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage1_pnlcalib"
link_file "SV_kp" "${PROJECT_ROOT}/stage_1_camera_v12/weights"
link_file "SV_lines" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage1_pnlcalib"
link_file "SV_lines" "${PROJECT_ROOT}/stage_1_camera_v12/weights"

echo "--- Stage 2: SST Detection & Stage 3 RTMW ---"
link_file "model.pth" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage2_sst_rtmw"
link_file "model.pth" "${PROJECT_ROOT}/stage2_sst_rtmw_v1.3/weights"
link_file "rtmw_l_384x288.onnx" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage2_sst_rtmw"
link_file "rtmw_l_384x288.onnx" "${PROJECT_ROOT}/stage2_sst_rtmw_v1.3/weights"

echo "--- Stage 4: SAM-3D Body ---"
link_file "model.ckpt" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage4_sam3d"
link_file "mhr_model.pt" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage4_sam3d"

echo "--- Stage 6: SoccerNet Ball YOLO ---"
link_file "yolo-sn-ball-opt.pt" "${PROJECT_ROOT}/notebook_demo/pretrained_models/stage6_ball"
link_file "yolo-sn-ball-opt.pt" "${PROJECT_ROOT}/stage6_ball_localization_v0.4.4/weights"

echo "--- Video Sample Check ---"
if [ ! -f "${PROJECT_ROOT}/video_bookmark/mixed.mp4" ]; then
    echo "Searching for sample mp4 video..."
    sample_mp4=$(find "${DATASTORE_BASE}" -not -path "${PROJECT_ROOT}/*" -name "*.mp4" -type f 2>/dev/null | head -n 1 || true)
    if [ -n "${sample_mp4}" ]; then
        ln -sf "${sample_mp4}" "${PROJECT_ROOT}/video_bookmark/mixed.mp4"
        echo "[LINKED] ${sample_mp4} -> ${PROJECT_ROOT}/video_bookmark/mixed.mp4"
    else
        echo "[NOT FOUND] No .mp4 file found on ${DATASTORE_BASE}. You can copy your video to video_bookmark/mixed.mp4"
    fi
else
    echo "[EXISTS] video_bookmark/mixed.mp4 already present"
fi

echo "=== Finished Weights Linking ==="
