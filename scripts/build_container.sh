#!/usr/bin/env bash
# ==============================================================================
# Script: Build Docker Image and/or Apptainer (Singularity) .sif for H200 Cluster
# Usage:
#   bash scripts/build_container.sh [docker|singularity|all]
# ==============================================================================

set -euo pipefail

MODE="${1:-all}"
IMAGE_TAG="nckh-pipeline:h200"
SIF_NAME="nckh_h200.sif"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== [1/3] Checking workspace: $ROOT_DIR ==="

build_docker() {
    echo "=== Building Docker image: $IMAGE_TAG ==="
    docker build -t "$IMAGE_TAG" -f Dockerfile .
    echo "[SUCCESS] Docker image $IMAGE_TAG built successfully."
}

build_singularity() {
    echo "=== Building Singularity/Apptainer SIF image: $SIF_NAME ==="
    if command -v apptainer &>/dev/null; then
        BUILDER="apptainer"
    elif command -v singularity &>/dev/null; then
        BUILDER="singularity"
    else
        echo "[ERROR] Neither apptainer nor singularity found in PATH."
        echo "If you already built Docker image, you can save and convert it on the cluster:"
        echo "  docker save $IMAGE_TAG -o nckh_image.tar"
        echo "  apptainer build $SIF_NAME docker-archive://nckh_image.tar"
        return 1
    fi

    # Build directly from docker-daemon if available, or build definition
    if docker image inspect "$IMAGE_TAG" &>/dev/null; then
        echo "Building from local Docker daemon image..."
        $BUILDER build --force "$SIF_NAME" "docker-daemon://$IMAGE_TAG"
    else
        echo "Building directly from Dockerfile via docker-archive or local context..."
        build_docker
        $BUILDER build --force "$SIF_NAME" "docker-daemon://$IMAGE_TAG"
    fi
    echo "[SUCCESS] SIF file created at: $ROOT_DIR/$SIF_NAME"
}

build_docker_light() {
    echo "=== Building Lightweight Docker image: $IMAGE_TAG (via Dockerfile.light) ==="
    docker build -t "$IMAGE_TAG" -f Dockerfile.light .
    echo "[SUCCESS] Lightweight Docker image $IMAGE_TAG built successfully."
}

case "$MODE" in
    docker)
        build_docker
        ;;
    light)
        build_docker_light
        ;;
    singularity)
        build_singularity
        ;;
    all)
        build_docker
        if command -v apptainer &>/dev/null || command -v singularity &>/dev/null; then
            build_singularity
        else
            echo "[INFO] Skipped SIF build (apptainer/singularity not installed locally)."
            echo "You can transfer Docker image to cluster or build SIF on cluster login node."
        fi
        ;;
    *)
        echo "Usage: $0 [docker|light|singularity|all]"
        exit 1
        ;;
esac

echo "=== Done! ==="


