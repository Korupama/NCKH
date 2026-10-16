#!/usr/bin/env python3
"""
Automated downloader for public pretrained models (Stages 1, 3, 6).
Downloads weights directly from official GitHub releases and OpenMMLab into datastore.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_SPECS = {
    "stage1_pnlcalib": [
        {
            "filename": "SV_kp",
            "url": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
            "sha256": "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113",
            "symlinks": [PROJECT_ROOT / "stage_1_camera_v12" / "weights" / "SV_kp"],
        },
        {
            "filename": "SV_lines",
            "url": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
            "sha256": "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455",
            "symlinks": [PROJECT_ROOT / "stage_1_camera_v12" / "weights" / "SV_lines"],
        },
    ],
    "stage6_ball": [
        {
            "filename": "yolo-sn-ball-opt.pt",
            "url": "https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/yolo-sn-ball-opt.pt",
            "sha256": "a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36",
            "symlinks": [PROJECT_ROOT / "stage6_ball_localization_v0.4.4" / "weights" / "yolo-sn-ball-opt.pt"],
        },
    ],
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def download_with_progress(url: str, dest: Path) -> None:
    temp_path = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "NCKH-H200-Deployment/1.0"})
    with urllib.request.urlopen(req, timeout=300) as response, temp_path.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total > 0:
                percent = done * 100.0 / total
                print(f"\r  Progress: {percent:5.1f}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)
            else:
                print(f"\r  Progress: {done / 1e6:.1f} MB", end="", flush=True)
    print()
    temp_path.replace(dest)


def download_rtmw_onnx(dest_dir: Path) -> Path:
    dest_file = dest_dir / "rtmw_l_384x288.onnx"
    expected_sha256 = "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054"
    if dest_file.is_file() and sha256_of_file(dest_file) == expected_sha256:
        print(f"[OK] rtmw_l_384x288.onnx already exists and verified.")
        return dest_file

    zip_url = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
    )
    print(f"\n[Stage 3] Downloading RTMW WholeBody ONNX package from OpenMMLab...")
    with tempfile.TemporaryDirectory() as td:
        tmp_zip = Path(td) / "rtmw.zip"
        download_with_progress(zip_url, tmp_zip)
        print("  Extracting ONNX model from zip archive...")
        with zipfile.ZipFile(tmp_zip) as z:
            members = [m for m in z.infolist() if m.filename.lower().endswith(".onnx")]
            if not members:
                raise RuntimeError("No ONNX file found in RTMW zip archive")
            member = max(members, key=lambda m: m.file_size)
            with z.open(member) as src, dest_file.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    actual_sha = sha256_of_file(dest_file)
    if actual_sha != expected_sha256:
        print(f"  WARNING: RTMW SHA256 mismatch (got {actual_sha}, expected {expected_sha256})")
    else:
        print(f"  [VERIFIED] rtmw_l_384x288.onnx verified.")

    # Symlink to stage2 weights
    stage2_weights = PROJECT_ROOT / "stage2_sst_rtmw_v1.3" / "weights"
    stage2_weights.mkdir(parents=True, exist_ok=True)
    symlink_file = stage2_weights / "rtmw_l_384x288.onnx"
    if not symlink_file.exists():
        try:
            symlink_file.symlink_to(dest_file)
            print(f"  [LINK] -> {symlink_file}")
        except Exception:
            shutil.copy2(dest_file, symlink_file)
    return dest_file


def main():
    print("=" * 70)
    print("Automated Downloader for Public Model Weights on HPC-UIT")
    print(f"Project directory: {PROJECT_ROOT}")
    print("=" * 70)

    pretrained_dir = PROJECT_ROOT / "notebook_demo" / "pretrained_models"

    for stage_folder, specs in TARGET_SPECS.items():
        out_dir = pretrained_dir / stage_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        for spec in specs:
            dest_file = out_dir / spec["filename"]
            expected_sha = spec.get("sha256", "").lower()

            print(f"\nChecking: {spec['filename']} in {out_dir.relative_to(PROJECT_ROOT)}...")
            if dest_file.is_file():
                actual_sha = sha256_of_file(dest_file)
                if expected_sha and actual_sha == expected_sha:
                    print(f"  [OK] Exists and SHA-256 verified.")
                else:
                    print(f"  [CORRUPTED] Hash mismatch, re-downloading...")
                    download_with_progress(spec["url"], dest_file)
            else:
                print(f"  Downloading from: {spec['url']}")
                download_with_progress(spec["url"], dest_file)
                if expected_sha:
                    actual_sha = sha256_of_file(dest_file)
                    if actual_sha != expected_sha:
                        raise RuntimeError(f"SHA-256 mismatch for {dest_file.name}: {actual_sha}")
                    print(f"  [VERIFIED] SHA-256 matches.")

            # Create symlinks
            for link_target in spec.get("symlinks", []):
                link_target.parent.mkdir(parents=True, exist_ok=True)
                if not link_target.exists():
                    try:
                        link_target.symlink_to(dest_file)
                        print(f"  [LINK] -> {link_target}")
                    except Exception:
                        shutil.copy2(dest_file, link_target)

    # Download Stage 3 RTMW ONNX
    stage2_dir = pretrained_dir / "stage2_sst_rtmw"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    download_rtmw_onnx(stage2_dir)

    print("\n" + "=" * 70)
    print("Public weights download finished successfully!")
    print("=" * 70)
    print("Remaining proprietary/gated weights (must be provided manually if running them):")
    print("1. Stage 2 SST:   notebook_demo/pretrained_models/stage2_sst_rtmw/model.pth (~237 MB)")
    print("2. Stage 4 SAM3D: notebook_demo/pretrained_models/stage4_sam3d/model.ckpt & mhr_model.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
