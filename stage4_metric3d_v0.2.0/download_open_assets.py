from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RTMW3D_CONFIG_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/"
    "projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"
)
RTMW3D_WEIGHT_URL = (
    "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/"
    "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)
KASPORTSFORMER_REPO = "https://github.com/jw0r1n/KASportsFormer.git"
# Official README Google Drive link for the WorldPose detected-2D checkpoint.
KASPORTSFORMER_WP_DET_GDRIVE_ID = "1OB5GKQrVeRJ8I-2XWnzYC_SpZCN1vBHG"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_rtmw3d(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / "rtmw3d-l_8xb64_cocktail14-384x288.py"
    weight = output_dir / "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
    if not config.exists():
        _download(RTMW3D_CONFIG_URL, config)
    if not weight.exists():
        _download(RTMW3D_WEIGHT_URL, weight)
    return {
        "asset": "rtmw3d-l-384x288",
        "config": str(config.resolve()),
        "checkpoint": str(weight.resolve()),
        "checkpoint_sha256": _sha256(weight),
        "source": "OpenMMLab MMPose v1.3.2 official project/model zoo",
        "global_metric_position_trusted": False,
    }


def prepare_kasportsformer(output_dir: Path, download_checkpoint: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = output_dir / "KASportsFormer"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", KASPORTSFORMER_REPO, str(repo)], check=True)
    result = {
        "asset": "kasportsformer-worldpose-detected-2d",
        "repo": str(repo.resolve()),
        "checkpoint_google_drive_id": KASPORTSFORMER_WP_DET_GDRIVE_ID,
        "role": "optional Stage-4 initializer ablation; not v0.1 production primary",
    }
    if download_checkpoint:
        try:
            import gdown  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install gdown to download the optional KASportsFormer checkpoint") from exc
        checkpoint = repo / "checkpoints" / "evaluate_checkpoint" / "kasportsformer-wp-det.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            downloaded = gdown.download(
                id=KASPORTSFORMER_WP_DET_GDRIVE_ID,
                output=str(checkpoint),
                quiet=False,
            )
            if not downloaded or not checkpoint.exists():
                raise RuntimeError("KASportsFormer checkpoint download did not produce the expected file")
        result["checkpoint"] = str(checkpoint.resolve())
        result["checkpoint_sha256"] = _sha256(checkpoint)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Download open/public Stage-4 model assets")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rtmw3d-l", help="Official OpenMMLab RTMW3D-L config + checkpoint")
    r.add_argument("--output-dir", default="weights/rtmw3d")
    k = sub.add_parser("kasportsformer", help="Optional official KASportsFormer repository")
    k.add_argument("--output-dir", default="third_party")
    k.add_argument("--download-checkpoint", action="store_true")
    args = p.parse_args()

    if args.command == "rtmw3d-l":
        result = download_rtmw3d(Path(args.output_dir))
    else:
        result = prepare_kasportsformer(Path(args.output_dir), args.download_checkpoint)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
