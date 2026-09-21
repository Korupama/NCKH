#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ASSETS = {
    "rtmw-l-384x288": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "type": "onnx_zip",
    },
    "3dsp": {
        "url": "https://raw.githubusercontent.com/calvinyeungck/3D-Shot-Posture-Dataset/master/3dsp.zip",
        "type": "dataset_zip",
    },
}


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent":"offside-stage3-pose2d/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, 1024 * 1024)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(description="Download public Stage-3 assets. Third-party files are not bundled with this package.")
    p.add_argument("asset", choices=sorted(ASSETS))
    p.add_argument("--output-dir", type=Path, default=Path("weights"))
    p.add_argument("--no-extract", action="store_true")
    args=p.parse_args()
    spec=ASSETS[args.asset]
    out=args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)/f"{args.asset}.zip"
        print(f"Downloading {spec['url']}")
        _download(spec["url"], tmp)
        print("sha256", _sha(tmp))
        if args.no_extract:
            target=out/f"{args.asset}.zip"; shutil.copy2(tmp,target); print(target); return
        if spec["type"] == "onnx_zip":
            with zipfile.ZipFile(tmp) as z:
                members=[m for m in z.infolist() if m.filename.lower().endswith(".onnx")]
                if not members: raise RuntimeError("No ONNX file in archive")
                member=max(members,key=lambda m:m.file_size)
                target=out/"rtmw_l_384x288.onnx"
                with z.open(member) as src, target.open("wb") as dst: shutil.copyfileobj(src,dst)
            print(target); print("model_sha256", _sha(target))
        else:
            target=out/"3dsp"; target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp) as z: z.extractall(out)
            print(out)

if __name__ == "__main__": main()
