from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import numpy as np

from stage_1_camera.contracts import CameraState, CameraStatus, Distortion, PitchSpec
from stage_1_camera.geometry import camera_view_metadata


def _pitch_from_payload(payload: Dict[str, Any]) -> PitchSpec:
    raw = payload.get("pitch") or {}
    allowed = {f.name for f in fields(PitchSpec)}
    return PitchSpec(**{k: raw[k] for k in raw if k in allowed})


def _distortion_from_payload(payload: Dict[str, Any]) -> Distortion:
    raw = payload.get("distortion") or {}
    return Distortion(
        model=raw.get("model", "opencv"),
        radial=list(raw.get("radial") or []),
        tangential=list(raw.get("tangential") or []),
        thin_prism=list(raw.get("thin_prism") or []),
    )


def enrich_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    image = payload.get("image") or {}
    intr = payload.get("intrinsics") or {}
    extr = payload.get("extrinsics") or {}
    K = intr.get("K")
    R = extr.get("R_world_to_camera")
    C = extr.get("camera_center_world_m")
    if K is None or R is None or C is None:
        raise ValueError("CameraState JSON must contain intrinsics.K, extrinsics.R_world_to_camera and extrinsics.camera_center_world_m")
    status_raw = str(payload.get("status", "INVALID")).upper()
    status = CameraStatus(status_raw) if status_raw in {x.value for x in CameraStatus} else CameraStatus.INVALID
    cam = CameraState(
        frame_index=int(payload.get("frame_index", -1)),
        image_width=int(image.get("width")),
        image_height=int(image.get("height")),
        K=np.asarray(K, dtype=float),
        R_world_to_camera=np.asarray(R, dtype=float),
        camera_center_world_m=np.asarray(C, dtype=float),
        distortion=_distortion_from_payload(payload),
        pitch=_pitch_from_payload(payload),
        timestamp_sec=payload.get("timestamp_sec"),
        status=status,
        schema_version=str(payload.get("schema_version", "1.2")),
    )
    out = dict(payload)
    out["view"] = camera_view_metadata(cam)
    return out


def process_file(src: Path, dst: Path) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    enriched = enrich_payload(payload)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Add Stage-1 view metadata to existing CameraState JSON artifacts")
    p.add_argument("--input", required=True, help="CameraState JSON file or directory")
    p.add_argument("--output", help="Output JSON file or directory; required unless --in-place")
    p.add_argument("--in-place", action="store_true")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)
    if args.in_place and args.output:
        p.error("use either --in-place or --output, not both")
    if not args.in_place and not args.output:
        p.error("--output is required unless --in-place is used")

    if src.is_file():
        dst = src if args.in_place else Path(args.output)
        process_file(src, dst)
        print(dst)
        return 0

    out_root = src if args.in_place else Path(args.output)
    count = 0
    for path in sorted(src.glob("camera_state_*.json")):
        dst = path if args.in_place else out_root / path.name
        process_file(path, dst)
        count += 1
    print(json.dumps({"processed": count, "output": str(out_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
