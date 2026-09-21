from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from stage4_metric3d.backends.field_converter.sam3d_cache import save_sam3d_cache
from stage4_metric3d.backends.field_converter.exporter import eligible_tracks, source_frames_for_tracks, build_boxes_from_stage3
from stage4_metric3d.stage3_adapter import load_stage3_state


def _ensure_tn(arr: np.ndarray, T: int, trailing: tuple[int, ...], name: str) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim != 2 + len(trailing) or tuple(x.shape[-len(trailing):]) != trailing:
        raise ValueError(f"{name}: expected (T,N,{trailing}) or (N,T,{trailing}), got {x.shape}")
    if x.shape[0] == T:
        return x
    if x.shape[1] == T:
        return np.swapaxes(x, 0, 1)
    raise ValueError(f"{name}: cannot identify T={T} dimension in {x.shape}")


def main() -> int:
    p = argparse.ArgumentParser(description="Wrap exact 25-joint SAM3D arrays into a Stage-4 cache")
    p.add_argument("--stage3-state", required=True)
    p.add_argument("--skel-2d", required=True)
    p.add_argument("--skel-3d-relative", required=True)
    p.add_argument("--output-cache", required=True)
    p.add_argument("--joint-schema", default=None)
    args = p.parse_args()

    s3 = load_stage3_state(args.stage3_state)
    tids = eligible_tracks(s3, ("player", "goalkeeper"))
    frames = source_frames_for_tracks(s3, tids)
    boxes = build_boxes_from_stage3(s3, frames, tids)
    T, N = len(frames), len(tids)
    s2d = _ensure_tn(np.load(args.skel_2d), T, (25, 2), "skel_2d").astype(np.float32)
    s3d = _ensure_tn(np.load(args.skel_3d_relative), T, (25, 3), "skel_3d_relative").astype(np.float32)
    if s2d.shape[:2] != (T, N) or s3d.shape[:2] != (T, N):
        raise ValueError(f"Person count mismatch: expected (T,N)=({T},{N}), got 2D={s2d.shape[:2]}, 3D={s3d.shape[:2]}")
    names = None
    semantic = False
    if args.joint_schema:
        schema = json.loads(Path(args.joint_schema).read_text(encoding="utf-8"))
        names = schema.get("names")
        semantic = bool(schema.get("validated", False))
        if not isinstance(names, list) or len(names) != 25:
            raise ValueError("joint schema must contain exactly 25 names")
    valid = np.isfinite(boxes).all(axis=-1) & np.isfinite(s2d).all(axis=(-1, -2)) & np.isfinite(s3d).all(axis=(-1, -2))
    path = save_sam3d_cache(
        args.output_cache,
        frame_indices=frames,
        track_ids=tids,
        boxes_xyxy=boxes,
        skel_2d_px=s2d,
        skel_3d_relative_m=s3d,
        valid_mask=valid,
        joint_names=names,
        semantic_mapping_validated=semantic,
        metadata={"producer": "import_sam3d_cache.py", "stage3_state": str(Path(args.stage3_state).resolve())},
    )
    print(json.dumps({"cache": str(path), "coverage": float(valid.mean()), "semantic_mapping_validated": semantic}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
