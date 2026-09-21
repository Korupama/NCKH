from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json
import numpy as np

from ...stage3_adapter import load_stage3_state
from .contracts import SAM3D_MANIFEST_SCHEMA
from .exporter import eligible_tracks, source_frames_for_tracks, build_boxes_from_stage3


def build_sam3d_manifest(
    *,
    stage3_state: str | Path,
    output_path: str | Path,
    include_roles: Sequence[str] = ("player", "goalkeeper"),
) -> dict:
    s3_path = Path(stage3_state).expanduser().resolve()
    s3 = load_stage3_state(s3_path)
    track_ids = eligible_tracks(s3, include_roles)
    if not track_ids:
        raise ValueError("No player/goalkeeper tracks available for SAM3D manifest")
    frames = source_frames_for_tracks(s3, track_ids)
    if not frames:
        raise ValueError("No observations available for SAM3D manifest")
    boxes = build_boxes_from_stage3(s3, frames, track_ids)
    finite = np.isfinite(boxes).all(axis=-1)
    manifest = {
        "schema_version": SAM3D_MANIFEST_SCHEMA,
        "stage3_state": str(s3_path),
        "selected_frame": int(s3.selected_frame),
        "source_fps": float(s3.replay_context.get("fps")),
        "image_size": [
            int(s3.replay_context.get("image_width")),
            int(s3.replay_context.get("image_height")),
        ],
        "frame_indices": [int(x) for x in frames],
        "track_ids": [str(x) for x in track_ids],
        "roles": {
            t.track_id: str(t.role)
            for t in s3.tracks if t.track_id in set(track_ids)
        },
        "bbox_coverage": {
            "finite": int(finite.sum()),
            "total": int(finite.size),
            "rate": float(finite.mean()) if finite.size else None,
        },
        "records": [
            {
                "frame_index": int(frame),
                "boxes_xyxy": [
                    None if not np.isfinite(boxes[i, j]).all() else [float(v) for v in boxes[i, j]]
                    for j in range(len(track_ids))
                ],
            }
            for i, frame in enumerate(frames)
        ],
    }
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(out)
    return manifest
