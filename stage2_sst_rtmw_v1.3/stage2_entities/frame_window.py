from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import cv2

from .contracts import ReplayContext


def extract_analysis_window(
    context: ReplayContext,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Decode exactly Stage-1's global analysis window to lossless PNG frames."""
    context.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frame_map.json"

    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("global_frame_indices") == context.frame_indices and all(
            Path(x["path"]).is_file() for x in existing.get("frames", [])
        ):
            return existing

    cap = cv2.VideoCapture(context.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(context.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, context.window_start)
    rows: List[Dict[str, Any]] = []
    for local, global_index in enumerate(context.frame_indices):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Video decode failed at global frame {global_index}")
        h, w = frame.shape[:2]
        if (w, h) != (context.image_width, context.image_height):
            cap.release()
            raise RuntimeError(
                f"Frame {global_index} has {w}x{h}, expected {context.image_width}x{context.image_height}"
            )
        path = output_dir / f"frame_{global_index:09d}.png"
        if overwrite or not path.is_file():
            if not cv2.imwrite(str(path), frame):
                cap.release()
                raise OSError(path)
        rows.append({
            "local_frame_index": local,
            "global_frame_index": global_index,
            "is_selected_frame": global_index == context.selected_frame,
            "path": str(path.resolve()),
        })
    cap.release()
    payload = {
        "schema_version": "stage2-frame-map-1.0",
        "coordinate_space": context.coordinate_space,
        "selected_frame": context.selected_frame,
        "global_frame_indices": context.frame_indices,
        "frames": rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
