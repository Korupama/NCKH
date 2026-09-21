from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import cv2
import numpy as np
from .wholebody133 import BODY_SKELETON_EDGES

STATE_COLOR = {
    "VALID": (60, 220, 60),
    "LOW_MODEL_EVIDENCE": (0, 210, 255),
    "GEOMETRIC_OUTLIER": (0, 0, 255),
    "TEMPORAL_OUTLIER": (255, 0, 255),
    "LEFT_RIGHT_SUSPECT": (255, 180, 0),
    "MISSING": (100, 100, 100),
}


def draw_pose(image: np.ndarray, pose_obs: Mapping[str, Any], track_id: str, role: str) -> np.ndarray:
    out = image.copy()
    bbox = pose_obs.get("source_bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(out, f"{track_id} {role} {pose_obs.get('pose_status')}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    kps = pose_obs.get("keypoints_133", [])
    def pt(i):
        if i >= len(kps): return None
        x, y = kps[i].get("x"), kps[i].get("y")
        if x is None or y is None: return None
        return int(round(float(x))), int(round(float(y)))
    for a, b in BODY_SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(out, pa, pb, (180, 180, 180), 1, cv2.LINE_AA)
    for i in range(min(23, len(kps))):
        p = pt(i)
        if not p: continue
        color = STATE_COLOR.get(str(kps[i].get("state")), (255, 255, 255))
        cv2.circle(out, p, 2, color, -1, cv2.LINE_AA)
    return out


def save_selected_frame_overlay(video_path: str | Path, frame_index: int, selected_poses, output_path: str | Path) -> Path:
    from .video import read_frame
    frame = read_frame(video_path, frame_index)
    for item in selected_poses:
        frame = draw_pose(frame, item["observation"], item["track_id"], item.get("upstream_role", ""))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    return output
