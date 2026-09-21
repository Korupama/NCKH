from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from stage4_metric3d.wholebody import WHOLEBODY_NAMES


def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = np.asarray(target, dtype=float) - np.asarray(C, dtype=float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    d /= np.linalg.norm(d)
    return np.stack([r, d, f])


def build_synthetic_stage3_and_cameras(root: Path, frames=(10, 11, 12), track_ids=("track_001", "track_002")):
    root.mkdir(parents=True, exist_ok=True)
    camera_dir = root / "cameras"
    camera_dir.mkdir(exist_ok=True)
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
    C = np.array([0.0, -75.0, 16.0])
    R = look_at(C, np.array([10.0, 0.0, 1.0]))
    for frame in frames:
        camera = {
            "schema_version": "1.2",
            "frame_index": int(frame),
            "status": "VALID",
            "image": {"width": 1280, "height": 720},
            "intrinsics": {"K": K.tolist()},
            "extrinsics": {"R_world_to_camera": R.tolist(), "camera_center_world_m": C.tolist()},
            "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0], "thin_prism": [0, 0, 0, 0]},
            "pitch": {"length_m": 105.0, "width_m": 68.0},
        }
        (camera_dir / f"camera_state_{frame:08d}.json").write_text(json.dumps(camera), encoding="utf-8")

    tracks = []
    for n, tid in enumerate(track_ids):
        observations = []
        for ti, frame in enumerate(frames):
            x1 = 300 + n * 300 + ti * 2
            y1, x2, y2 = 180, x1 + 80, 650
            kps = []
            for j, name in enumerate(WHOLEBODY_NAMES):
                if j < 23:
                    x = x1 + 40 + (j % 3 - 1) * 3
                    y = y1 + 30 + j * 12
                    state, score = "VALID", 0.95
                else:
                    x = y = None
                    state, score = "MISSING", None
                kps.append({"index": j, "name": name, "x": x, "y": y, "state": state, "raw_model_score": score})
            observations.append({
                "frame_index": int(frame),
                "source_bbox_xyxy": [x1, y1, x2, y2],
                "pose_status": "VALID",
                "keypoints_133": kps,
            })
        tracks.append({
            "track_id": tid,
            "upstream_role": "player" if n == 0 else "goalkeeper",
            "upstream_identity_confidence": 0.99,
            "observations": observations,
        })
    state = {
        "schema_version": "tracked-pose-2d-state-1.0",
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "keypoint_schema": {"count": 133, "names": list(WHOLEBODY_NAMES)},
        "replay_context": {
            "selected_frame": int(frames[len(frames)//2]),
            "fps": 30.0,
            "frame_count": 100,
            "image_width": 1280,
            "image_height": 720,
        },
        "tracks": tracks,
    }
    stage3_path = root / "tracked_pose_2d_state.json"
    stage3_path.write_text(json.dumps(state), encoding="utf-8")
    return stage3_path, camera_dir, list(frames), list(track_ids)
