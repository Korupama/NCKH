from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np

from ..camera import CameraTimelineLite
from ..stage3_adapter import load_stage3_state
from ..wholebody import POSE23_NAMES

FIXED_HEIGHT_V03_SCHEMA = "metric-body-proxy-state-1.0"
FIXED_HEIGHT_V03_VERSION = "stage4-fixed-height-baseline-0.3.0"
FIXED_HEIGHT_FRACTIONS = {
    "nose": .93, "left_eye": .95, "right_eye": .95, "left_ear": .94, "right_ear": .94,
    "left_shoulder": .82, "right_shoulder": .82,
    "left_elbow": .66, "right_elbow": .66,
    "left_wrist": .53, "right_wrist": .53,
    "left_hip": .53, "right_hip": .53,
    "left_knee": .285, "right_knee": .285,
    "left_ankle": .055, "right_ankle": .055,
    "left_big_toe": .015, "left_small_toe": .015, "left_heel": .018,
    "right_big_toe": .015, "right_small_toe": .015, "right_heel": .018,
}


def run_fixed_height_v03(
    *,
    stage3_state: str | Path,
    camera_dir: str | Path,
    output_dir: str | Path,
    reference_height_m: float = 1.80,
    selected_frame_only: bool = False,
) -> dict[str, Any]:
    s3 = load_stage3_state(stage3_state)
    cams = CameraTimelineLite.load_dir(camera_dir)
    tracks = []
    for tr in s3.tracks:
        observations = []
        for obs in tr.observations:
            if selected_frame_only and int(obs.frame_index) != int(s3.selected_frame):
                continue
            cam = cams.by_frame(obs.frame_index)
            joints = []
            for j, name in enumerate(POSE23_NAMES):
                state = str(obs.states23[j])
                uv = np.asarray(obs.uv23[j], dtype=np.float64)
                z = float(FIXED_HEIGHT_FRACTIONS[name]) * float(reference_height_m)
                xyz = None
                status = "MISSING_SOURCE"
                if cam is not None and cam.status != "INVALID" and state != "MISSING" and np.isfinite(uv).all():
                    value = np.asarray(cam.intersect_z_plane(uv, z), dtype=np.float64)
                    if value.shape == (3,) and np.isfinite(value).all():
                        xyz = [float(x) for x in value]
                        status = "VALID" if cam.status == "VALID" and state == "VALID" else "DEGRADED"
                    else:
                        status = "GEOMETRY_REJECTED"
                elif cam is None:
                    status = "CAMERA_MISSING"
                joints.append({
                    "index": j, "name": name,
                    "uv_raw_px": None if not np.isfinite(uv).all() else [float(x) for x in uv],
                    "z_prior_m": z,
                    "xyz_proxy_world_m": xyz,
                    "xy_pitch_m": None if xyz is None else xyz[:2],
                    "projection_status": status,
                })
            observations.append({"frame_index": int(obs.frame_index), "metric_body_proxy23": joints})
        tracks.append({"track_id": tr.track_id, "upstream_role": tr.role, "observations": observations})
    state = {
        "schema_version": FIXED_HEIGHT_V03_SCHEMA,
        "stage4_version": FIXED_HEIGHT_V03_VERSION,
        "method": "FIXED_HEIGHT_HORIZONTAL_PLANE_BASELINE",
        "selected_frame": int(s3.selected_frame),
        "reference_height_m": float(reference_height_m),
        "height_fractions": FIXED_HEIGHT_FRACTIONS,
        "research_accuracy_frozen": False,
        "note": "Baseline only. Elevated body points are not safe for production offside use.",
        "tracks": tracks,
    }
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metric_body_proxy_state_v03_baseline.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    state["artifact"] = str(path)
    return state
