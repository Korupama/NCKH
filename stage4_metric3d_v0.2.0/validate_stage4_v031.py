from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from stage4_metric3d.camera import CameraStateLite
from stage4_metric3d.height_profiles import load_height_profile
from stage4_metric3d.proxy_processor import run_stage4
from stage4_metric3d.proxy_schemas import Stage4ProjectionConfig
from stage4_metric3d.wholebody import POSE23_NAMES, WHOLEBODY_NAMES


def _camera(frame_index: int, width: int = 1280, height: int = 720) -> dict:
    center = np.asarray([0.0, -42.0, 15.0], dtype=np.float64)
    target = np.asarray([4.0, 4.0, 0.9], dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.stack([right, down, forward], axis=0)
    intrinsic = np.asarray([[1600.0, 0.0, width / 2], [0.0, 1600.0, height / 2], [0.0, 0.0, 1.0]])
    return {
        "schema_version": "1.2",
        "frame_index": frame_index,
        "status": "VALID",
        "image": {"width": width, "height": height},
        "intrinsics": {"K": intrinsic.tolist()},
        "extrinsics": {"R_world_to_camera": rotation.tolist(), "camera_center_world_m": center.tolist()},
        "distortion": {"radial": [0.0] * 6, "tangential": [0.0] * 2, "thin_prism": [0.0] * 4},
        "pitch": {
            "length_m": 105.0,
            "width_m": 68.0,
            "origin": "center",
            "x_axis": "goal_to_goal",
            "y_axis": "touchline_to_touchline",
            "z_axis": "up",
        },
    }


def _write_case(root: Path) -> tuple[Path, Path, np.ndarray]:
    profile = load_height_profile("canonical-body-height-v1")
    camera_dir = root / "cameras"
    camera_dir.mkdir(parents=True, exist_ok=True)
    camera_data = _camera(86)
    (camera_dir / "camera_state_000086.json").write_text(json.dumps(camera_data, indent=2), encoding="utf-8")
    camera = CameraStateLite.from_dict(camera_data)

    xyz = np.zeros((23, 3), dtype=np.float64)
    for index, name in enumerate(POSE23_NAMES):
        xyz[index] = [4.0 + 0.025 * index, 3.6 + 0.018 * ((index % 5) - 2), profile.z_m(name, 1.8)]
    uv = camera.project_world(xyz)
    records = []
    for index, name in enumerate(WHOLEBODY_NAMES):
        if index < 23:
            x, y = uv[index]
            records.append({
                "index": index,
                "name": name,
                "x": float(x),
                "y": float(y),
                "raw_model_score": 1.0,
                "state": "VALID",
            })
        else:
            records.append({
                "index": index,
                "name": name,
                "x": None,
                "y": None,
                "raw_model_score": None,
                "state": "MISSING",
            })
    stage3 = {
        "schema_version": "tracked-pose-2d-state-1.0",
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "keypoint_schema": {"name": "COCO_WHOLEBODY_133", "count": 133, "names": list(WHOLEBODY_NAMES)},
        "replay_context": {"selected_frame": 86, "image_width": 1280, "image_height": 720, "fps": 30.0},
        "tracks": [{
            "track_id": "track_001",
            "upstream_role": "player",
            "upstream_identity_confidence": 1.0,
            "observations": [{
                "frame_index": 86,
                "source_bbox_xyxy": [500.0, 200.0, 800.0, 700.0],
                "pose_status": "VALID",
                "keypoints_133": records,
            }],
        }],
    }
    stage3_path = root / "tracked_pose_2d_state.json"
    stage3_path.write_text(json.dumps(stage3, indent=2), encoding="utf-8")
    return stage3_path, camera_dir, xyz


def validate(output_dir: str | Path | None = None) -> dict:
    if output_dir is None:
        context = tempfile.TemporaryDirectory(prefix="stage4_v031_")
        root = Path(context.name)
    else:
        context = None
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    try:
        stage3_path, camera_dir, expected = _write_case(root / "inputs")
        state = run_stage4(
            stage3_state=stage3_path,
            camera_dir=camera_dir,
            output_dir=root / "output",
            config=Stage4ProjectionConfig(window_radius_frames=0, export_visualization=False),
        )
        points = state["tracks"][0]["observations"][0]["raw_height_plane_proxies_23"]
        actual = np.asarray([point["xyz_proxy_world_m"] for point in points], dtype=np.float64)
        max_error = float(np.max(np.linalg.norm(actual - expected, axis=1)))
        handoff = json.loads(Path(state["artifacts"]["stage4_downstream_handoff"]).read_text(encoding="utf-8"))
        handoff_withholds_raw = all(
            track.get("raw_height_plane_proxies_exported") is False
            and "selected_frame_observation" not in track
            for track in handoff["tracks"]
        )
        stage8_blocked = handoff["readiness"]["stage8_legal_body_handoff_ready"] is False
        result = {
            "schema_version": "stage4-v031-validation-1.0",
            "status": "PASS" if (
                max_error <= 1e-8
                and state["acceptance_gate"]["status"] == "PASS"
                and handoff_withholds_raw
                and stage8_blocked
            ) else "FAIL",
            "max_xyz_roundtrip_error_m": max_error,
            "projection_coverage_at_t0": state["metrics"]["ProjectionCoverageAtT0"],
            "reprojection_consistency_px": state["metrics"]["ReprojectionConsistencyPx"],
            "acceptance_gate": state["acceptance_gate"],
            "handoff_withholds_raw_height_plane_proxies": handoff_withholds_raw,
            "stage8_legal_body_handoff_blocked": stage8_blocked,
            "output": state["artifacts"]["metric_body_proxy_state"],
        }
        if output_dir is not None:
            (root / "validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if context is not None:
            context.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic synthetic validation for Stage 4 v0.3.1")
    parser.add_argument("--output-dir", default=None, help="Keep generated inputs/outputs instead of using a temporary directory")
    args = parser.parse_args()
    result = validate(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
