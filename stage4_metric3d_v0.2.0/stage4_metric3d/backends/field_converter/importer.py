from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import math
import numpy as np

from .contracts import HANDOFF_SCHEMA, OUTPUT_SCHEMA, STAGE4_V04_VERSION


def locate_predictions(output_root: str | Path, sequence_name: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    matches = sorted(root.glob(f"*/{sequence_name}/predictions.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one official Field Converter predictions.npz for {sequence_name} under {root}; found {matches}"
        )
    return matches[0]


def _safe_vec(arr: np.ndarray) -> list[float] | None:
    a = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(a).all():
        return None
    return [float(x) for x in a]


def _source_index_for_frame(frame_numbers_float: np.ndarray, source_frame: int) -> int | None:
    arr = np.asarray(frame_numbers_float, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    idxs = np.flatnonzero(finite)
    local = int(np.argmin(np.abs(arr[finite] - float(source_frame))))
    return int(idxs[local])


def skeleton_geometry_metrics(joints_world: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    xspans, yspans, zspans, diameters = [], [], [], []
    N, T, J, _ = joints_world.shape
    for n in range(N):
        for t in range(T):
            if not valid[n, t]:
                continue
            pts = np.asarray(joints_world[n, t], dtype=np.float64)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 2:
                continue
            xspans.append(float(np.ptp(pts[:, 0])))
            yspans.append(float(np.ptp(pts[:, 1])))
            zspans.append(float(np.ptp(pts[:, 2])))
            dmax = 0.0
            for i in range(len(pts)):
                d = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1) if i + 1 < len(pts) else np.empty(0)
                if d.size:
                    dmax = max(dmax, float(np.max(d)))
            diameters.append(dmax)

    def stats(values):
        if not values:
            return {"count": 0, "median": None, "p95": None, "max": None}
        a = np.asarray(values, dtype=np.float64)
        return {
            "count": int(a.size),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95.0)),
            "max": float(np.max(a)),
        }
    return {
        "x_span_m": stats(xspans),
        "y_span_m": stats(yspans),
        "z_span_m": stats(zspans),
        "skeleton_diameter_m": stats(diameters),
    }


def import_field_converter_predictions(
    *,
    predictions_path: str | Path,
    summary_path: str | Path | None,
    track_ids: Sequence[str],
    source_frames: Sequence[int],
    selected_frame: int,
    sam3d_joint_names: Sequence[str],
    semantic_mapping_validated: bool,
    camera_compatibility: dict,
    camera_domain: dict,
    catastrophic_xy_span_m: float,
    catastrophic_z_span_m: float,
) -> dict[str, Any]:
    p = Path(predictions_path).expanduser().resolve()
    with np.load(p, allow_pickle=True) as data:
        valid = np.asarray(data["valid_mask"], dtype=bool)
        frame_numbers_float = np.asarray(data["source_frame_numbers_float"], dtype=np.float64)
        roots_cam = np.asarray(data["root_pred_m"], dtype=np.float32)
        roots_source = np.asarray(data["root_source_world_pred_m"], dtype=np.float32)
        roots_init_cam = np.asarray(data["root_init_cam_m"], dtype=np.float32)
        joints_rel = np.asarray(data["joints_pred_cam_m"], dtype=np.float32) - roots_cam[..., None, :]
        joints_cam = np.asarray(data["joints_pred_cam_m"], dtype=np.float32)
        joints_source = np.asarray(data["joints_pred_source_world_m"], dtype=np.float32)
        joints_2d = np.asarray(data["joints_pred_2d"], dtype=np.float32)
        sam2d = None
        if "meta_json" in data.files:
            meta_json = str(np.asarray(data["meta_json"]).item())
        else:
            meta_json = "{}"
        delta_cam = np.asarray(data["root_delta_pred_cam_m"], dtype=np.float32) if "root_delta_pred_cam_m" in data.files else None
        world_alignment_rotation = np.asarray(data["world_alignment_rotation"], dtype=np.float32)
        output_fps = float(np.asarray(data["output_fps"]).item())
    if valid.shape[0] != len(track_ids):
        raise ValueError(f"Field Converter person count {valid.shape[0]} != exported track count {len(track_ids)}")
    if joints_source.shape[-2:] != (25, 3):
        raise ValueError(f"Unexpected Field Converter joint output shape: {joints_source.shape}")

    summary = {}
    if summary_path is not None and Path(summary_path).is_file():
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    else:
        try:
            summary = json.loads(json.loads(meta_json).get("summary", "{}"))
        except Exception:
            summary = {}

    selected_t = _source_index_for_frame(frame_numbers_float, int(selected_frame))
    tracks_out = []
    for n, tid in enumerate(track_ids):
        observations = []
        for frame in source_frames:
            t_idx = _source_index_for_frame(frame_numbers_float, int(frame))
            if t_idx is None:
                continue
            ok = bool(valid[n, t_idx])
            joint_records = []
            for j in range(25):
                joint_records.append({
                    "index": j,
                    "name": str(sam3d_joint_names[j]),
                    "xyz_relative_cam_m": _safe_vec(joints_rel[n, t_idx, j]) if ok else None,
                    "xyz_cam_m": _safe_vec(joints_cam[n, t_idx, j]) if ok else None,
                    "xyz_world_m": _safe_vec(joints_source[n, t_idx, j]) if ok else None,
                    "reprojected_uv_px": _safe_vec(joints_2d[n, t_idx, j]) if ok else None,
                    "valid": bool(ok and np.isfinite(joints_source[n, t_idx, j]).all()),
                })
            observations.append({
                "frame_index": int(frame),
                "field_converter_t_index": int(t_idx),
                "valid": ok,
                "root": {
                    "geometry_init_cam_m": _safe_vec(roots_init_cam[n, t_idx]),
                    "residual_cam_m": None if delta_cam is None else _safe_vec(delta_cam[n, t_idx]),
                    "pred_cam_m": _safe_vec(roots_cam[n, t_idx]) if ok else None,
                    "pred_world_m": _safe_vec(roots_source[n, t_idx]) if ok else None,
                },
                "joints": joint_records,
            })
        tracks_out.append({
            "track_id": str(tid),
            "selected_frame_status": "VALID" if (selected_t is not None and bool(valid[n, selected_t])) else "MISSING",
            "observations": observations,
        })

    geometry = skeleton_geometry_metrics(joints_source, valid)
    catastrophic = False
    for key in ("x_span_m", "y_span_m"):
        mx = geometry[key]["max"]
        catastrophic |= mx is not None and float(mx) > float(catastrophic_xy_span_m)
    zmax = geometry["z_span_m"]["max"]
    catastrophic |= zmax is not None and float(zmax) > float(catastrophic_z_span_m)

    predicted_count = int(valid.sum())
    impl_gate = "PASS" if predicted_count > 0 else "FAIL"
    geom_gate = "FAIL_CATASTROPHIC" if catastrophic else ("PASS_SANITY" if predicted_count > 0 else "NOT_EVALUATED")
    state = {
        "schema_version": OUTPUT_SCHEMA,
        "stage4_version": STAGE4_V04_VERSION,
        "method": "FIELD_CONVERTER_TCN_WORLD_GROUNDED_ROOT_PLUS_SAM3D_RELATIVE",
        "world_frame": {
            "origin": "stage1_pitch_center",
            "x_axis": "goal_to_goal",
            "y_axis": "touchline_to_touchline",
            "z_axis": "up",
            "units": "metres",
            "note": "Uses Field Converter source-world outputs after reversing any internal world alignment.",
        },
        "joint_schema": {
            "count": 25,
            "names": [str(x) for x in sam3d_joint_names],
            "semantic_mapping_validated": bool(semantic_mapping_validated),
            "stage8_semantic_ready": bool(semantic_mapping_validated),
        },
        "selected_frame": int(selected_frame),
        "field_converter_output_fps": output_fps,
        "field_converter_world_alignment_rotation": world_alignment_rotation.tolist(),
        "camera_compatibility": camera_compatibility,
        "camera_domain": camera_domain,
        "official_field_converter_summary": summary,
        "metrics": {
            "predicted_player_frames": predicted_count,
            "prediction_coverage": float(np.mean(valid)) if valid.size else None,
            "skeleton_geometry": geometry,
            "proxy_reprojection_to_observed_sam2d": summary.get("proxy_reprojection_to_observed_sam2d"),
            "delta_correction_camera": summary.get("delta_correction_camera"),
        },
        "quality_gates": {
            "implementation_gate": impl_gate,
            "geometric_quality_gate": geom_gate,
            "metric_accuracy_gate": "NOT_EVALUATED",
            "downstream_offside_gate": "NOT_EVALUATED",
            "research_accuracy_frozen": False,
        },
        "tracks": tracks_out,
        "provenance": {
            "predictions_npz": str(p),
            "official_field_converter_summary": None if summary_path is None else str(Path(summary_path).resolve()),
        },
    }
    return state


def build_downstream_handoff(state: dict, state_path: str | Path) -> dict:
    selected = int(state["selected_frame"])
    tracks = []
    for tr in state.get("tracks", []):
        obs = next((o for o in tr.get("observations", []) if int(o.get("frame_index", -1)) == selected), None)
        tracks.append({
            "track_id": tr.get("track_id"),
            "selected_frame_status": tr.get("selected_frame_status"),
            "root_world_m": None if obs is None else (obs.get("root") or {}).get("pred_world_m"),
            "joints_world_m": None if obs is None else [j.get("xyz_world_m") for j in obs.get("joints", [])],
            "joint_valid": None if obs is None else [bool(j.get("valid")) for j in obs.get("joints", [])],
        })
    return {
        "schema_version": HANDOFF_SCHEMA,
        "world_grounded_pose_state": str(Path(state_path).resolve()),
        "selected_frame": selected,
        "joint_schema": state.get("joint_schema"),
        "quality_gates": state.get("quality_gates"),
        "tracks": tracks,
        "explicit_non_ownership": [
            "team_id", "attacking_team", "attack_direction", "toucher",
            "ball_state", "legal_body_mask", "second_last_opponent", "offside_position_decision",
        ],
    }
