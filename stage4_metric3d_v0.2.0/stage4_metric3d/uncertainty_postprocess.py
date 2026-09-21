from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping
import copy
import hashlib
import json

import numpy as np

from .camera import CameraTimelineLite
from .initializers.cache import InitializerCache
from .output_semantics import build_stage5_handoff, selected_observation
from .processor import _frames_for_track, _json_safe, preflight_stage4
from .schemas import Stage4Config
from .stage3_adapter import load_stage3_state
from .uncertainty import pixel_sensitivity_monte_carlo


@dataclass(frozen=True)
class FrozenTrackSolution:
    estimated_height_m: float
    depth_beta_m: float | None
    initializer_used: bool
    frame_indices: np.ndarray
    xyz_world_m: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(value: str | Path | None, label: str, *, directory: bool = False) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _config_from_state(state: Mapping[str, Any]) -> Stage4Config:
    allowed = {item.name for item in fields(Stage4Config)}
    raw = dict(state.get("configuration") or {})
    return Stage4Config(**{key: value for key, value in raw.items() if key in allowed})


def _not_applicable(track_id: str, selected_frame: int, samples: int, reason: str, mode: str) -> dict[str, Any]:
    return {
        "schema_version": "stage4-longitudinal-sensitivity-1.0",
        "status": "NOT_APPLICABLE",
        "scope": (
            "stage3_pixel_only_camera_fixed_selected_frame"
            if mode == "selected_frame"
            else "stage3_pixel_only_camera_fixed_full_window"
        ),
        "calibrated": False,
        "interpretation": "engineering sensitivity, not calibrated posterior probability",
        "track_id": track_id,
        "selected_frame": selected_frame,
        "samples_requested": samples,
        "samples_usable": 0,
        "samples_successful": 0,
        "usable_fraction": 0.0,
        "reason": reason,
        "std_xyz_m": None,
        "q025_xyz_m": None,
        "q975_xyz_m": None,
        "longitudinal": None,
    }


def _attach_joint_uncertainty(observation: dict[str, Any] | None, uncertainty: Mapping[str, Any]) -> None:
    if observation is None:
        return
    usable = uncertainty.get("status") in {"OK", "DEGRADED"}
    std = np.asarray(uncertainty.get("std_xyz_m"), dtype=np.float64) if usable else None
    q025 = np.asarray(uncertainty.get("q025_xyz_m"), dtype=np.float64) if usable else None
    q975 = np.asarray(uncertainty.get("q975_xyz_m"), dtype=np.float64) if usable else None
    for joint in observation.get("metric_pose23") or []:
        index = int(joint.get("index", -1))
        if std is not None and std.shape == (23, 3) and 0 <= index < 23 and np.isfinite(std[index]).all():
            joint["std_xyz_m"] = std[index].tolist()
            joint["q025_xyz_m"] = q025[index].tolist()
            joint["q975_xyz_m"] = q975[index].tolist()
            joint["uncertainty_scope"] = uncertainty.get("scope")
            joint["uncertainty_calibrated"] = False
        else:
            joint["std_xyz_m"] = None
            joint.pop("q025_xyz_m", None)
            joint.pop("q975_xyz_m", None)
            joint.pop("uncertainty_scope", None)
            joint.pop("uncertainty_calibrated", None)


def _frozen_xyz(track: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    frame_indices = []
    poses = []
    for observation in track.get("observations") or []:
        xyz = np.full((23, 3), np.nan, dtype=np.float64)
        for joint in observation.get("metric_pose23") or []:
            index = int(joint.get("index", -1))
            value = joint.get("xyz_world_m")
            if 0 <= index < 23 and value is not None:
                xyz[index] = np.asarray(value, dtype=np.float64)
        frame_indices.append(int(observation.get("frame_index", -1)))
        poses.append(xyz)
    if not poses:
        return np.empty(0, dtype=np.int64), np.empty((0, 23, 3), dtype=np.float64)
    order = np.argsort(np.asarray(frame_indices, dtype=np.int64))
    return np.asarray(frame_indices, dtype=np.int64)[order], np.stack(poses)[order]


def apply_uncertainty_postprocess(
    *,
    base_state: str | Path,
    output_dir: str | Path,
    samples: int = 16,
    seed: int = 12345,
    mode: str = "selected_frame",
    uncertainty_max_nfev: int = 180,
    stage3_state: str | Path | None = None,
    camera_dir: str | Path | None = None,
    initializer_cache: str | Path | None = None,
    track_ids: list[str] | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive for uncertainty post-processing")
    if mode not in {"selected_frame", "full_window"}:
        raise ValueError("mode must be 'selected_frame' or 'full_window'")

    base_path = _existing_path(base_state, "base state")
    assert base_path is not None
    original = json.loads(base_path.read_text(encoding="utf-8"))
    state = copy.deepcopy(original)
    cfg = _config_from_state(state)
    cfg.uncertainty_samples = int(samples)
    cfg.uncertainty_seed = int(seed)
    cfg.uncertainty_mode = mode
    cfg.uncertainty_optimizer_max_nfev = int(uncertainty_max_nfev)

    inferred_stage3 = ((state.get("source_stage3") or {}).get("path"))
    inferred_camera = ((state.get("source_stage1") or {}).get("camera_state_directory"))
    inferred_initializer = ((state.get("source_initializer") or {}).get("cache"))
    stage3_path = _existing_path(stage3_state or inferred_stage3, "Stage-3 state")
    camera_path = _existing_path(camera_dir or inferred_camera, "camera", directory=True)
    initializer_value = initializer_cache if initializer_cache is not None else inferred_initializer
    initializer_path = _existing_path(initializer_value, "initializer cache") if initializer_value else None
    assert stage3_path is not None and camera_path is not None

    preflight = preflight_stage4(
        stage3_state=stage3_path,
        camera_dir=camera_path,
        initializer_cache=initializer_path,
        config=cfg,
    )
    if not preflight["ready"]:
        raise RuntimeError("Stage4 uncertainty preflight failed: " + ", ".join(preflight["errors"]))

    s3 = load_stage3_state(stage3_path)
    cams = CameraTimelineLite.load_dir(camera_path)
    init = InitializerCache.load_jsonl(initializer_path) if initializer_path else None
    selected_frame = int((state.get("replay_context") or {}).get("selected_frame", s3.selected_frame))
    state_tracks = {str(track.get("track_id")): track for track in state.get("tracks") or []}
    selected_outputs = {str(entry.get("track_id")): entry for entry in state.get("selected_frame_poses") or []}
    requested_track_ids = None if track_ids is None else {str(track_id) for track_id in track_ids}

    for source_track in s3.tracks:
        track_id = str(source_track.track_id)
        track = state_tracks.get(track_id)
        if track is None:
            continue
        if requested_track_ids is not None and track_id not in requested_track_ids:
            continue
        selected_status = str(track.get("selected_frame_pose_status", "MISSING"))
        if selected_status not in {"VALID", "DEGRADED"}:
            uncertainty = _not_applicable(track_id, selected_frame, samples, "selected_frame_metric_pose_unavailable", mode)
        else:
            frames, _, source_obs = _frames_for_track(s3, source_track, cams, init, cfg)
            selected_input = next((frame for frame in frames if frame.frame_index == selected_frame), None)
            if selected_input is None or selected_frame not in source_obs:
                uncertainty = _not_applicable(track_id, selected_frame, samples, "selected_frame_input_unavailable", mode)
            else:
                scale = dict(track.get("track_scale") or {})
                initializer_used = bool((track.get("initializer") or {}).get("used"))
                frozen_frames, frozen_poses = _frozen_xyz(track)
                frozen = FrozenTrackSolution(
                    estimated_height_m=float(scale.get("estimated_height_m", cfg.nominal_height_m)),
                    depth_beta_m=(None if scale.get("depth_beta_m") is None else float(scale.get("depth_beta_m"))),
                    initializer_used=initializer_used,
                    frame_indices=frozen_frames,
                    xyz_world_m=frozen_poses,
                )
                print(f"[Stage4 v0.2] {track_id} - uncertainty {mode}, N={samples}...", flush=True)
                uncertainty = pixel_sensitivity_monte_carlo(
                    frames,
                    frozen,
                    cfg,
                    selected_frame_index=selected_frame,
                    states23_by_frame={frame_index: obs.states23 for frame_index, obs in source_obs.items()},
                    track_id=track_id,
                    mode=mode,
                )
                print(
                    f"[Stage4 v0.2] {track_id} - {uncertainty['status']} | "
                    f"usable={uncertainty.get('samples_usable', 0)}/{samples}",
                    flush=True,
                )

        track["selected_frame_uncertainty"] = uncertainty
        observation = selected_observation(track, selected_frame)
        _attach_joint_uncertainty(observation if isinstance(observation, dict) else None, uncertainty)
        selected_output = selected_outputs.get(track_id)
        if isinstance(selected_output, dict):
            selected_output["uncertainty"] = uncertainty
            output_observation = selected_output.get("observation")
            if isinstance(output_observation, dict) and output_observation is not observation:
                _attach_joint_uncertainty(output_observation, uncertainty)

    eligible_track_count = sum(
        str(track.get("selected_frame_pose_status")) in {"VALID", "DEGRADED"}
        for track in state.get("tracks") or []
    )
    uncertainty_counts = Counter(
        str((track.get("selected_frame_uncertainty") or {}).get("status", "NOT_RUN"))
        for track in state.get("tracks") or []
    )

    outdir = Path(output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "metric_pose_3d_state.json"
    handoff_path = outdir / "stage5_handoff.json"
    state["stage4_version"] = "stage4-metric3d-0.2.0"
    state["configuration"] = cfg.to_dict()
    state["preflight"] = preflight
    state["uncertainty_analysis"] = {
        "schema_version": "stage4-uncertainty-analysis-1.0",
        "status_counts": {key: uncertainty_counts[key] for key in ("OK", "DEGRADED", "FAILED", "NOT_APPLICABLE")},
        "mode": mode,
        "samples_per_track": int(samples),
        "eligible_track_count": eligible_track_count,
        "completed_eligible_track_count": uncertainty_counts["OK"] + uncertainty_counts["DEGRADED"],
        "camera_sampled": False,
        "calibrated": False,
        "base_state": str(base_path),
        "base_state_sha256": _sha256(base_path),
        "base_stage4_version": original.get("stage4_version"),
        "base_optimizer_rerun": False,
        "track_filter": None if requested_track_ids is None else sorted(requested_track_ids),
    }
    acceptance = state.setdefault("acceptance_gate", {})
    acceptance["uncertainty_engineering_completed"] = bool(
        eligible_track_count > 0
        and uncertainty_counts["OK"] + uncertainty_counts["DEGRADED"] == eligible_track_count
    )
    acceptance["uncertainty_calibrated"] = False
    acceptance["research_accuracy_frozen"] = False
    acceptance["note"] = "v0.2 uncertainty is camera-fixed Stage-3 pixel sensitivity; calibration and numeric accuracy gates require real ground truth."
    semantics = state.setdefault("output_semantics", {})
    semantics["version"] = "2.0"
    semantics["uncertainty_postprocess"] = True
    semantics["base_xyz_modified"] = False
    semantics["attack_direction_applied"] = False
    state.setdefault("artifacts", {})["metric_pose_3d_state"] = str(state_path)
    state["artifacts"]["stage5_handoff"] = str(handoff_path)

    state_path.write_text(
        json.dumps(_json_safe(state), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    handoff = build_stage5_handoff(state, str(state_path))
    handoff_path.write_text(
        json.dumps(_json_safe(handoff), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return state
