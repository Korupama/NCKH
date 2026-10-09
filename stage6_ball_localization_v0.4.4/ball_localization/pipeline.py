from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import json

import cv2

from .camera import camera_state_for_frame
from .contracts import BallCandidate2D, BallFrameState, BallTrajectoryState
from .geometry import HybridGeometryConfig, TemporalRefinementConfig, estimate_frame, hybrid_trajectory_states, refine_temporal_trajectory
from .pitch_prior import apply_pitch_prior
from .providers import SoccerNetV3DYOLOProvider, load_stage2_candidates
from .stage1_context import load_replay_context_from_stage1
from .tracking import ViterbiConfig, select_ball_path
from .version import STAGE6_VERSION, runtime_provenance
from .visualization import render_minimap, render_selected_frame


def _infer_yolo_candidates(
    replay: Dict[str, Any],
    stage1_root: str | Path,
    provider: SoccerNetV3DYOLOProvider,
    *,
    pitch_margin_m: float,
    pitch_far_prior: float,
    progress_every: int = 10,
) -> Dict[int, list]:
    start, end = int(replay["window_start"]), int(replay["window_end"])
    if Path(replay["video_path"]).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
        if start != end or start != int(replay['selected_frame']):
            raise ValueError('A still image must describe exactly one selected frame')
        image = cv2.imread(str(replay['video_path']))
        if image is None:
            raise RuntimeError(f"Cannot decode image: {replay['video_path']}")
        if image.shape[:2] != (int(replay['image_height']), int(replay['image_width'])):
            raise ValueError('Image dimensions do not match replay context')
        return {start: apply_pitch_prior(provider.detect(image, start),
                    camera_state_for_frame(stage1_root, start),
                    margin_m=pitch_margin_m, far_prior=pitch_far_prior)}
    cap = cv2.VideoCapture(str(replay["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {replay['video_path']}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out: Dict[int, list] = {}
    total = end - start + 1
    for n, fi in enumerate(range(start, end + 1), start=1):
        ok, image = cap.read()
        if not ok or image is None:
            raise RuntimeError(f"Cannot decode frame {fi}")
        out[fi] = apply_pitch_prior(
            provider.detect(image, fi),
            camera_state_for_frame(stage1_root, fi),
            margin_m=pitch_margin_m,
            far_prior=pitch_far_prior,
        )
        if progress_every > 0 and (n == 1 or n % progress_every == 0 or n == total):
            print(f"[STAGE6 DETECT {n}/{total}] frame={fi} candidates={len(out[fi])}", flush=True)
    cap.release()
    return out


def _selected_path(
    *,
    candidates: Mapping[int, list[BallCandidate2D]],
    indices: list[int],
    replay: Mapping[str, Any],
    tracker_kind: str,
) -> tuple[Dict[int, BallCandidate2D | None], Dict[str, Any]]:
    if tracker_kind == "viterbi":
        selected = select_ball_path(
            dict(candidates),
            indices,
            image_width=int(replay["image_width"]),
            image_height=int(replay["image_height"]),
            config=ViterbiConfig(),
        )
        return selected, {"name": "offline-viterbi-single-ball", "missing_state": True}
    if tracker_kind == "top1":
        selected = {
            fi: (sorted(candidates.get(fi, []), key=lambda c: c.ranking_score, reverse=True)[0] if candidates.get(fi) else None)
            for fi in indices
        }
        return selected, {"name": "framewise-top1"}
    raise ValueError("tracker_kind must be viterbi or top1")


def _framewise_states(
    *,
    stage1_root: str | Path,
    selected: Mapping[int, BallCandidate2D | None],
    indices: list[int],
    fps: float,
    ball_radius_m: float,
    localization_mode: str,
) -> list[BallFrameState]:
    return [
        estimate_frame(
            camera_state_for_frame(stage1_root, fi),
            selected.get(fi),
            fps=fps,
            ball_radius_m=ball_radius_m,
            mode=localization_mode,
        )
        for fi in indices
    ]


def _temporal_states(
    *,
    stage1_root: str | Path,
    selected: Mapping[int, BallCandidate2D | None],
    indices: list[int],
    fps: float,
    ball_radius_m: float,
    config: TemporalRefinementConfig,
    fallback_frame_indices: set[int] | None = None,
) -> tuple[list[BallFrameState], Dict[str, Any]]:
    cameras = {fi: camera_state_for_frame(stage1_root, fi) for fi in indices}
    raw_states = {
        fi: estimate_frame(
            cameras[fi],
            selected.get(fi),
            fps=fps,
            ball_radius_m=ball_radius_m,
            mode="size-prior",
            pitch_margin_m=config.pitch_margin_m,
            max_size_prior_height_m=config.max_height_m,
        )
        for fi in indices
    }
    fallback_frames = set(fallback_frame_indices or set())
    temporal = refine_temporal_trajectory(
        cameras_by_frame=cameras,
        selected=selected,
        frame_indices=indices,
        fps=fps,
        ball_radius_m=ball_radius_m,
        config=config,
    )
    frames: list[BallFrameState] = []
    for fi in indices:
        raw = raw_states[fi]
        refined = temporal.frames[fi]
        temporal_diag = refined.to_dict()
        diagnostics = dict(raw.diagnostics)
        diagnostics["temporal"] = temporal_diag
        if refined.xyz_world_m is not None:
            frames.append(
                BallFrameState(
                    frame_index=fi,
                    timestamp_sec=raw.timestamp_sec,
                    candidate=raw.candidate,
                    observation_status="INTERPOLATED" if refined.center_imputed else raw.observation_status,
                    camera_status=raw.camera_status,
                    localization_status=refined.status,
                    ground_contact_xyz_world_m=raw.ground_contact_xyz_world_m,
                    ground_center_xyz_world_m=raw.ground_center_xyz_world_m,
                    size_prior_xyz_world_m=raw.size_prior_xyz_world_m,
                    selected_center_xyz_world_m=refined.xyz_world_m,
                    selected_method="TEMPORAL_DIAMETER_RANGE_TRAJECTORY",
                    diagnostics=diagnostics,
                )
            )
            continue
        # Explicit fallback keeps the production path useful at the boundary of a
        # short window while clearly exposing that temporal support was absent.
        if fi in fallback_frames and raw.selected_center_xyz_world_m is not None:
            frames.append(
                BallFrameState(
                    frame_index=fi,
                    timestamp_sec=raw.timestamp_sec,
                    candidate=raw.candidate,
                    observation_status=raw.observation_status,
                    camera_status=raw.camera_status,
                    localization_status=f"DEGRADED_TEMPORAL_FALLBACK_{raw.localization_status}",
                    ground_contact_xyz_world_m=raw.ground_contact_xyz_world_m,
                    ground_center_xyz_world_m=raw.ground_center_xyz_world_m,
                    size_prior_xyz_world_m=raw.size_prior_xyz_world_m,
                    selected_center_xyz_world_m=raw.selected_center_xyz_world_m,
                    selected_method="MONOCULAR_BALL_SIZE_PRIOR_FALLBACK",
                    diagnostics=diagnostics,
                )
            )
        else:
            frames.append(
                BallFrameState(
                    frame_index=fi,
                    timestamp_sec=raw.timestamp_sec,
                    candidate=raw.candidate,
                    observation_status=raw.observation_status,
                    camera_status=raw.camera_status,
                    localization_status=refined.status,
                    ground_contact_xyz_world_m=raw.ground_contact_xyz_world_m,
                    ground_center_xyz_world_m=raw.ground_center_xyz_world_m,
                    size_prior_xyz_world_m=raw.size_prior_xyz_world_m,
                    selected_center_xyz_world_m=None,
                    selected_method=None,
                    diagnostics=diagnostics,
                )
            )
    return frames, temporal.diagnostics


def _selected_frame_payload(
    frames: list[BallFrameState],
    selected_frame: int,
    ball_radius_m: float,
) -> Dict[str, Any]:
    state = next(x for x in frames if x.frame_index == int(selected_frame))
    temporal_diag = (state.diagnostics or {}).get("temporal")
    payload: Dict[str, Any] = {
        "frame_index": int(selected_frame),
        "status": state.localization_status,
        "candidate": None if state.candidate is None else state.candidate.to_dict(),
        "center_xyz_world_m": state.selected_center_xyz_world_m,
        "ground_contact_xyz_world_m": state.ground_contact_xyz_world_m,
        "ground_center_xyz_world_m": state.ground_center_xyz_world_m,
        "size_prior_xyz_world_m": state.size_prior_xyz_world_m,
        "method": state.selected_method,
        "camera_status": state.camera_status,
        "usable_for_offside_longitudinal_coordinate": bool(state.selected_center_xyz_world_m is not None),
    }
    if temporal_diag is not None:
        payload["temporal"] = temporal_diag
    if state.selected_center_xyz_world_m is not None:
        x, y, z = map(float, state.selected_center_xyz_world_m)
        payload.update({
            "X_world_m": x,
            "Y_world_m": y,
            "Z_world_m": z,
            "ball_center_x_extent_m": [x - float(ball_radius_m), x + float(ball_radius_m)],
        })
    return payload


def _finalize_state(
    *,
    stage1_root: str | Path,
    output_dir: str | Path,
    replay: Dict[str, Any],
    detector_info: Dict[str, Any],
    tracker_info: Dict[str, Any],
    candidates: Mapping[int, list[BallCandidate2D]],
    frames: list[BallFrameState],
    indices: list[int],
    ball_radius_m: float,
    localization_mode: str,
    pitch_margin_m: float,
    pitch_far_prior: float,
    temporal_diagnostics: Dict[str, Any] | None = None,
) -> BallTrajectoryState:
    t0 = int(replay["selected_frame"])
    selected_frame_ball = _selected_frame_payload(frames, t0, ball_radius_m)
    pitch_info = dict(camera_state_for_frame(stage1_root, t0).pitch or {})
    pitch_length_m = float(pitch_info.get("length_m", 105.0))
    pitch_width_m = float(pitch_info.get("width_m", 68.0))
    selected_count = sum(frame.candidate is not None for frame in frames)
    valid3d = sum(frame.selected_center_xyz_world_m is not None for frame in frames)
    uncertain = sum(frame.localization_status == "UNCERTAIN_AIRBORNE_OR_BBOX_SIZE" for frame in frames)
    diagnostics: Dict[str, Any] = {
        "runtime_provenance": runtime_provenance(),
        "window_frames": len(indices),
        "frames_with_candidates": sum(bool(candidates.get(fi)) for fi in indices),
        "frames_selected_by_tracker": int(selected_count),
        "frames_with_selected_3d": int(valid3d),
        "uncertain_airborne_or_bbox_size_frames": int(uncertain),
        "pitch_prior": {"margin_m": pitch_margin_m, "far_prior": pitch_far_prior},
        "pitch_dimensions": {"length_m": pitch_length_m, "width_m": pitch_width_m},
    }
    if temporal_diagnostics is not None:
        diagnostics["temporal_refinement"] = temporal_diagnostics

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = BallTrajectoryState(
        "ball-trajectory-state-1.1",
        STAGE6_VERSION,
        replay,
        detector_info,
        tracker_info,
        float(ball_radius_m),
        localization_mode,
        "VALID" if selected_frame_ball["usable_for_offside_longitudinal_coordinate"] else "DEGRADED",
        {str(fi): [c.to_dict() for c in candidates.get(fi, [])] for fi in indices},
        frames,
        selected_frame_ball,
        diagnostics,
        {},
    )
    path = state.save_json(out / "ball_trajectory_state.json")
    state.artifacts["ball_trajectory_state"] = str(path)
    try:
        state.artifacts["selected_frame_overlay"] = str(
            render_selected_frame(replay["video_path"], selected_frame_ball, out / "selected_frame_ball.png")
        )
    except Exception as exc:  # visualization is non-blocking by contract
        state.diagnostics["selected_frame_visualization_warning"] = str(exc)
    try:
        state.artifacts["trajectory_minimap"] = str(
            render_minimap(frames, t0, out / "ball_trajectory_minimap.png", length_m=pitch_length_m, width_m=pitch_width_m)
        )
    except Exception as exc:
        state.diagnostics["minimap_visualization_warning"] = str(exc)
    state.save_json(path)
    return state


def run_stage6(
    *,
    stage1_root: str | Path,
    output_dir: str | Path,
    provider_kind: str = "yolo",
    weights: str | Path | None = None,
    stage2_entity_tracks: str | Path | None = None,
    video_path: str | Path | None = None,
    conf_floor: float = 0.05,
    top_k: int = 10,
    imgsz: int = 1920,
    device: str = "cpu",
    tracker_kind: str = "viterbi",
    localization_mode: str = "ground-first",
    ball_radius_m: float = 0.11,
    pitch_margin_m: float = 12.0,
    pitch_far_prior: float = 0.20,
    progress_every: int = 10,
    temporal_config: TemporalRefinementConfig | None = None,
    hybrid_config: HybridGeometryConfig | None = None,
) -> BallTrajectoryState:
    if provider_kind == "stage2-sst":
        if stage2_entity_tracks is None:
            raise ValueError("stage2_entity_tracks required")
        replay, candidates = load_stage2_candidates(stage2_entity_tracks)
        detector_info = {"name": "stage2-sst-ball", "source": str(Path(stage2_entity_tracks).resolve())}
        for fi, rows in candidates.items():
            try:
                apply_pitch_prior(rows, camera_state_for_frame(stage1_root, fi), margin_m=pitch_margin_m, far_prior=pitch_far_prior)
            except FileNotFoundError:
                pass
    elif provider_kind in {"yolo", "fusion", "yolo-first"}:
        if weights is None:
            raise ValueError("weights required for YOLO provider")
        if provider_kind == 'yolo-first':
            if stage2_entity_tracks is None:
                raise ValueError('yolo-first requires stage2_entity_tracks')
            replay, auxiliary = load_stage2_candidates(stage2_entity_tracks)
            if video_path is not None and Path(video_path).resolve() != Path(replay['video_path']).resolve():
                raise ValueError('Replay mismatch: video_path')
        else:
            replay = load_replay_context_from_stage1(stage1_root, video_path)
        provider = SoccerNetV3DYOLOProvider(weights, conf_floor=conf_floor, top_k=top_k, imgsz=imgsz, device=device)
        candidates = _infer_yolo_candidates(
            replay,
            stage1_root,
            provider,
            pitch_margin_m=pitch_margin_m,
            pitch_far_prior=pitch_far_prior,
            progress_every=progress_every,
        )
        detector_info = provider.info()
        if provider_kind == 'yolo-first':
            from .providers.fusion import prefer_primary_candidates
            for fi, rows in auxiliary.items():
                apply_pitch_prior(rows, camera_state_for_frame(stage1_root, fi),
                                  margin_m=pitch_margin_m, far_prior=pitch_far_prior)
            candidates, selection = prefer_primary_candidates(candidates, auxiliary)
            detector_info = {'name': 'yolo-first', 'primary': detector_info,
                             'auxiliary_source': str(stage2_entity_tracks), **selection}
        if provider_kind == "fusion":
            from .contact import validate_replay
            from .providers.fusion import fuse_candidates
            if stage2_entity_tracks is None:
                raise ValueError("fusion requires stage2_entity_tracks")
            auxiliary_replay, auxiliary = load_stage2_candidates(stage2_entity_tracks)
            validate_replay(replay, auxiliary_replay)
            for fi, rows in auxiliary.items():
                if int(replay['window_start']) <= fi <= int(replay['window_end']):
                    apply_pitch_prior(rows, camera_state_for_frame(stage1_root,fi), margin_m=pitch_margin_m, far_prior=pitch_far_prior)
            candidates, fusion_counts = fuse_candidates(candidates, auxiliary)
            detector_info = {'name':'fusion','primary':detector_info,'auxiliary_source':str(stage2_entity_tracks),'counts':fusion_counts}
    else:
        raise ValueError("provider_kind must be yolo, stage2-sst, fusion or yolo-first")

    if str(replay.get("coordinate_space")) != "RAW_DISTORTED_PIXEL":
        raise ValueError("RAW_DISTORTED_PIXEL required")
    indices = list(range(int(replay["window_start"]), int(replay["window_end"]) + 1))
    selected, tracker_info = _selected_path(candidates=candidates, indices=indices, replay=replay, tracker_kind=tracker_kind)
    fps = float(replay["fps"])

    temporal_diagnostics = None
    if localization_mode == "temporal-3d":
        cfg = temporal_config or TemporalRefinementConfig(pitch_margin_m=min(float(pitch_margin_m), 6.0))
        frames, temporal_diagnostics = _temporal_states(
            stage1_root=stage1_root,
            selected=selected,
            indices=indices,
            fps=fps,
            ball_radius_m=ball_radius_m,
            config=cfg,
            fallback_frame_indices={int(replay["selected_frame"])},
        )
        tracker_info = dict(tracker_info)
        tracker_info["temporal_geometry"] = "diameter-refinement+ray-range-trajectory"
    elif localization_mode == "hybrid-3d":
        cfg = hybrid_config or HybridGeometryConfig()
        cameras = {fi: camera_state_for_frame(stage1_root, fi) for fi in indices}
        frames, hybrid_diagnostics = hybrid_trajectory_states(
            cameras_by_frame=cameras, selected=selected, frame_indices=indices, fps=fps,
            ball_radius_m=ball_radius_m, config=cfg, temporal_config=temporal_config,
        )
        temporal_diagnostics = {"hybrid_3d": hybrid_diagnostics}
        tracker_info = dict(tracker_info)
        tracker_info["temporal_geometry"] = "hybrid-ground-ballistic-size-prior"
    elif localization_mode in {"ground-first", "ground-only", "size-prior"}:
        frames = _framewise_states(
            stage1_root=stage1_root,
            selected=selected,
            indices=indices,
            fps=fps,
            ball_radius_m=ball_radius_m,
            localization_mode=localization_mode,
        )
    else:
        raise ValueError("localization_mode must be ground-first, ground-only, size-prior, temporal-3d or hybrid-3d")

    return _finalize_state(
        stage1_root=stage1_root,
        output_dir=output_dir,
        replay=replay,
        detector_info=detector_info,
        tracker_info=tracker_info,
        candidates=candidates,
        frames=frames,
        indices=indices,
        ball_radius_m=ball_radius_m,
        localization_mode=localization_mode,
        pitch_margin_m=pitch_margin_m,
        pitch_far_prior=pitch_far_prior,
        temporal_diagnostics=temporal_diagnostics,
    )


def _candidate_from_dict(payload: Mapping[str, Any] | None) -> BallCandidate2D | None:
    if not payload:
        return None
    return BallCandidate2D(
        frame_index=int(payload["frame_index"]),
        candidate_id=str(payload["candidate_id"]),
        bbox_xyxy=[float(x) for x in payload["bbox_xyxy"]],
        center_uv=[float(x) for x in payload["center_uv"]],
        detector_score=float(payload.get("detector_score", 0.0)),
        source=str(payload.get("source", "cached-stage6-state")),
        diameter_px=float(payload.get("diameter_px", 0.0)),
        pitch_prior=float(payload.get("pitch_prior", 1.0)),
        ranking_score=float(payload.get("ranking_score", 0.0)),
        coordinate_space=str(payload.get("coordinate_space", "RAW_DISTORTED_PIXEL")),
        metadata=dict(payload.get("metadata") or {}),
    )


def refine_existing_state(
    *,
    state_json: str | Path,
    stage1_root: str | Path,
    output_dir: str | Path,
    temporal_config: TemporalRefinementConfig | None = None,
    hybrid_config: HybridGeometryConfig | None = None,
) -> BallTrajectoryState:
    """Apply v0.4 temporal geometry to an existing Stage-6 state without YOLO rerun."""
    source_path = Path(state_json).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    replay = dict(source["replay_context"])
    if str(replay.get("coordinate_space")) != "RAW_DISTORTED_PIXEL":
        raise ValueError("RAW_DISTORTED_PIXEL required")
    indices = list(range(int(replay["window_start"]), int(replay["window_end"]) + 1))
    frame_map = {int(row["frame_index"]): row for row in source.get("frames", [])}
    selected = {fi: _candidate_from_dict((frame_map.get(fi) or {}).get("candidate")) for fi in indices}
    candidates: Dict[int, list[BallCandidate2D]] = {}
    for fi in indices:
        rows = (source.get("candidates_by_frame") or {}).get(str(fi), [])
        parsed = [_candidate_from_dict(row) for row in rows]
        candidates[fi] = [x for x in parsed if x is not None]
    cfg = temporal_config or TemporalRefinementConfig()
    if hybrid_config is None:
        frames, temporal_diagnostics = _temporal_states(
            stage1_root=stage1_root, selected=selected, indices=indices, fps=float(replay["fps"]),
            ball_radius_m=float(source.get("ball_radius_m", 0.11)), config=cfg,
            fallback_frame_indices={int(replay["selected_frame"])},
        )
        localization_mode = "temporal-3d"
    else:
        cameras = {fi: camera_state_for_frame(stage1_root, fi) for fi in indices}
        frames, hybrid = hybrid_trajectory_states(
            cameras_by_frame=cameras, selected=selected, frame_indices=indices, fps=float(replay["fps"]),
            ball_radius_m=float(source.get("ball_radius_m", 0.11)), config=hybrid_config, temporal_config=cfg,
        )
        temporal_diagnostics = {"hybrid_3d": hybrid}
        localization_mode = "hybrid-3d"
    detector_info = dict(source.get("detector") or {})
    tracker_info = dict(source.get("tracker") or {})
    tracker_info["temporal_geometry"] = "hybrid-ground-ballistic-size-prior" if hybrid_config is not None else "diameter-refinement+ray-range-trajectory"
    temporal_diagnostics["source_state_json"] = str(source_path)
    temporal_diagnostics["source_stage6_version"] = source.get("stage6_version")
    return _finalize_state(
        stage1_root=stage1_root,
        output_dir=output_dir,
        replay=replay,
        detector_info=detector_info,
        tracker_info=tracker_info,
        candidates=candidates,
        frames=frames,
        indices=indices,
        ball_radius_m=float(source.get("ball_radius_m", 0.11)),
        localization_mode=localization_mode,
        pitch_margin_m=float(cfg.pitch_margin_m),
        pitch_far_prior=float(((source.get("diagnostics") or {}).get("pitch_prior") or {}).get("far_prior", 0.20)),
        temporal_diagnostics=temporal_diagnostics,
    )
