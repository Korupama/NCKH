from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import hashlib
import json
import numpy as np

from .adapters import load_stage2_state, load_stage3_state, observation_by_frame, validate_stage2_stage3_alignment
from .clustering import fit_two_teams
from .config import Stage5Config
from .features import aggregate_features, extract_color_feature, fuse_region_features
from .goalkeeper import assign_goalkeeper, lower_body_centroids_by_team
from .regions import region_polygon
from .video import read_frames, video_metadata
from .visualization import render_selected_frame

STAGE5_VERSION = "stage5-team-affiliation-0.1.0"
OUTPUT_SCHEMA = "team-affiliation-state-1.0"
HANDOFF_SCHEMA = "stage5-downstream-handoff-1.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_frame_indices(track: Mapping[str, Any], config: Stage5Config) -> List[int]:
    frames = sorted(int(o["frame_index"]) for o in track.get("observations", []))
    frames = frames[::config.sample_every_n_frames]
    if len(frames) > config.max_samples_per_track:
        idx = np.linspace(0, len(frames)-1, config.max_samples_per_track).round().astype(int)
        frames = [frames[i] for i in idx]
    return sorted(set(frames))


def _selected_bbox(track: Mapping[str, Any], selected_frame: int):
    for o in track.get("observations", []):
        if int(o["frame_index"]) == selected_frame:
            return o.get("source_bbox_xyxy")
    return None


def _image_x_samples(track: Mapping[str, Any], frame_indices: List[int]) -> List[float]:
    wanted = set(frame_indices)
    values: List[float] = []
    for obs in track.get("observations", []):
        if int(obs.get("frame_index", -1)) not in wanted:
            continue
        bbox = obs.get("source_bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, x2 = float(bbox[0]), float(bbox[2])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x1) and np.isfinite(x2):
            values.append((x1 + x2) * 0.5)
    return values


def _load_stage4_pitch_x(
    stage4_handoff: str | Path | None,
    selected_frame: int,
) -> Tuple[Optional[Path], Dict[str, List[float]], Optional[Dict[str, Any]]]:
    if stage4_handoff is None:
        return None, {}, None
    path = Path(stage4_handoff).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage4 handoff not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "stage4-downstream-handoff-2.1":
        raise ValueError(f"Unsupported Stage4 handoff schema: {data.get('schema_version')!r}")
    stage4_frame = int(data.get("selected_frame"))
    if stage4_frame != selected_frame:
        raise ValueError(
            f"Stage3 selected frame {selected_frame} does not match Stage4 selected frame {stage4_frame}"
        )

    pitch_x: Dict[str, List[float]] = {}
    rejected_outside_pitch: List[str] = []
    rejected_invalid_track: List[str] = []
    for track in data.get("tracks", []):
        tid = str(track.get("track_id") or "")
        if not tid:
            continue
        selected_status = str(track.get("selected_frame_status") or "").upper()
        if selected_status and selected_status != "VALID":
            rejected_invalid_track.append(tid)
            continue
        values: List[float] = []
        for obs in track.get("observations", []):
            if int(obs.get("frame_index", -1)) != selected_frame:
                continue
            quality = obs.get("quality") if isinstance(obs.get("quality"), dict) else {}
            if str(quality.get("ground_rejection_reason") or "").lower() == "outside_pitch_bounds":
                rejected_outside_pitch.append(tid)
                continue
            root = obs.get("root_world_m")
            if not isinstance(root, (list, tuple)) or len(root) < 1:
                continue
            try:
                x = float(root[0])
            except (TypeError, ValueError):
                continue
            if np.isfinite(x):
                values.append(x)
        if values:
            pitch_x[tid] = values
    source = {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": data.get("schema_version"),
        "selected_frame": stage4_frame,
        "tracks_with_pitch_x": len(pitch_x),
        "tracks_rejected_outside_pitch": sorted(set(rejected_outside_pitch)),
        "tracks_rejected_invalid": sorted(set(rejected_invalid_track)),
    }
    return path, pitch_x, source


def run_stage5(
    *,
    stage3_state: str | Path,
    video_path: str | Path | None = None,
    output_dir: str | Path,
    stage2_state: str | Path | None = None,
    stage4_handoff: str | Path | None = None,
    config: Optional[Stage5Config] = None,
) -> Dict[str, Any]:
    config = config or Stage5Config()
    config.validate()
    stage3_path = Path(stage3_state).expanduser().resolve()
    s3 = load_stage3_state(stage3_path)
    s2 = load_stage2_state(stage2_state)
    alignment = validate_stage2_stage3_alignment(s2, s3)
    if s2 is not None and not alignment.get("ready", False):
        raise ValueError(f"Stage2/Stage3 alignment failed: {alignment}")

    rc = s3.get("replay_context") or {}
    selected_frame = int(rc.get("selected_frame"))
    stage4_path, pitch_x_by_track, source_stage4 = _load_stage4_pitch_x(
        stage4_handoff, selected_frame
    )
    if video_path is None:
        video_path = rc.get("video_path")
    if not video_path or not Path(video_path).is_file():
        raise FileNotFoundError(
            "Stage5 requires the replay video to extract kit appearance. "
            f"Not found: {video_path!r}"
        )
    video_path = str(Path(video_path).expanduser().resolve())
    meta = video_metadata(video_path)
    expected_size = (int(rc.get("image_width", meta["width"])), int(rc.get("image_height", meta["height"])))
    if (meta["width"], meta["height"]) != expected_size:
        raise ValueError(f"Video size {(meta['width'],meta['height'])} does not match Stage3 {expected_size}")

    tracks = [dict(t) for t in s3.get("tracks", [])]
    candidate_tracks = [t for t in tracks if str(t.get("upstream_role")) in {"player", "goalkeeper", "referee"}]
    needed_frames = set()
    sampled_by_track: Dict[str, List[int]] = {}
    for t in candidate_tracks:
        fs = _sample_frame_indices(t, config)
        sampled_by_track[str(t["track_id"])] = fs
        needed_frames.update(fs)
    needed_frames.add(selected_frame)
    frames = read_frames(video_path, needed_frames)

    per_track: Dict[str, Dict[str, Any]] = {}
    torso_features: Dict[str, np.ndarray] = {}
    lower_features: Dict[str, np.ndarray] = {}
    fused_features: Dict[str, np.ndarray] = {}
    image_x_by_track: Dict[str, List[float]] = {}
    for t in candidate_tracks:
        tid = str(t["track_id"])
        role = str(t.get("upstream_role") or "other")
        obs_idx = observation_by_frame(t)
        torso_vecs: List[np.ndarray] = []
        lower_vecs: List[np.ndarray] = []
        samples: List[Dict[str, Any]] = []
        for fi in sampled_by_track[tid]:
            obs = obs_idx.get(fi)
            frame = frames.get(fi)
            if obs is None or frame is None:
                continue
            torso_poly, torso_source = region_polygon(obs, "torso", config.allow_bbox_torso_fallback)
            lower_poly, lower_source = region_polygon(obs, "lower", config.allow_bbox_lower_body_fallback)
            torso_feat = None; torso_diag = {"status":"UNAVAILABLE"}
            lower_feat = None; lower_diag = {"status":"UNAVAILABLE"}
            if torso_poly is not None:
                torso_feat, torso_diag = extract_color_feature(frame, torso_poly, config)
                if torso_feat is not None:
                    torso_vecs.append(torso_feat)
            if lower_poly is not None:
                lower_feat, lower_diag = extract_color_feature(frame, lower_poly, config)
                if lower_feat is not None:
                    lower_vecs.append(lower_feat)
            samples.append({
                "frame_index": fi,
                "pose_status": obs.get("pose_status"),
                "torso_source": torso_source,
                "torso_status": torso_diag.get("status"),
                "lower_source": lower_source,
                "lower_status": lower_diag.get("status"),
            })
        tf = aggregate_features(torso_vecs)
        lf = aggregate_features(lower_vecs)
        if tf is not None:
            torso_features[tid] = tf
        if lf is not None:
            lower_features[tid] = lf
        fused = (fuse_region_features(
            tf, lf,
            torso_weight=config.torso_feature_weight,
            lower_weight=config.lower_feature_weight,
        ) if config.feature_fusion_enabled else tf)
        if fused is not None:
            fused_features[tid] = fused
        image_x_by_track[tid] = _image_x_samples(t, sampled_by_track[tid])
        per_track[tid] = {
            "track_id": tid,
            "role": role,
            "upstream_role_score": t.get("upstream_role_score"),
            "upstream_identity_confidence": t.get("upstream_identity_confidence"),
            "selected_frame_bbox_xyxy": _selected_bbox(t, selected_frame),
            "appearance": {
                "sampled_frames": sampled_by_track[tid],
                "valid_torso_frames": len(torso_vecs),
                "valid_lower_body_frames": len(lower_vecs),
                "samples": samples,
            },
        }

    outside_pitch_tracks = set(
        (source_stage4 or {}).get("tracks_rejected_outside_pitch", [])
    )
    outfield = {
        tid: fused_features[tid]
        for tid, rec in per_track.items()
        if rec["role"] == "player"
        and tid not in outside_pitch_tracks
        and tid in fused_features
        and rec["appearance"]["valid_torso_frames"] >= config.min_valid_torso_frames
    }
    cluster_error = None
    try:
        cluster = fit_two_teams(outfield, config)
        team_lower_centroids = lower_body_centroids_by_team(cluster.labels, cluster.status, lower_features)
    except ValueError as exc:
        cluster = None
        cluster_error = str(exc)
        team_lower_centroids = {}

    team_pitch_x: Dict[int, List[float]] = {0: [], 1: []}
    team_image_x: Dict[int, List[float]] = {0: [], 1: []}
    if cluster is not None:
        for tid, label in cluster.labels.items():
            if cluster.status.get(tid) != "VALID":
                continue
            team = int(label)
            team_pitch_x[team].extend(pitch_x_by_track.get(tid, []))
            if tid not in outside_pitch_tracks:
                team_image_x[team].extend(image_x_by_track.get(tid, []))

    output_tracks: List[Dict[str, Any]] = []
    for tid in sorted(per_track):
        rec = dict(per_track[tid])
        role = rec["role"]
        if role == "referee":
            rec.update({
                "team_id": None,
                "team_status": "NOT_APPLICABLE",
                "assignment_method": "REFEREE_EXCLUDED",
            })
        elif role == "player":
            if tid in outside_pitch_tracks:
                rec.update({
                    "team_id": None,
                    "team_status": "UNKNOWN",
                    "assignment_method": "OUTSIDE_PITCH_EXCLUDED",
                })
            elif cluster is None or tid not in cluster.labels:
                rec.update({"team_id": None, "team_status": "UNKNOWN", "assignment_method": "INSUFFICIENT_APPEARANCE" if tid not in fused_features else "TEAM_CLUSTERING_UNAVAILABLE"})
            else:
                status = cluster.status[tid]
                rec.update({
                    "team_id": int(cluster.labels[tid]) if status == "VALID" else None,
                    "team_status": status,
                    "assignment_method": "OUTFIELD_KMEANS_FUSED_APPEARANCE",
                    "cluster_id_raw": int(cluster.labels[tid]),
                    "cluster_distances": cluster.distances[tid],
                    "cluster_margin": cluster.margins[tid],
                })
        elif role == "goalkeeper":
            result = assign_goalkeeper(
                lower_features.get(tid),
                team_lower_centroids,
                config,
                gk_pitch_x=pitch_x_by_track.get(tid),
                gk_image_x=image_x_by_track.get(tid),
                team_pitch_x=team_pitch_x,
                team_image_x=team_image_x,
            )
            rec.update({
                "team_id": result.get("team_id"),
                "team_status": result.get("status"),
                "assignment_method": result.get("method"),
                "goalkeeper_assignment": result,
            })
        else:
            rec.update({"team_id": None, "team_status": "NOT_APPLICABLE", "assignment_method": "ROLE_NOT_SUPPORTED"})
        output_tracks.append(rec)

    selected_present = [r for r in output_tracks if r.get("selected_frame_bbox_xyxy") and r["role"] in {"player","goalkeeper"}]
    selected_usable = [r for r in selected_present if r.get("team_status") == "VALID"]
    unknown = [r for r in output_tracks if r.get("team_status") == "UNKNOWN"]
    refs = [r for r in output_tracks if r.get("role") == "referee"]

    out_dir = Path(output_dir).expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": OUTPUT_SCHEMA,
        "stage5_version": STAGE5_VERSION,
        "replay_context": rc,
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "source_stage3": {
            "path": str(stage3_path),
            "sha256": _sha256(stage3_path),
            "schema_version": s3.get("schema_version"),
            "stage3_version": s3.get("stage3_version"),
        },
        "source_stage2": None if stage2_state is None else {
            "path": str(Path(stage2_state).expanduser().resolve()),
            "alignment": alignment,
        },
        "source_stage4": source_stage4,
        "configuration": config.to_dict(),
        "method": {
            "outfield": "POSE_GUIDED_FUSED_TORSO_LOWER_COLOR_KMEANS_K2",
            "goalkeeper": "SPATIAL_PITCH_OR_IMAGE_THEN_LOWER_BODY_APPEARANCE_FAIL_CLOSED",
            "referee": "EXCLUDED",
            "pretrained_model": None,
            "feature_fusion": {
                "enabled": config.feature_fusion_enabled,
                "torso_weight": config.torso_feature_weight,
                "lower_weight": config.lower_feature_weight,
                "missing_region_renormalization": True,
            },
        },
        "tracks": output_tracks,
        "metrics": {
            "tracks_total": len(output_tracks),
            "outfield_cluster_input_tracks": len(outfield),
            "referee_tracks_excluded": len(refs),
            "unknown_tracks": len(unknown),
            "unknown_rate_all_tracks": float(len(unknown) / max(1, len(output_tracks))),
            "SelectedFrameTeamCoverage": float(len(selected_usable) / max(1, len(selected_present))),
            "selected_frame_team_usable": len(selected_usable),
            "selected_frame_team_candidates": len(selected_present),
        },
        "clustering": {
            "status": "VALID" if cluster is not None else "UNAVAILABLE",
            "reason": cluster_error,
            "cluster_sizes": None if cluster is None else cluster.diagnostics.get("cluster_sizes"),
            "inertia": None if cluster is None else cluster.diagnostics.get("inertia"),
            "distance_thresholds": None if cluster is None else cluster.diagnostics.get("distance_thresholds"),
            "cluster_ids_are_permutation_invariant": True,
        },
        "acceptance_gate": {
            "implementation": "VALIDATED_SYNTHETIC_ONLY",
            "real_team_accuracy": "NOT_EVALUATED",
            "selected_frame_team_coverage_available": True,
            "research_accuracy_frozen": False,
        },
        "diagnostics": {
            "stage2_stage3_alignment": alignment,
            "stage4_dependency": stage4_path is not None,
            "attacking_team_applied": False,
            "attack_direction_applied": False,
            "offside_semantics_applied": False,
        },
        "artifacts": {},
    }
    state_path = out_dir / "team_affiliation_state.json"
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "stage5_version": STAGE5_VERSION,
        "selected_frame": selected_frame,
        "track_team": {
            r["track_id"]: {
                "role": r["role"],
                "team_id": r.get("team_id"),
                "team_status": r.get("team_status"),
                "assignment_method": r.get("assignment_method"),
            }
            for r in output_tracks
        },
        "cluster_ids_are_arbitrary": True,
        "attacking_team_not_resolved": True,
        "research_accuracy_frozen": False,
    }
    handoff_path = out_dir / "stage5_downstream_handoff.json"
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    handoff_path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    state["artifacts"].update({"team_affiliation_state": str(state_path), "downstream_handoff": str(handoff_path)})

    if selected_frame in frames:
        overlay = render_selected_frame(frames[selected_frame], output_tracks, out_dir / "selected_frame_team_affiliation.png")
        state["artifacts"]["selected_frame_overlay"] = str(overlay)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state
