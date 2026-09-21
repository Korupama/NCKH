from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import hashlib
import json
import numpy as np

from .adapters import load_stage2_state, load_stage3_state, observation_by_frame, validate_stage2_stage3_alignment
from .clustering import fit_two_teams
from .config import Stage5Config
from .features import aggregate_features, extract_color_feature
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


def run_stage5(
    *,
    stage3_state: str | Path,
    video_path: str | Path | None = None,
    output_dir: str | Path,
    stage2_state: str | Path | None = None,
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

    outfield = {
        tid: torso_features[tid]
        for tid, rec in per_track.items()
        if rec["role"] == "player" and tid in torso_features and rec["appearance"]["valid_torso_frames"] >= config.min_valid_torso_frames
    }
    cluster_error = None
    try:
        cluster = fit_two_teams(outfield, config)
        team_lower_centroids = lower_body_centroids_by_team(cluster.labels, cluster.status, lower_features)
    except ValueError as exc:
        cluster = None
        cluster_error = str(exc)
        team_lower_centroids = {}

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
            if cluster is None or tid not in cluster.labels:
                rec.update({"team_id": None, "team_status": "UNKNOWN", "assignment_method": "INSUFFICIENT_TORSO_APPEARANCE" if tid not in torso_features else "TEAM_CLUSTERING_UNAVAILABLE"})
            else:
                status = cluster.status[tid]
                rec.update({
                    "team_id": int(cluster.labels[tid]) if status == "VALID" else None,
                    "team_status": status,
                    "assignment_method": "OUTFIELD_KMEANS_TORSO_COLOR",
                    "cluster_id_raw": int(cluster.labels[tid]),
                    "cluster_distances": cluster.distances[tid],
                    "cluster_margin": cluster.margins[tid],
                })
        elif role == "goalkeeper":
            result = assign_goalkeeper(lower_features.get(tid), team_lower_centroids, config)
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
        "configuration": config.to_dict(),
        "method": {
            "outfield": "POSE_GUIDED_TORSO_COLOR_KMEANS_K2",
            "goalkeeper": "LOWER_BODY_APPEARANCE_AFFINITY_FAIL_CLOSED",
            "referee": "EXCLUDED",
            "pretrained_model": None,
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
            "stage4_dependency": False,
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
