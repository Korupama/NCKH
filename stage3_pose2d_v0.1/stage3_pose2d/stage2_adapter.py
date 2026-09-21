from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import json

from .wholebody133 import WHOLEBODY_KEYPOINT_NAMES

RAW_PIXEL_SPACE = "RAW_DISTORTED_PIXEL"

@dataclass
class Stage2Bundle:
    entity_state_path: Path
    rtmw_cache_path: Path
    handoff_path: Path
    entity_state: Dict[str, Any]
    rtmw_cache: Dict[str, Any]
    handoff: Dict[str, Any]
    validation: Dict[str, Any]

    @property
    def replay_context(self) -> Dict[str, Any]:
        return dict(self.entity_state.get("replay_context") or {})

    @property
    def selected_frame(self) -> int:
        return int(self.replay_context["selected_frame"])

    @property
    def candidate_track_ids(self) -> List[str]:
        return [str(x) for x in self.handoff.get("candidate_track_ids", [])]

    def track_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {str(t["track_id"]): t for t in self.entity_state.get("tracks", [])}

    def cache_track_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {str(k): dict(v) for k, v in (self.rtmw_cache.get("tracks") or {}).items()}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_stage2_paths(
    *,
    stage2_dir: str | Path | None = None,
    entity_state: str | Path | None = None,
    rtmw_cache: str | Path | None = None,
    handoff: str | Path | None = None,
) -> Tuple[Path, Path, Path]:
    if stage2_dir is not None:
        base = Path(stage2_dir).expanduser().resolve()
        entity_state = entity_state or base / "stage2_entity_tracks.json"
        rtmw_cache = rtmw_cache or base / "stage2_rtmw_track_cache.json"
        handoff = handoff or base / "stage3_handoff.json"
    if entity_state is None or rtmw_cache is None or handoff is None:
        raise ValueError("Provide --stage2-dir or all of entity_state, rtmw_cache, handoff")
    return tuple(Path(p).expanduser().resolve() for p in (entity_state, rtmw_cache, handoff))  # type: ignore[return-value]


def validate_stage2_bundle(entity: Mapping[str, Any], cache: Mapping[str, Any], handoff: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    entity_schema = str(entity.get("schema_version", ""))
    if not entity_schema.startswith("entity-track-state-"):
        errors.append(f"Unsupported EntityTrackState schema: {entity_schema!r}")
    cache_schema = str(cache.get("schema_version", ""))
    if cache_schema != "stage2-raw-rtmw-track-cache-1.0":
        errors.append(f"Unsupported RTMW cache schema: {cache_schema!r}")
    coord_entity = str((entity.get("replay_context") or {}).get("coordinate_space", ""))
    coord_cache = str(cache.get("coordinate_space", ""))
    if coord_entity != RAW_PIXEL_SPACE or coord_cache != RAW_PIXEL_SPACE:
        errors.append(f"Stage 3 requires {RAW_PIXEL_SPACE}; entity={coord_entity!r}, cache={coord_cache!r}")
    semantics = str(cache.get("score_semantics", ""))
    if semantics != "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY":
        warnings.append(f"Unexpected RTMW score semantics: {semantics!r}")
    ctx = entity.get("replay_context") or {}
    for key in ("selected_frame", "window_start", "window_end", "image_width", "image_height"):
        if key not in ctx:
            errors.append(f"replay_context missing {key}")
    tracks = {str(t.get("track_id")): t for t in entity.get("tracks", [])}
    cache_tracks = {str(k): v for k, v in (cache.get("tracks") or {}).items()}
    candidate_ids = [str(x) for x in handoff.get("candidate_track_ids", [])]
    if not candidate_ids:
        warnings.append("No Stage-3 candidate tracks")
    missing_tracks = [tid for tid in candidate_ids if tid not in tracks]
    if missing_tracks:
        errors.append(f"Candidate IDs missing from EntityTrackState: {missing_tracks}")
    t0 = int(ctx.get("selected_frame", -1))
    missing_bbox_t0: List[str] = []
    missing_pose_t0: List[str] = []
    ambiguous_pose_keys: List[str] = []
    invalid_pose_records: List[str] = []
    candidate_flag_mismatch: List[str] = []
    duplicate_entity_frames: List[str] = []
    for tid in candidate_ids:
        tr = tracks.get(tid, {})
        if tr and not bool(tr.get("candidate_for_stage3", False)):
            candidate_flag_mismatch.append(tid)
        entity_frames: Dict[int, int] = {}
        for o in tr.get("observations", []):
            fi = int(o.get("frame_index", -999999))
            entity_frames[fi] = entity_frames.get(fi, 0) + 1
        for fi, count in entity_frames.items():
            if count > 1:
                duplicate_entity_frames.append(f"{tid}:frame_{fi}")
        obs = [o for o in tr.get("observations", []) if int(o.get("frame_index", -999999)) == t0]
        if len(obs) != 1 or not obs[0].get("bbox_xyxy"):
            missing_bbox_t0.append(tid)

        ctrack = cache_tracks.get(tid, {})
        cobs_all = list(ctrack.get("observations", []))
        cache_frames: Dict[int, int] = {}
        for co in cobs_all:
            fi = int(co.get("frame_index", -999999))
            cache_frames[fi] = cache_frames.get(fi, 0) + 1
            pose = co.get("pose") or {}
            kps = pose.get("keypoints") or []
            if len(kps) != 133:
                invalid_pose_records.append(f"{tid}:frame_{fi}:count={len(kps)}")
                continue
            for i, kp in enumerate(kps):
                if int(kp.get("index", i)) != i or str(kp.get("name", "")) != WHOLEBODY_KEYPOINT_NAMES[i]:
                    invalid_pose_records.append(f"{tid}:frame_{fi}:keypoint_{i}_schema")
                    break
        for fi, count in cache_frames.items():
            if count > 1:
                ambiguous_pose_keys.append(f"{tid}:frame_{fi}")
        cobs = [o for o in cobs_all if int(o.get("frame_index", -999999)) == t0]
        if len(cobs) == 0:
            missing_pose_t0.append(tid)
    if missing_bbox_t0:
        errors.append(f"Candidate tracks missing unique bbox at t0: {missing_bbox_t0}")
    if candidate_flag_mismatch:
        errors.append(f"Handoff candidates not marked candidate_for_stage3 in EntityTrackState: {candidate_flag_mismatch}")
    if duplicate_entity_frames:
        errors.append(f"Duplicate EntityTrackState observations for track/frame: {duplicate_entity_frames}")
    if ambiguous_pose_keys:
        errors.append(f"Ambiguous RTMW cache track/frame observations: {ambiguous_pose_keys}")
    if invalid_pose_records:
        errors.append(f"Invalid WholeBody133 cache records: {invalid_pose_records[:20]}")
    if missing_pose_t0:
        warnings.append(f"Candidate tracks missing RTMW pose at t0: {missing_pose_t0}")
    return {
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_tracks": len(candidate_ids),
        "missing_pose_at_selected_frame": missing_pose_t0,
        "ambiguous_track_frame_pose_mapping": ambiguous_pose_keys,
        "invalid_wholebody_records": invalid_pose_records,
        "score_semantics": semantics,
        "coordinate_space": coord_entity,
    }


def load_stage2_bundle(
    *,
    stage2_dir: str | Path | None = None,
    entity_state: str | Path | None = None,
    rtmw_cache: str | Path | None = None,
    handoff: str | Path | None = None,
    strict: bool = True,
) -> Stage2Bundle:
    ep, cp, hp = resolve_stage2_paths(
        stage2_dir=stage2_dir, entity_state=entity_state, rtmw_cache=rtmw_cache, handoff=handoff
    )
    entity, cache, ho = _read_json(ep), _read_json(cp), _read_json(hp)
    report = validate_stage2_bundle(entity, cache, ho)
    if strict and not report["ready"]:
        raise ValueError("Stage2->Stage3 preflight failed:\n- " + "\n- ".join(report["errors"]))
    return Stage2Bundle(ep, cp, hp, entity, cache, ho, report)
