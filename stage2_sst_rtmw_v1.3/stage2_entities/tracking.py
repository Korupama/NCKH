from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .consolidation import box_iou
from .contracts import EntityTrack, EntityTrackState, ReplayContext, TrackObservation

STAGE2_VERSION = "stage2-sst-rtmw-1.3.0"
CANDIDATE_ROLES = {"player", "goalkeeper"}


@dataclass
class _Observation:
    frame_index: int
    physical_human_id: str
    bbox_xyxy: np.ndarray
    detector_score: float
    role: str
    role_evidence: Dict[str, float]
    role_status: str
    source_detection_ids: List[str]
    pose: Optional[Dict[str, Any]]
    candidate_hint: bool = False
    superclass: str = "other"
    observation_tier: str = "primary"
    rescue_only: bool = False

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.asarray([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)

    @property
    def height(self) -> float:
        return max(1.0, float(self.bbox_xyxy[3] - self.bbox_xyxy[1]))


@dataclass
class _Anchor:
    track_id: str
    center: _Observation
    last: _Observation
    gap: int = 0
    observations: Dict[int, _Observation] = field(default_factory=dict)
    accepted_costs: Dict[int, float] = field(default_factory=dict)
    accepted_tiers: Dict[int, str] = field(default_factory=dict)

def _motion_prediction(anchor: _Anchor, target_frame: int) -> Optional[np.ndarray]:
    """Constant-velocity center prediction for short target-window gaps."""
    history = sorted(anchor.observations.values(), key=lambda o: o.frame_index)
    if len(history) < 2:
        return None
    o0, o1 = history[-2:]
    dt = max(1, int(o1.frame_index) - int(o0.frame_index))
    steps = max(1, int(target_frame) - int(o1.frame_index))
    velocity = (o1.center - o0.center) / float(dt)
    # Avoid unstable extrapolation after long detector gaps.
    velocity = np.clip(velocity, -0.25 * o1.height, 0.25 * o1.height)
    return o1.center + velocity * float(min(steps, 3))

def motion_cost(anchor: _Anchor, observation: _Observation, image_hw: Tuple[int, int]) -> Optional[float]:
    predicted = _motion_prediction(anchor, observation.frame_index)
    if predicted is None:
        return None
    h, w = image_hw
    scale = max(10.0, 0.5 * (h + w))
    return float(np.clip(np.linalg.norm(predicted - observation.center) / scale * 8.0, 0.0, 2.0))


def _pose_arrays(pose: Optional[Mapping[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    if not pose:
        return None, None, 0.0
    keypoints = pose.get("keypoints") or []
    if len(keypoints) < 17:
        return None, None, 0.0
    xy = np.asarray([
        [np.nan if k.get("x") is None else float(k["x"]), np.nan if k.get("y") is None else float(k["y"])]
        for k in keypoints[:17]
    ], dtype=np.float32)
    score = np.asarray([float(k.get("raw_score", 0.0)) for k in keypoints[:17]], dtype=np.float32)
    threshold = float(pose.get("keypoint_visibility_threshold", 1.0))
    return xy, score, threshold


def _pose_distance(a: _Observation, b: _Observation) -> Optional[float]:
    axy, ascore, ath = _pose_arrays(a.pose)
    bxy, bscore, bth = _pose_arrays(b.pose)
    if axy is None or bxy is None:
        return None
    valid = (
        np.isfinite(axy).all(axis=1) & np.isfinite(bxy).all(axis=1)
        & (ascore >= ath) & (bscore >= bth)
    )
    if int(valid.sum()) < 4:
        return None
    scale = max(10.0, 0.5 * (a.height + b.height))
    return float(np.clip(np.median(np.linalg.norm(axy[valid] - bxy[valid], axis=1)) / scale, 0.0, 2.0))


def _role_penalty(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a in CANDIDATE_ROLES and b in CANDIDATE_ROLES:
        # Player/GK disagreement is deliberately cheap: both are footballers.
        return 0.03
    if (a in CANDIDATE_ROLES) != (b in CANDIDATE_ROLES):
        return 0.12
    return 0.06


def association_cost(a: _Observation, b: _Observation, image_hw: Tuple[int, int]) -> float:
    h, w = image_hw
    diag = max(1.0, math.hypot(w, h))
    iou_cost = 1.0 - box_iou(a.bbox_xyxy, b.bbox_xyxy)
    center_cost = min(float(np.linalg.norm(a.center - b.center) / diag) * 8.0, 2.0)
    pose_cost = _pose_distance(a, b)
    if pose_cost is None:
        geom = 0.625 * iou_cost + 0.375 * center_cost
    else:
        geom = 0.50 * iou_cost + 0.30 * center_cost + 0.20 * pose_cost
    return float(geom + _role_penalty(a.role, b.role))


def _load_perception(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _observation_from_human(
    payload: Mapping[str, Any],
    human: Mapping[str, Any],
    *,
    pose_cache: Mapping[str, Any],
    tier: str,
) -> _Observation:
    hid = str(human["physical_human_id"])
    role = str(human.get("resolved_role", "other"))
    superclass = str(human.get("superclass", "footballer" if role in CANDIDATE_ROLES else role))
    return _Observation(
        frame_index=int(payload["frame_index"]),
        physical_human_id=hid,
        bbox_xyxy=np.asarray(human["bbox_xyxy"], dtype=np.float32),
        detector_score=float(human.get("detector_score", 0.0)),
        role=role,
        role_evidence={k: float(v) for k, v in human.get("role_evidence", {}).items()},
        role_status=str(human.get("role_status", "VALID")),
        source_detection_ids=[str(x) for x in human.get("source_detection_ids", [])],
        pose=pose_cache.get(hid) if tier == "primary" else None,
        candidate_hint=bool(human.get("candidate_hint", role in CANDIDATE_ROLES)),
        superclass=superclass,
        observation_tier=tier,
        rescue_only=bool(human.get("rescue_only", tier == "rescue")),
    )


def _observations_from_frame(payload: Mapping[str, Any]) -> Tuple[List[_Observation], List[_Observation]]:
    pose_cache = payload.get("pose_cache") or {}
    primary = [
        _observation_from_human(payload, h, pose_cache=pose_cache, tier="primary")
        for h in payload.get("humans", [])
    ]
    rescue = [
        _observation_from_human(payload, h, pose_cache={}, tier="rescue")
        for h in payload.get("rescue_humans", [])
        if bool(h.get("candidate_hint", str(h.get("resolved_role", "other")) in CANDIDATE_ROLES))
    ]
    return primary, rescue


def _max_consecutive_gap(observed_frames: Sequence[int], all_frames: Sequence[int]) -> int:
    observed = set(observed_frames)
    best = current = 0
    for fi in all_frames:
        if fi in observed:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def _aggregate_role(observations: Sequence[_Observation], min_margin: float = 0.08):
    totals = {"player": 0.0, "goalkeeper": 0.0, "referee": 0.0, "other": 0.0}
    for obs in observations:
        # Rescue-only observations are low-confidence evidence and therefore down-weighted.
        tier_weight = 0.35 if obs.rescue_only else 1.0
        for role in totals:
            totals[role] += tier_weight * max(0.0, float(obs.role_evidence.get(role, 0.0)))
    grand = sum(totals.values())
    if grand <= 1e-12:
        return "other", 0.0, 0.0, "DEGRADED", totals
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    role, top = ranked[0]
    second = ranked[1][1]
    score = top / grand
    margin = (top - second) / grand
    status = "VALID" if margin >= min_margin else "DEGRADED"
    return role, float(score), float(margin), status, {k: float(v/grand) for k,v in totals.items()}


class DecisionFrameAnchoredTracker:
    """Offline bidirectional tracker anchored strictly to primary detections at t0.

    M1 uses a two-threshold association policy. Primary observations are matched first.
    Only still-unmatched *footballer* anchors may consume low-score rescue observations,
    and rescue observations can never create new anchors/tracks.
    """

    def __init__(
        self,
        *,
        max_assignment_cost: float = 0.92,
        max_gap: int = 6,
        use_temporal_rescue: bool = True,
        rescue_max_assignment_cost: float = 0.78,
        use_motion_prior: bool = True,
        motion_weight: float = 0.15,
    ) -> None:
        self.max_assignment_cost = float(max_assignment_cost)
        self.max_gap = int(max_gap)
        self.use_temporal_rescue = bool(use_temporal_rescue)
        self.rescue_max_assignment_cost = float(rescue_max_assignment_cost)
        self.use_motion_prior = bool(use_motion_prior)
        self.motion_weight = float(np.clip(motion_weight, 0.0, 0.5))

    @staticmethod
    def _assign(
        anchors: List[_Anchor],
        anchor_indices: Sequence[int],
        observations: Sequence[_Observation],
        image_hw: Tuple[int, int],
        *,
        max_cost: float,
        pos: int,
        tier: str,
        use_motion_prior: bool = True,
        motion_weight: float = 0.15,
    ) -> Tuple[set[int], set[int]]:
        assigned_obs: set[int] = set()
        assigned_anchor: set[int] = set()
        if not anchor_indices or not observations:
            return assigned_anchor, assigned_obs
        matrix = np.full((len(anchor_indices), len(observations)), 10.0, dtype=np.float64)
        for r, ai in enumerate(anchor_indices):
            for c, obs in enumerate(observations):
                base = association_cost(anchors[ai].last, obs, image_hw)
                prior = motion_cost(anchors[ai], obs, image_hw) if use_motion_prior else None
                matrix[r, c] = ((1.0 - motion_weight) * base + motion_weight * prior
                                if prior is not None else base)
        rows, cols = linear_sum_assignment(matrix)
        for r, c in zip(rows.tolist(), cols.tolist()):
            value = float(matrix[r, c])
            if value > max_cost:
                continue
            ai = int(anchor_indices[r])
            anchors[ai].observations[pos] = observations[c]
            anchors[ai].accepted_costs[pos] = value
            anchors[ai].accepted_tiers[pos] = tier
            anchors[ai].last = observations[c]
            anchors[ai].gap = 0
            assigned_anchor.add(ai)
            assigned_obs.add(c)
        return assigned_anchor, assigned_obs

    def track(self, frames: Sequence[Mapping[str, Any]], context: ReplayContext) -> Dict[str, Any]:
        if not frames:
            return {"anchors": [], "orphans": {}, "rescue_orphans": {}}
        frame_indices = [int(x["frame_index"]) for x in frames]
        if frame_indices != sorted(frame_indices):
            raise ValueError("Perception frames must be sorted by global frame_index")
        try:
            center_pos = frame_indices.index(context.selected_frame)
        except ValueError as exc:
            raise ValueError("Selected frame is missing from perception manifest") from exc

        paired = [_observations_from_frame(x) for x in frames]
        primary_obs = [p for p, _ in paired]
        rescue_obs = [r for _, r in paired]
        image_sizes = [(int(x["image_height"]), int(x["image_width"])) for x in frames]

        # Critical invariant: t0 anchors are PRIMARY only. Rescue boxes never create entities.
        center = sorted(primary_obs[center_pos], key=lambda x: (float(x.center[0]), float(x.center[1])))
        anchors = [
            _Anchor(
                track_id=f"track_{i:03d}",
                center=o,
                last=o,
                observations={center_pos: o},
                accepted_tiers={center_pos: "primary"},
            )
            for i, o in enumerate(center, start=1)
        ]
        orphans: Dict[int, List[Dict[str, Any]]] = {context.selected_frame: []}
        rescue_orphans: Dict[int, List[Dict[str, Any]]] = {context.selected_frame: []}

        def propagate(positions: Sequence[int]) -> None:
            for pos in positions:
                primary = primary_obs[pos]
                rescue = rescue_obs[pos]
                available = [i for i, a in enumerate(anchors) if a.gap <= self.max_gap]

                assigned_primary_anchor, assigned_primary_obs = self._assign(
                    anchors, available, primary, image_sizes[pos],
                    max_cost=self.max_assignment_cost, pos=pos, tier="primary",
                    use_motion_prior=self.use_motion_prior, motion_weight=self.motion_weight,
                )
                assigned_anchor = set(assigned_primary_anchor)
                assigned_rescue_obs: set[int] = set()

                if self.use_temporal_rescue and rescue:
                    rescue_eligible = [
                        ai for ai in available
                        if ai not in assigned_anchor and anchors[ai].center.candidate_hint
                    ]
                    rescued_anchor, assigned_rescue_obs = self._assign(
                        anchors, rescue_eligible, rescue, image_sizes[pos],
                        max_cost=self.rescue_max_assignment_cost, pos=pos, tier="rescue",
                        use_motion_prior=self.use_motion_prior, motion_weight=self.motion_weight,
                    )
                    assigned_anchor.update(rescued_anchor)

                for ai in available:
                    if ai not in assigned_anchor:
                        anchors[ai].gap += 1

                orphans[frame_indices[pos]] = [
                    {
                        "physical_human_id": primary[c].physical_human_id,
                        "role": primary[c].role,
                        "bbox_xyxy": primary[c].bbox_xyxy.round(3).tolist(),
                        "detector_score": round(primary[c].detector_score, 6),
                    }
                    for c in range(len(primary)) if c not in assigned_primary_obs
                ]
                rescue_orphans[frame_indices[pos]] = [
                    {
                        "physical_human_id": rescue[c].physical_human_id,
                        "role": rescue[c].role,
                        "bbox_xyxy": rescue[c].bbox_xyxy.round(3).tolist(),
                        "detector_score": round(rescue[c].detector_score, 6),
                    }
                    for c in range(len(rescue)) if c not in assigned_rescue_obs
                ]

        propagate(list(range(center_pos - 1, -1, -1)))
        for anchor in anchors:
            anchor.last = anchor.center
            anchor.gap = 0
        propagate(list(range(center_pos + 1, len(frames))))
        return {
            "anchors": anchors,
            "orphans": orphans,
            "rescue_orphans": rescue_orphans,
            "center_pos": center_pos,
            "frame_indices": frame_indices,
        }


def build_entity_track_state(
    perception_manifest: Mapping[str, Any] | str | Path,
    context: ReplayContext,
    output_dir: str | Path,
    *,
    max_assignment_cost: float = 0.92,
    max_gap: int = 6,
    use_temporal_rescue: bool = True,
    rescue_max_assignment_cost: float = 0.78,
    use_motion_prior: bool = True,
    motion_weight: float = 0.15,
) -> EntityTrackState:
    if not isinstance(perception_manifest, Mapping):
        perception_manifest = json.loads(Path(perception_manifest).read_text(encoding="utf-8"))
    frame_records = sorted(perception_manifest.get("frames", []), key=lambda x: int(x["frame_index"]))
    frames = [_load_perception(x["perception_json"]) for x in frame_records]
    tracker = DecisionFrameAnchoredTracker(
        max_assignment_cost=max_assignment_cost,
        max_gap=max_gap,
        use_temporal_rescue=use_temporal_rescue,
        rescue_max_assignment_cost=rescue_max_assignment_cost,
        use_motion_prior=use_motion_prior,
        motion_weight=motion_weight,
    )
    tracked = tracker.track(frames, context)
    all_frame_indices = tracked.get("frame_indices", [])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_pose_handoff: Dict[str, Any] = {
        "schema_version": "stage2-raw-rtmw-track-cache-1.0",
        "score_semantics": "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
        "coordinate_space": context.coordinate_space,
        "tracks": {},
    }

    entity_tracks: List[EntityTrack] = []
    selected_entities: List[Dict[str, Any]] = []
    for anchor in tracked.get("anchors", []):
        ordered_pairs = sorted(anchor.observations.items())
        obs_values = [o for _, o in ordered_pairs]
        role, role_score, role_margin, role_status, role_distribution = _aggregate_role(obs_values)
        observed_frames = [o.frame_index for o in obs_values]
        ratio = len(observed_frames) / max(1, len(all_frame_indices))
        mean_cost = float(np.mean(list(anchor.accepted_costs.values()))) if anchor.accepted_costs else 0.0
        identity_conf = float(np.clip(np.exp(-1.6 * mean_cost) * np.sqrt(max(ratio, 0.0)), 0.0, 1.0))
        max_consecutive_gap = _max_consecutive_gap(observed_frames, all_frame_indices)
        candidate = role in CANDIDATE_ROLES
        status = "VALID" if identity_conf >= 0.55 and role_status == "VALID" else "DEGRADED"
        rescue_observation_count = sum(o.rescue_only for o in obs_values)

        track_observations: List[TrackObservation] = []
        pose_entries = []
        for pos, obs in ordered_pairs:
            pose_key = None
            pose_quality_score = pose_quality_level = None
            if obs.pose is not None:
                pose_key = f"{anchor.track_id}:frame_{obs.frame_index}"
                q = obs.pose.get("quality") or {}
                pose_quality_score = None if q.get("score") is None else float(q["score"])
                pose_quality_level = q.get("level")
                pose_entries.append({
                    "frame_index": obs.frame_index,
                    "physical_human_id": obs.physical_human_id,
                    "pose_cache_key": pose_key,
                    "bbox_xyxy": obs.bbox_xyxy.tolist(),
                    "pose": obs.pose,
                })
            track_observations.append(TrackObservation(
                frame_index=obs.frame_index,
                bbox_xyxy=[float(x) for x in obs.bbox_xyxy],
                detector_score=obs.detector_score,
                frame_role=obs.role,
                role_evidence=dict(obs.role_evidence),
                physical_human_id=obs.physical_human_id,
                source_detection_ids=list(obs.source_detection_ids),
                pose_cache_key=pose_key,
                pose_quality_score=pose_quality_score,
                pose_quality_level=pose_quality_level,
                association_cost=anchor.accepted_costs.get(pos),
                observation_tier=anchor.accepted_tiers.get(pos, obs.observation_tier),
                rescue_only=obs.rescue_only,
            ))
        raw_pose_handoff["tracks"][anchor.track_id] = {
            "candidate_for_stage3": candidate,
            "final_stage2_role": role,
            "observations": pose_entries,
        }

        entity = EntityTrack(
            track_id=anchor.track_id,
            role=role,
            role_score=role_score,
            role_margin=role_margin,
            role_status=role_status,
            candidate_for_stage3=candidate,
            status=status,
            start_frame=min(observed_frames), end_frame=max(observed_frames),
            observed_frames=len(observed_frames), total_window_frames=len(all_frame_indices),
            observed_ratio=ratio, max_consecutive_gap=max_consecutive_gap,
            identity_confidence=identity_conf,
            observations=track_observations,
            diagnostics={
                "role_distribution": role_distribution,
                "mean_association_cost": mean_cost,
                "center_physical_human_id": anchor.center.physical_human_id,
                "center_source_detection_ids": anchor.center.source_detection_ids,
                "rescue_observation_count": int(rescue_observation_count),
                "rescue_observation_ratio": float(rescue_observation_count / max(1, len(obs_values))),
            },
        )
        entity_tracks.append(entity)

        center_obs = anchor.observations.get(tracked["center_pos"])
        if center_obs is not None:
            selected_entities.append({
                "track_id": anchor.track_id,
                "role": role,
                "role_score": role_score,
                "role_status": role_status,
                "candidate_for_stage3": candidate,
                "bbox_xyxy": center_obs.bbox_xyxy.tolist(),
                "detector_score": center_obs.detector_score,
                "identity_confidence": identity_conf,
                "status": status,
                "coordinate_space": context.coordinate_space,
                "camera_ref": {"timeline_frame": context.selected_frame},
            })

    pose_path = out / "stage2_rtmw_track_cache.json"
    pose_path.write_text(json.dumps(raw_pose_handoff, indent=2, ensure_ascii=False), encoding="utf-8")

    excluded_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    balls_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    duplicate_groups = 0
    raw_humans = 0
    low_score_raw_humans = 0
    consolidated_humans = 0
    rescue_human_hypotheses = 0
    for frame in frames:
        fi = str(int(frame["frame_index"]))
        excluded_by_frame[fi] = [
            h for h in frame.get("humans", []) if h.get("resolved_role") not in CANDIDATE_ROLES
        ]
        balls_by_frame[fi] = list(frame.get("ball_detections", []))
        raw_humans += sum(int(d.get("label_id", -1)) in {2,3,4,5,6} for d in frame.get("raw_detections", []))
        low_score_raw_humans += sum(int(d.get("label_id", -1)) in {2,3,4,5,6} for d in frame.get("raw_low_score_detections", frame.get("raw_detections", [])))
        consolidated_humans += len(frame.get("humans", []))
        rescue_human_hypotheses += len(frame.get("rescue_humans", []))
        duplicate_groups += sum(len(h.get("source_detection_ids", [])) > 1 for h in frame.get("humans", []))

    candidates = [t for t in entity_tracks if t.candidate_for_stage3]
    degraded = [t for t in candidates if t.status != "VALID"]
    candidate_pose_missing_at_t0 = []
    for track in candidates:
        cached = raw_pose_handoff["tracks"].get(track.track_id, {}).get("observations", [])
        if not any(int(item.get("frame_index", -1)) == context.selected_frame for item in cached):
            candidate_pose_missing_at_t0.append(track.track_id)
    status = "INVALID" if not candidates else ("DEGRADED" if degraded else "VALID")
    handoff = {
        "stage3_input_ready": bool(candidates) and not candidate_pose_missing_at_t0,
        "candidate_track_ids": [t.track_id for t in candidates],
        "candidate_tracks_missing_rtmw_at_selected_frame": candidate_pose_missing_at_t0,
        "entity_track_state_schema": "entity-track-state-1.0",
        "raw_rtmw_track_cache": str(pose_path.resolve()),
        "note": "RTMW cache is raw shared perception evidence, not Stage-3 quality-accepted pose output.",
    }
    return EntityTrackState(
        schema_version="entity-track-state-1.0",
        stage2_version=STAGE2_VERSION,
        replay_context=context.to_dict(),
        status=status,
        tracks=entity_tracks,
        selected_frame_entities=selected_entities,
        excluded_detections_by_frame=excluded_by_frame,
        auxiliary_ball_detections_by_frame=balls_by_frame,
        stage3_handoff=handoff,
        backend={
            "name": "SST M1 + geometry-aware consolidation + RTMW cue + two-threshold anchored tracker",
            "perception_manifest_schema": perception_manifest.get("schema_version"),
            "tracking": {
                "max_assignment_cost": float(max_assignment_cost),
                "max_gap": int(max_gap),
                "use_temporal_rescue": bool(use_temporal_rescue),
                "rescue_max_assignment_cost": float(rescue_max_assignment_cost),
            },
        },
        diagnostics={
            "num_tracks": len(entity_tracks),
            "num_candidate_tracks": len(candidates),
            "num_excluded_tracks": len(entity_tracks)-len(candidates),
            "num_degraded_candidate_tracks": len(degraded),
            "candidate_tracks_missing_rtmw_at_selected_frame": candidate_pose_missing_at_t0,
            "raw_human_detections": raw_humans,
            "raw_low_score_human_detections": low_score_raw_humans,
            "consolidated_human_hypotheses": consolidated_humans,
            "rescue_human_hypotheses": rescue_human_hypotheses,
            "rescue_observations_used": int(sum(t.diagnostics.get("rescue_observation_count", 0) for t in entity_tracks)),
            "cross_class_duplicate_groups_collapsed": duplicate_groups,
            "orphan_observations_by_frame": tracked.get("orphans", {}),
            "unused_rescue_observations_by_frame": tracked.get("rescue_orphans", {}),
        },
        artifacts={"rtmw_track_cache": str(pose_path.resolve())},
    )
