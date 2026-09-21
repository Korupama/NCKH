from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
import json
import time
import hashlib
import os

import cv2
import numpy as np
from PIL import Image

from .consolidation import (
    BALL_LABEL_ID,
    PhysicalHumanHypothesis,
    box_iou,
    consolidate_human_detections,
)
from .backend_core.sst_single_frame_inference import (
    CLASS_NAMES,
    build_sst_model,
    predict_frame,
    resolve_device,
)
from .backend_core.sst_rtmw_pose_pipeline import (
    RTMWOpenCVDNNPose,
    WHOLEBODY_KEYPOINT_NAMES,
    expand_box,
    pose_quality,
    postprocess_sst_prediction,
)

STAGE2_VERSION = "stage2-sst-rtmw-1.3.0"


def _atomic_write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, p)
    return p


def _payload_fingerprints(payload: Mapping[str, Any]) -> Dict[str, str]:
    prov = payload.get("provenance") or {}
    return {
        "sst": str(prov.get("sst_sha256", "")),
        "rtmw": str(prov.get("rtmw_sha256", "")),
    }


def _sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _stable_payload_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class RawPoseCache:
    physical_human_id: str
    source_detection_ids: List[str]
    bbox_xyxy: List[float]
    expanded_pose_bbox_xyxy: List[float]
    backend: str
    is_learned_model: bool
    score_semantics: str
    keypoint_visibility_threshold: float
    quality: Dict[str, Any]
    keypoints: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FramePerception:
    frame_index: int
    image_path: str
    image_width: int
    image_height: int
    # High-threshold detections used as normal Stage-2 observations.
    raw_detections: List[Dict[str, Any]]
    # Class-local NMS output retained down to raw_human_score_floor. This enables
    # threshold sweeps and rescue-only tracking without rerunning SST.
    raw_low_score_detections: List[Dict[str, Any]]
    humans: List[Dict[str, Any]]
    # Low-score footballer hypotheses never create t0 anchors; they can only rescue
    # already anchored tracks in adjacent frames.
    rescue_humans: List[Dict[str, Any]]
    pose_cache: Dict[str, Dict[str, Any]]
    ball_detections: List[Dict[str, Any]]
    timing_seconds: Dict[str, float]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _raw_detection_record(detection) -> Dict[str, Any]:
    return {
        "detection_id": str(detection.detection_id),
        "label_id": int(detection.label_id),
        "label": str(detection.label),
        "score": float(detection.score),
        "bbox_xyxy": [float(x) for x in detection.bbox_xyxy],
    }


def _active_records(
    records: List[Dict[str, Any]], class_thresholds: Mapping[int, float]
) -> List[Dict[str, Any]]:
    return [
        dict(r)
        for r in records
        if float(r.get("score", 0.0)) >= float(class_thresholds.get(int(r.get("label_id", -1)), 1.0))
    ]


def _pose_to_cache(h: PhysicalHumanHypothesis, expanded, pose, quality, threshold: float) -> RawPoseCache:
    keypoints = []
    xy = np.asarray(pose.keypoints_xy, dtype=float)
    scores = np.asarray(pose.scores, dtype=float)
    for idx, name in enumerate(WHOLEBODY_KEYPOINT_NAMES):
        raw = float(scores[idx]) if idx < len(scores) else 0.0
        point = xy[idx] if idx < len(xy) else [np.nan, np.nan]
        keypoints.append({
            "index": idx,
            "name": name,
            "x": None if not np.isfinite(point[0]) else float(point[0]),
            "y": None if not np.isfinite(point[1]) else float(point[1]),
            "raw_score": raw,
            "visible": bool(np.isfinite(point).all() and raw >= threshold),
        })
    return RawPoseCache(
        physical_human_id=h.physical_human_id,
        source_detection_ids=list(h.source_detection_ids),
        bbox_xyxy=list(h.bbox_xyxy),
        expanded_pose_bbox_xyxy=[float(x) for x in expanded],
        backend=str(pose.backend),
        is_learned_model=True,
        score_semantics="RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
        keypoint_visibility_threshold=float(threshold),
        quality=asdict(quality),
        keypoints=keypoints,
    )


class SSTRTMWStage2Backend:
    """Stage-2 M1 perception backend.

    SST is executed once per frame at a low human-score floor. The low-score output is
    persisted, while the normal high-threshold detections are derived from that cache.
    This makes threshold analysis possible without another SST pass and supplies
    rescue-only observations for decision-frame-anchored tracking.

    Cross-class physical-human consolidation is still performed BEFORE RTMW. RTMW is
    run only for normal/ambiguous active hypotheses; rescue-only boxes intentionally do
    not consume pose inference.
    """

    def __init__(
        self,
        *,
        sst_checkpoint: str | Path,
        rtmw_model: str | Path,
        sst_device: str = "auto",
        rtmw_device: str = "cpu",
        rtmw_input_width: int = 288,
        rtmw_input_height: int = 384,
        rtmw_bbox_padding: float = 1.25,
        player_threshold: float = 0.50,
        goalkeeper_threshold: float = 0.50,
        ball_threshold: float = 0.35,
        referee_threshold: float = 0.55,
        staff_threshold: float = 0.60,
        raw_human_score_floor: float = 0.20,
        person_nms_iou: float = 0.65,
        ball_nms_iou: float = 0.30,
        ball_top_k: int = 1,
        # Backwards-compatible direct-IoU override. If supplied, it replaces
        # consolidation_high_iou only; the geometry-consistent path remains active.
        cross_class_iou_threshold: float | None = None,
        consolidation_high_iou: float = 0.82,
        consolidation_low_iou: float = 0.60,
        consolidation_max_center_distance: float = 0.18,
        consolidation_min_area_similarity: float = 0.65,
        consolidation_min_intersection_over_min: float = 0.75,
        rescue_overlap_with_active_iou: float = 0.65,
        role_ambiguous_margin: float = 0.05,
        person_box_expansion: float = 1.0,
        keypoint_threshold: float = 1.0,
    ) -> None:
        self.sst_device = resolve_device(sst_device)
        started = time.perf_counter()
        self.sst_model, self.sst_metadata = build_sst_model(sst_checkpoint, self.sst_device)
        self.sst_load_seconds = time.perf_counter() - started
        self.pose_estimator = RTMWOpenCVDNNPose(
            rtmw_model,
            input_width=rtmw_input_width,
            input_height=rtmw_input_height,
            bbox_padding=rtmw_bbox_padding,
            device=rtmw_device,
        )
        self.class_thresholds = {
            1: float(ball_threshold), 2: float(player_threshold), 3: float(goalkeeper_threshold),
            4: float(referee_threshold), 5: float(referee_threshold), 6: float(staff_threshold),
        }
        floor = float(raw_human_score_floor)
        if not 0.0 <= floor <= 1.0:
            raise ValueError("raw_human_score_floor must be in [0,1]")
        self.raw_human_score_floor = floor
        self.raw_cache_thresholds = {
            1: float(ball_threshold),
            2: min(float(player_threshold), floor),
            3: min(float(goalkeeper_threshold), floor),
            4: min(float(referee_threshold), floor),
            5: min(float(referee_threshold), floor),
            6: min(float(staff_threshold), floor),
        }
        self.person_nms_iou = float(person_nms_iou)
        self.ball_nms_iou = float(ball_nms_iou)
        self.ball_top_k = int(ball_top_k)
        self.consolidation_high_iou = float(
            consolidation_high_iou if cross_class_iou_threshold is None else cross_class_iou_threshold
        )
        # legacy attribute retained for callers/tests and manifests
        self.cross_class_iou_threshold = self.consolidation_high_iou
        self.consolidation_low_iou = float(consolidation_low_iou)
        self.consolidation_max_center_distance = float(consolidation_max_center_distance)
        self.consolidation_min_area_similarity = float(consolidation_min_area_similarity)
        self.consolidation_min_intersection_over_min = float(consolidation_min_intersection_over_min)
        self.rescue_overlap_with_active_iou = float(rescue_overlap_with_active_iou)
        self.role_ambiguous_margin = float(role_ambiguous_margin)
        self.person_box_expansion = float(person_box_expansion)
        self.keypoint_threshold = float(keypoint_threshold)
        self.sst_checkpoint = str(Path(sst_checkpoint).expanduser().resolve())
        self.rtmw_model = str(Path(rtmw_model).expanduser().resolve())
        self.sst_checkpoint_sha256 = _sha256_file(self.sst_checkpoint)
        self.rtmw_model_sha256 = _sha256_file(self.rtmw_model)

    def _consolidate(self, records: List[Dict[str, Any]]) -> List[PhysicalHumanHypothesis]:
        return consolidate_human_detections(
            records,
            high_iou_threshold=float(getattr(self, "consolidation_high_iou", getattr(self, "cross_class_iou_threshold", 0.82))),
            low_iou_threshold=float(getattr(self, "consolidation_low_iou", 0.60)),
            max_center_distance=float(getattr(self, "consolidation_max_center_distance", 0.18)),
            min_area_similarity=float(getattr(self, "consolidation_min_area_similarity", 0.65)),
            min_intersection_over_min=float(getattr(self, "consolidation_min_intersection_over_min", 0.75)),
            ambiguous_margin=float(getattr(self, "role_ambiguous_margin", 0.05)),
        )

    def configuration(self) -> Dict[str, Any]:
        return {
            "class_thresholds": {CLASS_NAMES.get(k, str(k)): v for k, v in self.class_thresholds.items()},
            "person_nms_iou": self.person_nms_iou,
            "ball_nms_iou": self.ball_nms_iou,
            "consolidation_high_iou": float(getattr(self, "consolidation_high_iou", self.cross_class_iou_threshold)),
            "consolidation_low_iou": float(getattr(self, "consolidation_low_iou", 0.60)),
            "consolidation_max_center_distance": float(getattr(self, "consolidation_max_center_distance", 0.18)),
            "consolidation_min_area_similarity": float(getattr(self, "consolidation_min_area_similarity", 0.65)),
            "consolidation_min_intersection_over_min": float(getattr(self, "consolidation_min_intersection_over_min", 0.75)),
            "raw_human_score_floor": float(getattr(self, "raw_human_score_floor", 0.20)),
            "rescue_overlap_with_active_iou": float(getattr(self, "rescue_overlap_with_active_iou", 0.65)),
            "role_ambiguous_margin": self.role_ambiguous_margin,
            "keypoint_threshold": self.keypoint_threshold,
        }

    def model_info(self) -> Dict[str, Any]:
        return {
            "sst": {
                **self.sst_metadata,
                "checkpoint_path": self.sst_checkpoint,
                "sha256": self.sst_checkpoint_sha256,
                "device": str(self.sst_device),
                "load_seconds": round(self.sst_load_seconds, 4),
            },
            "rtmw": {
                "model_path": self.rtmw_model,
                "sha256": self.rtmw_model_sha256,
                "backend": self.pose_estimator.name,
                "input_width_height": [self.pose_estimator.input_width, self.pose_estimator.input_height],
                "bbox_padding": self.pose_estimator.bbox_padding,
                "device": self.pose_estimator.device,
                "score_semantics": "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
            },
        }

    def process_frame(self, image_path: str | Path, frame_index: int) -> FramePerception:
        path = Path(image_path).expanduser().resolve()
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(path)
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)

        t0 = time.perf_counter()
        prediction = predict_frame(self.sst_model, pil, self.sst_device)
        raw_cache_thresholds = getattr(self, "raw_cache_thresholds", self.class_thresholds)
        low_detections = postprocess_sst_prediction(
            prediction, width, height,
            class_thresholds=raw_cache_thresholds,
            person_nms_iou=self.person_nms_iou,
            ball_nms_iou=self.ball_nms_iou,
            ball_top_k=self.ball_top_k,
        )
        detection_seconds = time.perf_counter() - t0
        low_records = [_raw_detection_record(x) for x in low_detections]
        raw_records = _active_records(low_records, self.class_thresholds)

        humans = self._consolidate(raw_records)
        low_humans = self._consolidate(low_records)
        active_detection_ids = {str(d["detection_id"]) for d in raw_records}
        active_boxes = [h.bbox_xyxy for h in humans]
        rescue_humans: List[PhysicalHumanHypothesis] = []
        overlap_limit = float(getattr(self, "rescue_overlap_with_active_iou", 0.65))
        for h in low_humans:
            if not h.candidate_hint:
                continue
            # If any source member already passes the normal threshold, this is not a
            # rescue-only hypothesis. Also suppress geometric duplicates of active humans.
            if any(sid in active_detection_ids for sid in h.source_detection_ids):
                continue
            if any(box_iou(h.bbox_xyxy, box) >= overlap_limit for box in active_boxes):
                continue
            rescue_humans.append(h)

        balls = [r for r in raw_records if int(r["label_id"]) == BALL_LABEL_ID]

        pose_humans = [h for h in humans if h.pose_required]
        expanded = [
            expand_box(h.bbox_xyxy, width, height, factor=self.person_box_expansion)
            for h in pose_humans
        ]
        p0 = time.perf_counter()
        poses = self.pose_estimator.infer(image_bgr, expanded) if expanded else []
        pose_seconds = time.perf_counter() - p0
        if len(poses) != len(pose_humans):
            raise RuntimeError(f"RTMW returned {len(poses)} poses for {len(pose_humans)} boxes")

        pose_cache: Dict[str, Dict[str, Any]] = {}
        for human, box, pose in zip(pose_humans, expanded, poses):
            quality = pose_quality(
                pose,
                detection_score=human.detector_score,
                bbox_xyxy=human.bbox_xyxy,
                keypoint_threshold=self.keypoint_threshold,
            )
            pose_cache[human.physical_human_id] = _pose_to_cache(
                human, box, pose, quality, self.keypoint_threshold
            ).to_dict()

        primary_payload = []
        for h in humans:
            item = h.to_dict()
            item["observation_tier"] = "primary"
            primary_payload.append(item)
        rescue_payload = []
        for h in rescue_humans:
            item = h.to_dict()
            item["observation_tier"] = "rescue"
            item["rescue_only"] = True
            rescue_payload.append(item)

        return FramePerception(
            frame_index=int(frame_index),
            image_path=str(path), image_width=width, image_height=height,
            raw_detections=raw_records,
            raw_low_score_detections=low_records,
            humans=primary_payload,
            rescue_humans=rescue_payload,
            pose_cache=pose_cache,
            ball_detections=balls,
            timing_seconds={
                "sst_detection": round(detection_seconds, 4),
                "rtmw_pose": round(pose_seconds, 4),
                "total_after_models_loaded": round(time.perf_counter() - t0, 4),
            },
            provenance={
                "backend": "stage2-sst-rtmw-m1",
                "stage2_version": STAGE2_VERSION,
                "raw_low_score_cache": True,
                "raw_human_score_floor": float(getattr(self, "raw_human_score_floor", 0.20)),
                "configuration_sha256": _stable_payload_hash(self.configuration()),
                "two_threshold_temporal_rescue_ready": True,
                "cross_class_consolidation_before_pose": True,
                "hierarchical_superclass_semantics": True,
                "coordinate_space": "RAW_DISTORTED_PIXEL",
                "sst_sha256": str(getattr(self, "sst_checkpoint_sha256", "")),
                "rtmw_sha256": str(getattr(self, "rtmw_model_sha256", "")),
            },
        )
    def process_window(
        self,
        frame_map: Mapping[str, Any],
        output_dir: str | Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Process a window with crash-safe frame checkpoints and resumable progress.

        Each frame JSON is atomically committed. ``perception_checkpoint.json`` is written
        before the first inference and after every completed frame, so an interrupted run can
        safely reuse completed frames only when model fingerprints still match.
        """
        out = Path(output_dir)
        frames_dir = out / "frame_perception"
        frames_dir.mkdir(parents=True, exist_ok=True)
        partial_path = out / "perception_checkpoint.json"
        manifest_path = out / "perception_manifest.json"
        model_info = self.model_info()
        expected_hashes = {
            "sst": self.sst_checkpoint_sha256,
            "rtmw": self.rtmw_model_sha256,
        }
        configuration = self.configuration()

        verified_partial = False
        completed_from_partial = set()
        if partial_path.is_file() and not overwrite:
            try:
                old = json.loads(partial_path.read_text(encoding="utf-8"))
                old_models = old.get("models") or {}
                old_hashes = {
                    "sst": str((old_models.get("sst") or {}).get("sha256", "")),
                    "rtmw": str((old_models.get("rtmw") or {}).get("sha256", "")),
                }
                if old_hashes != expected_hashes:
                    raise RuntimeError(
                        "Interrupted perception checkpoint uses different model weights. "
                        "Use --overwrite-perception or a new output directory."
                    )
                if dict(old.get("configuration") or {}) != configuration:
                    raise RuntimeError(
                        "Interrupted perception checkpoint uses a different perception configuration. "
                        "Use --overwrite-perception or a new output directory."
                    )
                verified_partial = True
                completed_from_partial = {int(x) for x in old.get("completed_frames", [])}
            except RuntimeError:
                raise
            except Exception:
                verified_partial = False
                completed_from_partial = set()

        if overwrite:
            completed_from_partial = set()

        checkpoint = {
            "schema_version": "stage2-perception-checkpoint-1.0",
            "stage2_version": STAGE2_VERSION,
            "status": "RUNNING",
            "models": model_info,
            "configuration": configuration,
            "total_frames": len(frame_map.get("frames", [])),
            "completed_frames": sorted(completed_from_partial),
            "updated_unix": time.time(),
        }
        _atomic_write_json(partial_path, checkpoint)

        records = []
        started = time.perf_counter()
        total = len(frame_map["frames"])
        for idx, item in enumerate(frame_map["frames"], start=1):
            fi = int(item["global_frame_index"])
            target = frames_dir / f"frame_{fi:09d}_perception.json"
            reused = False
            payload = None
            if target.is_file() and not overwrite:
                try:
                    candidate = json.loads(target.read_text(encoding="utf-8"))
                    hashes = _payload_fingerprints(candidate)
                    prov = candidate.get("provenance") or {}
                    config_hash = str(prov.get("configuration_sha256", ""))
                    expected_config_hash = _stable_payload_hash(configuration)
                    if (hashes == expected_hashes and config_hash == expected_config_hash) or (verified_partial and fi in completed_from_partial):
                        payload = candidate
                        reused = True
                except Exception:
                    payload = None
            if payload is None:
                payload = self.process_frame(item["path"], fi).to_dict()
                _atomic_write_json(target, payload)

            records.append({"frame_index": fi, "perception_json": str(target.resolve())})
            checkpoint["completed_frames"] = sorted(set(checkpoint["completed_frames"]) | {fi})
            checkpoint["updated_unix"] = time.time()
            checkpoint["last_frame"] = fi
            _atomic_write_json(partial_path, checkpoint)

            event = {
                "sequence_index": idx,
                "sequence_total": total,
                "frame_index": fi,
                "reused": reused,
                "humans": len(payload.get("humans", [])),
                "poses": len(payload.get("pose_cache", {})),
                "balls": len(payload.get("ball_detections", [])),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_path": str(partial_path.resolve()),
            }
            if progress_callback is not None:
                progress_callback(event)
            else:
                print(
                    f"[PERCEPTION {idx}/{total}] frame={fi} {'reuse' if reused else 'infer'} "
                    f"humans={event['humans']} pose={event['poses']} ball={event['balls']}",
                    flush=True,
                )

        manifest = {
            "schema_version": "stage2-sst-rtmw-perception-manifest-1.0",
            "stage2_version": STAGE2_VERSION,
            "models": model_info,
            "configuration": configuration,
            "frames": records,
            "timing_seconds": round(time.perf_counter() - started, 4),
        }
        _atomic_write_json(manifest_path, manifest)
        checkpoint["status"] = "COMPLETE"
        checkpoint["completed_frames"] = [int(x["frame_index"]) for x in records]
        checkpoint["updated_unix"] = time.time()
        checkpoint["manifest_path"] = str(manifest_path.resolve())
        _atomic_write_json(partial_path, checkpoint)
        return manifest

