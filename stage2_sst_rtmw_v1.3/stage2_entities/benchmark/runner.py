from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import tempfile
import time
import hashlib
import platform
import sys

from ..contracts import ReplayContext
from ..perception import SSTRTMWStage2Backend
from ..backend_core.sst_rtmw_pose_pipeline import (
    RTMWOpenCVDNNPose, WHOLEBODY_KEYPOINT_NAMES, pose_quality,
)
from ..tracking import build_entity_track_state
from .soccernet_gsr import SoccerNetGSRDataset, SoccerNetSequence, HUMAN_ROLES, CANDIDATE_ROLES
from .protocols import BenchmarkProtocol, WindowSpec, build_protocol_windows
from .metrics_detection import (
    DetectionAccumulator, detection_ap, evaluate_selected_frame,
    gt_cross_class_duplicate_rate, gt_post_consolidation_duplicate_rate,
)
from .metrics_tracking import evaluate_tracking_window, combine_tracking_results
from .reporting import write_json, write_rows_csv, write_confusion_matrix, flatten_summary, write_failure_overlay
from .checkpointing import BenchmarkCheckpointStore, ConsoleProgress, atomic_write_json, stable_json_hash


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_frame_map(manifest: Mapping[str, Any]) -> Dict[int, str]:
    return {int(x["frame_index"]): str(x["perception_json"]) for x in manifest.get("frames", [])}


def _slice_manifest(manifest: Mapping[str, Any], start: int, end: int, *, strip_pose: bool = False, temp_dir: Optional[Path] = None) -> Dict[str, Any]:
    records=[]
    for rec in manifest.get("frames",[]):
        fi=int(rec["frame_index"])
        if not (start<=fi<=end): continue
        if not strip_pose:
            records.append(dict(rec)); continue
        if temp_dir is None: raise ValueError("temp_dir required when strip_pose=True")
        payload=_load_json(rec["perception_json"]); payload["pose_cache"]={}
        target=temp_dir/f"frame_{fi:09d}_no_pose.json"; write_json(target,payload)
        records.append({"frame_index":fi,"perception_json":str(target.resolve())})
    return {"schema_version":manifest.get("schema_version","benchmark-slice"),"models":manifest.get("models",{}),"configuration":manifest.get("configuration",{}),"frames":records,"benchmark_strip_pose":strip_pose}


def _oracle_manifest(seq: SoccerNetSequence, window: WindowSpec, temp_dir: Path) -> Dict[str, Any]:
    records=[]
    for fi in window.frame_indices:
        humans=[]
        # Per-frame local IDs intentionally do not encode GT identity, so association cannot cheat.
        for j,g in enumerate(seq.gt(fi),start=1):
            evidence={r:0.0 for r in ("player","goalkeeper","referee","other")}; evidence[g.role]=1.0
            humans.append({
                "physical_human_id":f"oracle_human_{j:03d}","bbox_xyxy":list(g.bbox_xyxy),"detector_score":1.0,
                "resolved_role":g.role,"role_evidence":evidence,"role_status":"VALID","source_detection_ids":[f"oracle_det_{j:03d}"],
            })
        payload={"frame_index":fi,"image_width":seq.width,"image_height":seq.height,"raw_detections":[],"humans":humans,"pose_cache":{},"ball_detections":[],"provenance":{"backend":"oracle-gt-boxes-no-identity-no-pose"}}
        p=temp_dir/f"oracle_{fi:09d}.json"; write_json(p,payload)
        records.append({"frame_index":fi,"perception_json":str(p.resolve())})
    return {"schema_version":"stage2-oracle-boxes-1.0","frames":records}



def _oracle_pose_manifest(
    seq: SoccerNetSequence,
    window: WindowSpec,
    cache_root: Path,
    pose_estimator: RTMWOpenCVDNNPose,
    *,
    keypoint_threshold: float = 1.0,
    expected_rtmw_sha256: str = "",
    overwrite: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """GT boxes + GT role + RTMW pose, without exposing GT identity to association.

    Per-frame oracle pose payloads are cached by sequence frame so overlapping t0 windows do
    not rerun RTMW. The physical-human IDs are local to each frame and deliberately unrelated
    to SoccerNet track IDs.
    """
    import cv2
    import numpy as np

    cache_root.mkdir(parents=True, exist_ok=True)
    frame_map = seq.materialize_frames(window.frame_indices, cache_root.parent / "oracle_gt_frames")
    image_by_frame = {int(x["global_frame_index"]): Path(x["path"]) for x in frame_map["frames"]}
    records: List[Dict[str, Any]] = []
    oracle_started = time.perf_counter()
    total_oracle_frames = len(window.frame_indices)
    for oracle_index, fi in enumerate(window.frame_indices, start=1):
        target = cache_root / f"frame_{fi:09d}_oracle_pose.json"
        reused = False
        if target.is_file() and not overwrite:
            try:
                cached = _load_json(target)
                cached_hash = str((cached.get("provenance") or {}).get("rtmw_sha256", ""))
                if expected_rtmw_sha256 and cached_hash and cached_hash.lower() != expected_rtmw_sha256.lower():
                    raise RuntimeError(
                        f"{seq.sequence_id} frame {fi}: oracle-pose cache uses a different RTMW model. "
                        "Pass --overwrite-perception or use a new output directory."
                    )
                if expected_rtmw_sha256 and not cached_hash:
                    raise RuntimeError(
                        f"{seq.sequence_id} frame {fi}: oracle-pose cache has no RTMW fingerprint. "
                        "Pass --overwrite-perception to rebuild it safely."
                    )
                reused = True
            except RuntimeError:
                raise
            except Exception:
                reused = False
        if not target.is_file() or overwrite or not reused:
            image_path = image_by_frame[fi]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            gt = seq.gt(fi)
            humans: List[Dict[str, Any]] = []
            pose_cache: Dict[str, Any] = {}
            pose_gt = [g for g in gt if g.role in CANDIDATE_ROLES]
            pose_boxes = [list(g.bbox_xyxy) for g in pose_gt]
            poses = pose_estimator.infer(image, pose_boxes) if pose_boxes else []
            if len(poses) != len(pose_gt):
                raise RuntimeError(f"{seq.sequence_id} frame {fi}: RTMW returned {len(poses)} poses for {len(pose_gt)} GT candidate boxes")
            pose_by_index = {id(g): pose for g, pose in zip(pose_gt, poses)}
            for j, g in enumerate(gt, start=1):
                hid = f"oracle_human_{j:03d}"
                evidence = {r: 0.0 for r in ("player", "goalkeeper", "referee", "other")}
                evidence[g.role] = 1.0
                humans.append({
                    "physical_human_id": hid,
                    "bbox_xyxy": list(g.bbox_xyxy),
                    "detector_score": 1.0,
                    "resolved_role": g.role,
                    "role_evidence": evidence,
                    "role_status": "VALID",
                    "source_detection_ids": [f"oracle_det_{j:03d}"],
                })
                pose = pose_by_index.get(id(g))
                if pose is not None:
                    quality = pose_quality(pose, detection_score=1.0, bbox_xyxy=g.bbox_xyxy, keypoint_threshold=keypoint_threshold)
                    xy = np.asarray(pose.keypoints_xy, dtype=float)
                    scores = np.asarray(pose.scores, dtype=float)
                    keypoints = []
                    for ki, name in enumerate(WHOLEBODY_KEYPOINT_NAMES):
                        raw = float(scores[ki]) if ki < scores.shape[0] else 0.0
                        point = xy[ki] if ki < xy.shape[0] else np.asarray([np.nan, np.nan])
                        keypoints.append({
                            "index": ki,
                            "name": name,
                            "x": None if not np.isfinite(point[0]) else float(point[0]),
                            "y": None if not np.isfinite(point[1]) else float(point[1]),
                            "raw_score": raw,
                            "visible": bool(np.isfinite(point).all() and raw >= keypoint_threshold),
                        })
                    pose_cache[hid] = {
                        "physical_human_id": hid,
                        "source_detection_ids": [f"oracle_det_{j:03d}"],
                        "bbox_xyxy": list(g.bbox_xyxy),
                        "expanded_pose_bbox_xyxy": list(g.bbox_xyxy),
                        "backend": str(pose.backend),
                        "is_learned_model": True,
                        "score_semantics": "RTMW_RAW_SIMCC_MAX_NOT_CALIBRATED_PROBABILITY",
                        "keypoint_visibility_threshold": float(keypoint_threshold),
                        "quality": {
                            "score": float(quality.score), "level": str(quality.level),
                            "visible_core_points": int(quality.visible_core_points),
                            "total_core_points": int(quality.total_core_points),
                            "visible_core_ratio": float(quality.visible_core_ratio),
                            "mean_core_confidence": float(quality.mean_core_confidence),
                            "bbox_height_px": float(quality.bbox_height_px),
                        },
                        "keypoints": keypoints,
                    }
            payload = {
                "frame_index": fi,
                "image_path": str(image_path.resolve()),
                "image_width": seq.width,
                "image_height": seq.height,
                "raw_detections": [],
                "humans": humans,
                "pose_cache": pose_cache,
                "ball_detections": [],
                "provenance": {
                    "backend": "oracle-gt-boxes-and-roles-plus-rtmw-no-identity",
                    "warning": "GT bbox/role are oracle inputs; SoccerNet track identity is never passed to Stage-2 association.",
                    "rtmw_sha256": expected_rtmw_sha256,
                },
            }
            atomic_write_json(target, payload)
        records.append({"frame_index": fi, "perception_json": str(target.resolve())})
        if progress_callback is not None:
            progress_callback({
                "sequence_index": oracle_index,
                "sequence_total": total_oracle_frames,
                "frame_index": fi,
                "reused": reused and not overwrite,
                "humans": len(seq.gt(fi)),
                "poses": len([g for g in seq.gt(fi) if g.role in CANDIDATE_ROLES]),
                "balls": 0,
                "elapsed_seconds": time.perf_counter() - oracle_started,
            })
    return {"schema_version": "stage2-oracle-boxes-rtmw-1.0", "frames": records}


def _context(seq: SoccerNetSequence, w: WindowSpec) -> ReplayContext:
    source=str(seq.video_path or seq.labels_path)
    ctx=ReplayContext(
        schema_version="benchmark-soccernet-gsr-1.0",video_path=source,video_id=seq.sequence_id,
        fps=seq.fps,frame_count=seq.num_frames,image_width=seq.width,image_height=seq.height,
        selected_frame=w.target_frame,window_start=w.window_start,window_end=w.window_end,
        shot_start=0,shot_end=seq.num_frames-1,coordinate_space="RAW_DISTORTED_PIXEL",
        extra={"source":"SoccerNet-GSR","dataset_version":seq.version,"labels_path":str(seq.labels_path),"benchmark_assumption":"sequence treated as one shot; Stage 2 does not run shot detection during benchmark"},
    ); ctx.validate(); return ctx


def _pred_records(state, *, candidate_only: bool) -> List[Dict[str, Any]]:
    out=[]
    for track in state.tracks:
        if candidate_only and not track.candidate_for_stage3: continue
        for obs in track.observations:
            out.append({"frame_index":obs.frame_index,"track_id":track.track_id,"bbox_xyxy":list(obs.bbox_xyxy),"role":track.role,"candidate_for_stage3":track.candidate_for_stage3,"score":float(obs.detector_score)})
    return out


def _gt_records(seq: SoccerNetSequence, w: WindowSpec, *, candidate_only: bool) -> List[Dict[str, Any]]:
    roles=CANDIDATE_ROLES if candidate_only else HUMAN_ROLES
    anchors={g.track_id for g in seq.gt(w.target_frame,roles=set(roles))}
    result=[]
    for fi in w.frame_indices:
        for g in seq.gt(fi,roles=set(roles)):
            if g.track_id in anchors: result.append(g.to_dict())
    return result


def _selected_payload(manifest: Mapping[str, Any], target_frame: int) -> Dict[str, Any]:
    rec=next((x for x in manifest.get("frames",[]) if int(x["frame_index"])==target_frame),None)
    if rec is None: raise KeyError(f"Perception manifest missing target frame {target_frame}")
    return _load_json(rec["perception_json"])


def _ap_records_from_payload(seq_id: str, target_frame: int, payload: Mapping[str, Any], gt: Sequence[Mapping[str,Any]]):
    key=f"{seq_id}:{target_frame}"
    preds=[]
    for h in payload.get("humans",[]):
        preds.append({"frame_key":key,"bbox_xyxy":h["bbox_xyxy"],"score":float(h.get("detector_score",0.0)),"role":str(h.get("resolved_role","other"))})
    gts=[{"frame_key":key,"bbox_xyxy":x["bbox_xyxy"],"role":x["role"]} for x in gt]
    return preds,gts


CURRENT_STAGE2_VERSION = "stage2-sst-rtmw-1.3.0"
COMPATIBLE_PERCEPTION_VERSIONS = {"stage2-sst-rtmw-1.3.0"}


def _sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _expected_model_hashes(cfg: "BenchmarkConfig") -> Dict[str, str]:
    result: Dict[str, str] = {}
    if cfg.sst_checkpoint is not None:
        if not Path(cfg.sst_checkpoint).is_file():
            raise FileNotFoundError(f"SST checkpoint not found: {cfg.sst_checkpoint}")
        result["sst"] = _sha256_file(cfg.sst_checkpoint)
    if cfg.rtmw_model is not None:
        if not Path(cfg.rtmw_model).is_file():
            raise FileNotFoundError(f"RTMW model not found: {cfg.rtmw_model}")
        result["rtmw"] = _sha256_file(cfg.rtmw_model)
    return result


def _expected_perception_configuration(cfg: "BenchmarkConfig") -> Dict[str, Any]:
    return {
        "class_thresholds": {
            "Ball": float(cfg.ball_threshold), "Player": float(cfg.player_threshold), "Goalkeeper": float(cfg.goalkeeper_threshold),
            "Main referee": float(cfg.referee_threshold), "Side referee": float(cfg.referee_threshold), "Staff members": float(cfg.staff_threshold),
        },
        "person_nms_iou": float(cfg.person_nms_iou),
        "ball_nms_iou": float(cfg.ball_nms_iou),
        "consolidation_high_iou": float(cfg.consolidation_high_iou),
        "consolidation_low_iou": float(cfg.consolidation_low_iou),
        "consolidation_max_center_distance": float(cfg.consolidation_max_center_distance),
        "consolidation_min_area_similarity": float(cfg.consolidation_min_area_similarity),
        "consolidation_min_intersection_over_min": float(cfg.consolidation_min_intersection_over_min),
        "raw_human_score_floor": float(cfg.raw_human_score_floor),
        "rescue_overlap_with_active_iou": float(cfg.rescue_overlap_with_active_iou),
        "role_ambiguous_margin": 0.05,
        "keypoint_threshold": 1.0,
    }


def _cache_status(path: Path, required: Sequence[int], expected_hashes: Mapping[str, str], expected_configuration: Optional[Mapping[str, Any]] = None) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Return (status, reason, manifest).

    status is one of: missing, incomplete, reusable, incompatible.
    """
    if not path.is_file():
        return "missing", "perception_manifest.json does not exist", None
    try:
        manifest = _load_json(path)
    except Exception as exc:
        return "incompatible", f"cannot parse cache manifest: {exc}", None
    cached_version = str(manifest.get("stage2_version", ""))
    if cached_version not in COMPATIBLE_PERCEPTION_VERSIONS:
        return "incompatible", (
            f"cache stage2_version={cached_version!r}, compatible={sorted(COMPATIBLE_PERCEPTION_VERSIONS)!r}"
        ), manifest
    have = set(_manifest_frame_map(manifest))
    missing = sorted(set(required) - have)
    # If caller supplied model files, their content must match the cached content hashes.
    models = manifest.get("models") or {}
    cached_hashes = {
        "sst": str((models.get("sst") or {}).get("sha256", "")),
        "rtmw": str((models.get("rtmw") or {}).get("sha256", "")),
    }
    for key, expected in expected_hashes.items():
        cached = cached_hashes.get(key, "")
        if not cached:
            return "incompatible", f"cache has no {key} SHA256 fingerprint", manifest
        if cached.lower() != expected.lower():
            return "incompatible", f"{key} model SHA256 differs from cache", manifest
    if expected_configuration is not None and dict(manifest.get("configuration") or {}) != dict(expected_configuration):
        return "incompatible", "perception configuration differs from requested M1 configuration", manifest
    if missing:
        return "incomplete", f"missing {len(missing)} required frames ({missing[:3]}{'...' if len(missing)>3 else ''})", manifest
    return "reusable", "all required frames and cache fingerprints are compatible", manifest


def _environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "scipy", "cv2", "torch", "torchvision"):
        try:
            module = __import__(module_name)
            info[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            info[module_name] = None
    return info


def _gate_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    d=summary["detection"]; c=summary["tracking_candidate"]
    hard={
        "CandidateRecall>=0.95": d.get("CandidateRecall",0)>=0.95,
        "CandidatePrecision>=0.95": d.get("CandidatePrecision",0)>=0.95,
        "RefereeLeakage<=0.02": d.get("RefereeLeakageRate",1)<=0.02,
        "AnyDuplicateAfter<0.02": d.get("AnyDuplicateRate_after",1)<0.02,
        "HOTA-Candidate>=0.60": c.get("HOTA",0)>=0.60,
        "AssA-Candidate>=0.55": c.get("AssA",0)>=0.55,
        "IDF1-Candidate>=0.60": c.get("IDF1",0)>=0.60,
        "TCR@1s>=0.90": c.get("TCR",0)>=0.90,
    }
    quality={
        "PlayerRecall>=0.95": d.get("role_metrics",{}).get("player",{}).get("recall",0)>=0.95,
        "GoalkeeperRecall>=0.90": d.get("role_metrics",{}).get("goalkeeper",{}).get("recall",0)>=0.90,
        "RoleMacroF1>=0.90": d.get("role_macro_f1_supported",0)>=0.90,
        "AnchorCoverage>=0.95": c.get("AnchorCoverage",0)>=0.95,
        "ConditionalTCR>=0.95": c.get("ConditionalTCR",0)>=0.95,
    }
    return {
        "checks": hard,
        "hard_checks": hard,
        "quality_checks": quality,
        "passed": all(hard.values()),
        "quality_all_passed": all(quality.values()),
        "note": "Exact Player/GK role quality is diagnostic; Stage-3 eligibility is footballer-vs-non-footballer.",
    }


@dataclass
class BenchmarkConfig:
    dataset_root: Path
    split: str
    output_dir: Path
    protocol: BenchmarkProtocol
    sequence_ids: Optional[List[str]] = None
    max_sequences: Optional[int] = None
    sst_checkpoint: Optional[Path] = None
    rtmw_model: Optional[Path] = None
    sst_device: str = "auto"
    rtmw_device: str = "cpu"
    overwrite_perception: bool = False
    require_version_13: bool = True
    experiments: Tuple[str,...] = ("primary","no_pose","oracle_boxes_geom")
    player_threshold: float = 0.50
    goalkeeper_threshold: float = 0.50
    referee_threshold: float = 0.55
    staff_threshold: float = 0.60
    ball_threshold: float = 0.35
    person_nms_iou: float = 0.65
    ball_nms_iou: float = 0.30
    max_assignment_cost: float = 0.92
    max_gap: int = 6
    use_temporal_rescue: bool = True
    rescue_max_assignment_cost: float = 0.78
    raw_human_score_floor: float = 0.20
    consolidation_high_iou: float = 0.82
    consolidation_low_iou: float = 0.60
    consolidation_max_center_distance: float = 0.18
    consolidation_min_area_similarity: float = 0.65
    consolidation_min_intersection_over_min: float = 0.75
    rescue_overlap_with_active_iou: float = 0.65
    failure_images: int = 20
    restart_evaluation: bool = False
    progress_every: int = 10
    quiet: bool = False
    # Test-only fault injection. Not exposed by the CLI.
    debug_interrupt_after_new_eval_units: Optional[int] = None


def _benchmark_code_fingerprint() -> str:
    files = [
        Path(__file__),
        Path(__file__).with_name("metrics_detection.py"),
        Path(__file__).with_name("metrics_tracking.py"),
        Path(__file__).with_name("protocols.py"),
        Path(__file__).with_name("soccernet_gsr.py"),
        Path(__file__).parent.parent / "tracking.py",
        Path(__file__).parent.parent / "consolidation.py",
    ]
    h = hashlib.sha256()
    for item in files:
        h.update(item.name.encode("utf-8"))
        h.update(item.read_bytes())
    return h.hexdigest()


def _benchmark_run_identity(cfg: BenchmarkConfig, plans: Sequence[Tuple[str, SoccerNetSequence, Sequence[WindowSpec], Sequence[int]]]) -> Dict[str, Any]:
    return {
        "benchmark_code_sha256": _benchmark_code_fingerprint(),
        "stage2_version": CURRENT_STAGE2_VERSION,
        "dataset_root": str(Path(cfg.dataset_root).resolve()),
        "split": cfg.split,
        "require_version_13": cfg.require_version_13,
        "protocol": {
            "name": cfg.protocol.name,
            "target_fractions": list(cfg.protocol.target_fractions),
            "half_window_seconds": cfg.protocol.half_window_seconds,
        },
        "experiments": list(cfg.experiments),
        "detector_thresholds": {
            "player": cfg.player_threshold, "goalkeeper": cfg.goalkeeper_threshold,
            "referee": cfg.referee_threshold, "staff": cfg.staff_threshold, "ball": cfg.ball_threshold,
            "person_nms_iou": cfg.person_nms_iou, "ball_nms_iou": cfg.ball_nms_iou,
        },
        "max_assignment_cost": cfg.max_assignment_cost,
        "max_gap": cfg.max_gap,
        "use_temporal_rescue": cfg.use_temporal_rescue,
        "rescue_max_assignment_cost": cfg.rescue_max_assignment_cost,
        "raw_human_score_floor": cfg.raw_human_score_floor,
        "consolidation": {
            "high_iou": cfg.consolidation_high_iou,
            "low_iou": cfg.consolidation_low_iou,
            "max_center_distance": cfg.consolidation_max_center_distance,
            "min_area_similarity": cfg.consolidation_min_area_similarity,
            "min_intersection_over_min": cfg.consolidation_min_intersection_over_min,
        },
        "sequences": [
            {
                "sequence_id": sid,
                "labels_sha256": _sha256_file(seq.labels_path),
                "version": seq.version,
                "fps": seq.fps,
                "num_frames": seq.num_frames,
                "windows": [w.to_dict() for w in windows],
                "required_frames": list(required),
            }
            for sid, seq, windows, required in plans
        ],
    }


def _tracking_text(result: Mapping[str, Any]) -> str:
    return (
        f"HOTA={float(result.get('HOTA',0)):.3f} "
        f"AssA={float(result.get('AssA',0)):.3f} "
        f"IDF1={float(result.get('IDF1',0)):.3f} "
        f"TCR={float(result.get('TCR',0)):.3f} "
        f"AC={float(result.get('AnchorCoverage',0)):.3f} "
        f"cTCR={float(result.get('ConditionalTCR',0)):.3f}"
    )


def _write_partial_summary(store: BenchmarkCheckpointStore, output_dir: Path) -> None:
    detection_acc = DetectionAccumulator()
    for rec in store.iter_detection():
        detection_acc.add(rec["selected_frame_metrics"], rec["raw_duplicate"], rec["post_duplicate"])
    primary_candidate = [rec["candidate"] for rec in store.iter_tracking("primary") if "candidate" in rec]
    det = detection_acc.summary() if detection_acc.selected_frames else None
    track = combine_tracking_results(primary_candidate) if primary_candidate else None
    atomic_write_json(output_dir / "checkpoints" / "partial_summary.json", {
        "schema_version": "stage2-benchmark-partial-summary-1.0",
        "checkpoint_status": store.status(),
        "detection": det,
        "tracking_candidate_primary": track,
    })


def _aggregate_completed(
    *,
    cfg: BenchmarkConfig,
    out: Path,
    ids: Sequence[str],
    plans: Sequence[Tuple[str, SoccerNetSequence, Sequence[WindowSpec], Sequence[int]]],
    store: BenchmarkCheckpointStore,
    observed_model_fingerprints: Mapping[str, str],
    observed_perception_configurations: Sequence[Mapping[str, Any]],
    dataset_versions: Mapping[str, int],
    cache_audit: Sequence[Mapping[str, Any]],
    perception_timing: Mapping[str, float],
    started: float,
    require_complete: bool = True,
) -> Dict[str, Any]:
    detection_acc = DetectionAccumulator()
    ap_preds: List[Dict[str, Any]] = []
    ap_gt: List[Dict[str, Any]] = []
    tracking: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
        e: {"human": [], "candidate": []} for e in cfg.experiments
    }
    window_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []
    failure_image_count = 0
    tracking_runtime = 0.0

    for sid, seq, windows, _required in plans:
        for w in windows:
            protocol_rows.append(w.to_dict())
            det_ck = store.load_detection(sid, w.target_frame)
            tr_cks = {exp: store.load_tracking(sid, w.target_frame, exp) for exp in cfg.experiments}
            if det_ck is None or any(x is None for x in tr_cks.values()):
                if require_complete:
                    missing = ["detection"] if det_ck is None else []
                    missing += [f"tracking:{e}" for e,x in tr_cks.items() if x is None]
                    raise RuntimeError(f"Incomplete benchmark checkpoint for {sid} t0={w.target_frame}: {missing}")
                continue

            sel_metrics = det_ck["selected_frame_metrics"]
            raw_dup = det_ck["raw_duplicate"]
            post_dup = det_ck["post_duplicate"]
            detection_acc.add(sel_metrics, raw_dup, post_dup)
            ap_preds.extend(det_ck.get("ap_predictions", []))
            ap_gt.extend(det_ck.get("ap_ground_truth", []))

            row = {
                "sequence_id": sid,
                "target_frame": w.target_frame,
                "window_start": w.window_start,
                "window_end": w.window_end,
                "CandidatePrecision": sel_metrics.get("CandidatePrecision", 0.0),
                "CandidateRecall": sel_metrics["CandidateRecall"],
                "RefereeLeakageRate": sel_metrics["RefereeLeakageRate"],
                "RoleMacroF1": sel_metrics["role_macro_f1_supported"],
                "AnyDuplicateRate_before": raw_dup.get("AnyDuplicateRate", 0.0),
                "AnyDuplicateRate_after": post_dup.get("AnyDuplicateRate", 0.0),
                "CrossClassConflictRate_before": raw_dup.get("CrossClassConflictRate", raw_dup.get("CrossClassDuplicateRate", 0.0)),
                "CrossClassConflictRate_after": post_dup.get("CrossClassConflictRate", 0.0),
            }
            for exp, ck in tr_cks.items():
                assert ck is not None
                tracking[exp]["human"].append(ck["human"])
                tracking[exp]["candidate"].append(ck["candidate"])
                tracking_runtime += float(ck.get("tracking_runtime_seconds", 0.0) or 0.0) if exp == "primary" else 0.0
                if exp == "primary":
                    result = ck["candidate"]
                    row.update({f"candidate_{k}": result.get(k) for k in ("HOTA","DetA","AssA","IDF1","TCR","AnchorCoverage","ConditionalTCR","IDSW","Fragmentations")})
            window_rows.append(row)

            primary_candidate = tr_cks["primary"]["candidate"]
            if (
                sel_metrics["CandidateRecall"] < 1.0
                or sel_metrics["RefereeLeakageRate"] > 0
                or sel_metrics["role_accuracy_on_matched"] < 1.0
                or post_dup.get("AnyDuplicateRate", post_dup.get("PostConsolidationDuplicateRate", 0)) > 0
                or primary_candidate.get("TCR", 1) < 0.90
                or primary_candidate.get("AssA", 1) < 0.55
                or primary_candidate.get("IDF1", 1) < 0.60
            ):
                fail = {
                    "sequence_id": sid,
                    "target_frame": w.target_frame,
                    "window": [w.window_start, w.window_end],
                    "selected_frame_metrics": sel_metrics,
                    "raw_duplicate": raw_dup,
                    "post_duplicate": post_dup,
                    "primary_candidate_tracking": primary_candidate,
                }
                if failure_image_count < cfg.failure_images:
                    img_path = det_ck.get("image_path")
                    if img_path and Path(img_path).is_file():
                        viz = write_failure_overlay(
                            img_path,
                            det_ck.get("gt_selected", []),
                            det_ck.get("pred_selected", []),
                            out / "failure_cases" / f"{sid}_t{w.target_frame:06d}.png",
                            title=f"{sid} t0={w.target_frame}",
                        )
                        if viz is not None:
                            fail["overlay"] = str(viz.resolve())
                            failure_image_count += 1
                failures.append(fail)

    tracking_summary = {
        exp: {name: combine_tracking_results(vals) for name, vals in groups.items()}
        for exp, groups in tracking.items()
    }
    ap = {"all_humans": detection_ap(ap_preds, ap_gt)}
    for r in ("player","goalkeeper","referee","other"):
        ap[r] = detection_ap(ap_preds, ap_gt, role=r)

    completed_windows = len(window_rows)
    expected_windows = sum(len(windows) for _sid,_seq,windows,_req in plans)
    summary = {
        "schema_version": "stage2-benchmark-report-1.3",
        "stage2_version": CURRENT_STAGE2_VERSION,
        "dataset": {
            "root": str(cfg.dataset_root.resolve()), "split": cfg.split,
            "sequences": len(ids), "sequence_ids": list(ids),
            "versions": dict(dataset_versions), "require_version_13": cfg.require_version_13,
        },
        "protocol": {
            "name": cfg.protocol.name,
            "target_fractions": list(cfg.protocol.target_fractions),
            "half_window_seconds": cfg.protocol.half_window_seconds,
            "windows": completed_windows,
            "expected_windows": expected_windows,
        },
        "configuration": {
            "experiments": list(cfg.experiments),
            "detector_thresholds": {
                "player": cfg.player_threshold, "goalkeeper": cfg.goalkeeper_threshold,
                "referee": cfg.referee_threshold, "staff": cfg.staff_threshold, "ball": cfg.ball_threshold,
                "person_nms_iou": cfg.person_nms_iou, "ball_nms_iou": cfg.ball_nms_iou,
            },
            "max_assignment_cost": cfg.max_assignment_cost,
            "max_gap": cfg.max_gap, "use_temporal_rescue": cfg.use_temporal_rescue,
            "rescue_max_assignment_cost": cfg.rescue_max_assignment_cost,
            "raw_human_score_floor": cfg.raw_human_score_floor,
            "consolidation_high_iou": cfg.consolidation_high_iou,
            "consolidation_low_iou": cfg.consolidation_low_iou,
            "consolidation_max_center_distance": cfg.consolidation_max_center_distance,
            "consolidation_min_area_similarity": cfg.consolidation_min_area_similarity,
            "consolidation_min_intersection_over_min": cfg.consolidation_min_intersection_over_min,
            "sst_device": cfg.sst_device, "rtmw_device": cfg.rtmw_device,
            "checkpoint_resume": True,
        },
        "model_fingerprints": dict(observed_model_fingerprints),
        "perception_configurations": [dict(x) for x in observed_perception_configurations],
        "environment": _environment_info(),
        "cache_audit": list(cache_audit),
        "checkpoint": store.status(),
        "detection": detection_acc.summary(),
        "detection_ap": ap,
        "tracking_human": tracking_summary["primary"]["human"],
        "tracking_candidate": tracking_summary["primary"]["candidate"],
        "ablations": {exp: groups for exp, groups in tracking_summary.items() if exp != "primary"},
        "runtime_seconds": time.perf_counter() - started,
        "runtime": {
            "perception_frames": int(perception_timing.get("frames",0)),
            "sst_ms_per_frame": 1000.0 * float(perception_timing.get("sst_detection",0.0)) / max(1,int(perception_timing.get("frames",0))),
            "rtmw_ms_per_frame": 1000.0 * float(perception_timing.get("rtmw_pose",0.0)) / max(1,int(perception_timing.get("frames",0))),
            "perception_total_ms_per_frame": 1000.0 * float(perception_timing.get("total_after_models_loaded",0.0)) / max(1,int(perception_timing.get("frames",0))),
            "tracking_ms_per_window_primary": 1000.0 * tracking_runtime / max(1,completed_windows),
            "note": "Perception timing comes from cached per-frame JSON. Tracking timing comes from checkpointed evaluation tasks and survives resume.",
        },
    }
    summary["acceptance_gate"] = _gate_summary(summary)

    write_json(out / "benchmark_summary.json", summary)
    write_rows_csv(out / "benchmark_summary.csv", [flatten_summary(summary)])
    write_rows_csv(out / "window_metrics.csv", window_rows)
    write_json(out / "protocol_windows.json", protocol_rows)
    write_json(out / "failure_cases.json", failures)
    write_json(out / "detection_ap.json", ap)
    write_rows_csv(out / "role_confusion_matrix.csv", [
        {"gt_role": g, **{p: summary["detection"]["confusion"][g][p] for p in ("player","goalkeeper","referee","other")}}
        for g in ("player","goalkeeper","referee","other")
    ])
    png = write_confusion_matrix(out / "role_confusion_matrix.png", summary["detection"]["confusion"])
    write_json(out / "benchmark_manifest.json", {
        "summary": str((out / "benchmark_summary.json").resolve()),
        "window_metrics": str((out / "window_metrics.csv").resolve()),
        "failure_cases": str((out / "failure_cases.json").resolve()),
        "confusion_png": str(png.resolve()) if png else None,
        "checkpoint_state": str((out / "checkpoints" / "benchmark_state.json").resolve()),
        "partial_summary": str((out / "checkpoints" / "partial_summary.json").resolve()),
        "note": "HOTA and IDF1 use a self-contained implementation of the public TrackEval reference equations in image IoU space. Use SoccerNet sn-trackeval as an external cross-check before publication.",
    })
    return summary


def _clear_final_reports(output_dir: Path) -> None:
    for name in (
        "benchmark_summary.json", "benchmark_summary.csv", "window_metrics.csv",
        "protocol_windows.json", "failure_cases.json", "detection_ap.json",
        "role_confusion_matrix.csv", "role_confusion_matrix.png", "benchmark_manifest.json",
    ):
        p = output_dir / name
        if p.is_file():
            p.unlink()
    failure_dir = output_dir / "failure_cases"
    if failure_dir.is_dir():
        import shutil
        shutil.rmtree(failure_dir)


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    out = cfg.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cfg.restart_evaluation:
        _clear_final_reports(out)
    dataset = SoccerNetGSRDataset(cfg.dataset_root, cfg.split, require_version_13=cfg.require_version_13)
    ids = cfg.sequence_ids or dataset.discover()
    limit = cfg.max_sequences if cfg.max_sequences is not None else cfg.protocol.max_sequences
    if limit is not None:
        ids = ids[:int(limit)]
    if not ids:
        raise RuntimeError("No SoccerNet-GSR sequences selected")

    # Build a deterministic plan before any expensive model work. The run signature includes
    # label-file hashes and exact windows, so stale evaluation checkpoints cannot be mixed.
    plans: List[Tuple[str, SoccerNetSequence, Sequence[WindowSpec], Sequence[int]]] = []
    dataset_versions: Dict[str, int] = {}
    for sid in ids:
        seq = dataset.load(sid)
        windows = build_protocol_windows(seq, cfg.protocol)
        required = sorted({fi for w in windows for fi in w.frame_indices})
        plans.append((sid, seq, windows, required))
        dataset_versions[seq.version] = dataset_versions.get(seq.version, 0) + 1
    total_windows = sum(len(windows) for _sid,_seq,windows,_required in plans)
    if total_windows == 0:
        raise RuntimeError("Benchmark protocol produced no windows")

    run_identity = _benchmark_run_identity(cfg, plans)
    store = BenchmarkCheckpointStore(
        out, run_identity, total_windows=total_windows, experiments=cfg.experiments,
        restart=cfg.restart_evaluation,
    )
    progress = ConsoleProgress(
        total_eval_units=store.total_units,
        enabled=not cfg.quiet,
        perception_every=cfg.progress_every,
    )
    store.mark_status("RUNNING")
    initial_status = store.status()
    progress.set_initial_completed(int(initial_status.get("completed_eval_units", 0)))
    progress.log(
        f"[START] split={cfg.split} protocol={cfg.protocol.name} sequences={len(ids)} "
        f"windows={total_windows} experiments={','.join(cfg.experiments)}"
    )
    if progress.initial_completed_eval_units:
        progress.log(
            f"[RESUME] found {progress.initial_completed_eval_units}/{store.total_units} completed evaluation units "
            f"({100.0*progress.initial_completed_eval_units/max(1,store.total_units):.1f}%)."
        )

    backend = None
    oracle_pose_estimator = None
    expected_hashes = _expected_model_hashes(cfg)
    expected_perception_configuration = _expected_perception_configuration(cfg)
    observed_model_fingerprints: Dict[str, str] = dict(expected_hashes)
    observed_perception_configurations: List[Dict[str, Any]] = []
    cache_audit: List[Dict[str, Any]] = []
    started = time.perf_counter()
    perception_timing = {"frames":0,"sst_detection":0.0,"rtmw_pose":0.0,"total_after_models_loaded":0.0}
    new_eval_units = 0

    def maybe_debug_interrupt() -> None:
        nonlocal new_eval_units
        if cfg.debug_interrupt_after_new_eval_units is not None and new_eval_units >= cfg.debug_interrupt_after_new_eval_units:
            raise KeyboardInterrupt("debug checkpoint interrupt")

    try:
        for si, (sid, seq, windows, required) in enumerate(plans, start=1):
            progress.log(
                f"[SEQ {si}/{len(plans)}] {sid} version={seq.version} fps={seq.fps:g} "
                f"windows={len(windows)} inference_frames={len(required)}"
            )
            seq_out = out / "sequences" / sid
            cache_dir = seq_out / "perception"
            manifest_path = cache_dir / "perception_manifest.json"
            cache_status, cache_reason, cached_manifest = _cache_status(
                manifest_path, required, expected_hashes, expected_perception_configuration
            )
            cache_audit.append({"sequence_id": sid, "status": cache_status, "reason": cache_reason})

            if cache_status == "reusable" and not cfg.overwrite_perception:
                manifest = cached_manifest
                progress.log(f"[CACHE] {sid}: reuse complete perception cache ({len(required)} frames)")
            else:
                if cache_status == "incompatible" and not cfg.overwrite_perception:
                    raise RuntimeError(
                        f"{sid}: existing perception cache is incompatible ({cache_reason}). "
                        "Use a new --output-dir or pass --overwrite-perception to deliberately rebuild it."
                    )
                if cfg.sst_checkpoint is None or cfg.rtmw_model is None:
                    raise RuntimeError(
                        f"{sid}: perception cache is {cache_status} ({cache_reason}). Provide --sst-checkpoint and --rtmw-model, "
                        "or point --output-dir to a compatible complete benchmark cache."
                    )
                if backend is None:
                    progress.log("[MODEL] loading SST + RTMW once for benchmark perception...")
                    backend = SSTRTMWStage2Backend(
                        sst_checkpoint=cfg.sst_checkpoint, rtmw_model=cfg.rtmw_model,
                        sst_device=cfg.sst_device, rtmw_device=cfg.rtmw_device,
                        player_threshold=cfg.player_threshold,
                        goalkeeper_threshold=cfg.goalkeeper_threshold,
                        referee_threshold=cfg.referee_threshold,
                        staff_threshold=cfg.staff_threshold,
                        ball_threshold=cfg.ball_threshold,
                        person_nms_iou=cfg.person_nms_iou,
                        ball_nms_iou=cfg.ball_nms_iou,
                        raw_human_score_floor=cfg.raw_human_score_floor,
                        consolidation_high_iou=cfg.consolidation_high_iou,
                        consolidation_low_iou=cfg.consolidation_low_iou,
                        consolidation_max_center_distance=cfg.consolidation_max_center_distance,
                        consolidation_min_area_similarity=cfg.consolidation_min_area_similarity,
                        consolidation_min_intersection_over_min=cfg.consolidation_min_intersection_over_min,
                        rescue_overlap_with_active_iou=cfg.rescue_overlap_with_active_iou,
                    )
                    progress.log("[MODEL] SST + RTMW loaded")
                frame_map = seq.materialize_frames(required, seq_out / "frames")
                force_overwrite = bool(cfg.overwrite_perception or cache_status == "incompatible")
                manifest = backend.process_window(
                    frame_map, cache_dir, overwrite=force_overwrite,
                    progress_callback=lambda ev, sid=sid: progress.perception(
                        sequence_id=sid, index=int(ev["sequence_index"]), total=int(ev["sequence_total"]),
                        frame_index=int(ev["frame_index"]), reused=bool(ev["reused"]),
                        humans=int(ev["humans"]), poses=int(ev["poses"]), balls=int(ev["balls"]),
                        elapsed=float(ev["elapsed_seconds"]),
                    ),
                )
                progress.log(f"[CACHE] {sid}: perception checkpoint complete ({len(required)} frames; previous={cache_status})")

            manifest_models = manifest.get("models") or {}
            sequence_hashes: Dict[str, str] = {}
            for key in ("sst", "rtmw"):
                value = str((manifest_models.get(key) or {}).get("sha256", ""))
                if value:
                    previous = observed_model_fingerprints.get(key)
                    if previous and previous.lower() != value.lower():
                        raise RuntimeError(f"{sid}: benchmark run would mix two {key} model fingerprints")
                    observed_model_fingerprints[key] = value
                    sequence_hashes[key] = value
            store.assert_or_set_model_fingerprints(sequence_hashes)
            conf = dict(manifest.get("configuration") or {})
            if conf and conf not in observed_perception_configurations:
                observed_perception_configurations.append(conf)

            manifest_index = _manifest_frame_map(manifest)
            for fi in required:
                payload_t = _load_json(manifest_index[fi])
                timing = payload_t.get("timing_seconds") or {}
                perception_timing["frames"] += 1
                for key in ("sst_detection","rtmw_pose","total_after_models_loaded"):
                    perception_timing[key] += float(timing.get(key,0.0) or 0.0)

            for wi, w in enumerate(windows, start=1):
                progress.log(
                    f"[WINDOW {wi}/{len(windows)}] {sid} t0={w.target_frame} range={w.window_start}..{w.window_end}"
                )
                ctx = _context(seq, w)
                det_ck = store.load_detection(sid, w.target_frame)
                tracking_ck = {exp: store.load_tracking(sid, w.target_frame, exp) for exp in cfg.experiments}
                need_primary_state = det_ck is None or tracking_ck.get("primary") is None

                with tempfile.TemporaryDirectory(prefix="stage2_bench_") as td_raw:
                    td = Path(td_raw)
                    states: Dict[str, Any] = {}
                    primary_manifest = _slice_manifest(manifest, w.window_start, w.window_end)
                    if need_primary_state:
                        tr0 = time.perf_counter()
                        states["primary"] = build_entity_track_state(
                            primary_manifest, ctx, td / "primary",
                            max_assignment_cost=cfg.max_assignment_cost, max_gap=cfg.max_gap,
                            use_temporal_rescue=cfg.use_temporal_rescue,
                            rescue_max_assignment_cost=cfg.rescue_max_assignment_cost,
                        )
                        primary_build_runtime = time.perf_counter() - tr0
                    else:
                        primary_build_runtime = 0.0

                    if det_ck is None:
                        state_primary = states["primary"]
                        gt_sel = [g.to_dict() for g in seq.gt(w.target_frame, roles=HUMAN_ROLES)]
                        sel_metrics = evaluate_selected_frame(state_primary.selected_frame_entities, gt_sel, iou_threshold=0.5)
                        payload = _selected_payload(manifest, w.target_frame)
                        raw_dup = gt_cross_class_duplicate_rate(payload.get("raw_detections",[]), gt_sel)
                        post_dup = gt_post_consolidation_duplicate_rate(payload.get("humans",[]), gt_sel)
                        p_ap, g_ap = _ap_records_from_payload(sid, w.target_frame, payload, gt_sel)
                        store.save_detection(sid, w.target_frame, {
                            "window": w.to_dict(),
                            "selected_frame_metrics": sel_metrics,
                            "raw_duplicate": raw_dup,
                            "post_duplicate": post_dup,
                            "ap_predictions": p_ap,
                            "ap_ground_truth": g_ap,
                            "gt_selected": gt_sel,
                            "pred_selected": list(state_primary.selected_frame_entities),
                            "image_path": payload.get("image_path"),
                        })
                        new_eval_units += 1
                        progress.eval_done(
                            f"{sid} t0={w.target_frame} detection",
                            metric_text=f"CandRecall={sel_metrics['CandidateRecall']:.3f} RefLeak={sel_metrics['RefereeLeakageRate']:.3f}",
                        )
                        maybe_debug_interrupt()
                    else:
                        progress.log(f"[RESUME] {sid} t0={w.target_frame} detection checkpoint already complete")

                    for exp in cfg.experiments:
                        ck = tracking_ck.get(exp)
                        if ck is not None:
                            progress.log(f"[RESUME] {sid} t0={w.target_frame} tracking:{exp} already complete")
                            continue

                        state_runtime = 0.0
                        if exp == "primary":
                            state = states.get("primary")
                            if state is None:
                                tr0 = time.perf_counter()
                                state = build_entity_track_state(
                                    primary_manifest, ctx, td / "primary_resume",
                                    max_assignment_cost=cfg.max_assignment_cost, max_gap=cfg.max_gap,
                                    use_temporal_rescue=cfg.use_temporal_rescue,
                                    rescue_max_assignment_cost=cfg.rescue_max_assignment_cost,
                                )
                                state_runtime = time.perf_counter() - tr0
                            else:
                                state_runtime = primary_build_runtime
                        elif exp == "no_pose":
                            no_pose_dir = td / "no_pose_frames"
                            no_pose_dir.mkdir(parents=True, exist_ok=True)
                            m = _slice_manifest(manifest, w.window_start, w.window_end, strip_pose=True, temp_dir=no_pose_dir)
                            tr0 = time.perf_counter()
                            state = build_entity_track_state(
                                m, ctx, td / "no_pose", max_assignment_cost=cfg.max_assignment_cost, max_gap=cfg.max_gap,
                                use_temporal_rescue=cfg.use_temporal_rescue, rescue_max_assignment_cost=cfg.rescue_max_assignment_cost,
                            )
                            state_runtime = time.perf_counter() - tr0
                        elif exp == "oracle_boxes_geom":
                            od = td / "oracle_frames"
                            od.mkdir(parents=True, exist_ok=True)
                            m = _oracle_manifest(seq, w, od)
                            tr0 = time.perf_counter()
                            state = build_entity_track_state(
                                m, ctx, td / "oracle", max_assignment_cost=cfg.max_assignment_cost, max_gap=cfg.max_gap,
                                use_temporal_rescue=False, rescue_max_assignment_cost=cfg.rescue_max_assignment_cost,
                            )
                            state_runtime = time.perf_counter() - tr0
                        elif exp == "oracle_boxes_pose":
                            if cfg.rtmw_model is None:
                                raise RuntimeError("oracle_boxes_pose requires --rtmw-model even when the primary perception cache is reused")
                            if oracle_pose_estimator is None:
                                progress.log("[MODEL] loading RTMW for oracle_boxes_pose...")
                                oracle_pose_estimator = RTMWOpenCVDNNPose(
                                    cfg.rtmw_model, input_width=288, input_height=384,
                                    bbox_padding=1.25, device=cfg.rtmw_device,
                                )
                            oracle_pose_cache = seq_out / "oracle_gt_pose" / "frame_perception"
                            m = _oracle_pose_manifest(
                                seq, w, oracle_pose_cache, oracle_pose_estimator,
                                expected_rtmw_sha256=observed_model_fingerprints.get("rtmw", expected_hashes.get("rtmw", "")),
                                overwrite=cfg.overwrite_perception,
                                progress_callback=lambda ev, sid=sid: progress.perception(
                                    sequence_id=f"{sid}:oracle", index=int(ev["sequence_index"]), total=int(ev["sequence_total"]),
                                    frame_index=int(ev["frame_index"]), reused=bool(ev["reused"]),
                                    humans=int(ev["humans"]), poses=int(ev["poses"]), balls=0,
                                    elapsed=float(ev["elapsed_seconds"]),
                                ),
                            )
                            tr0 = time.perf_counter()
                            state = build_entity_track_state(
                                m, ctx, td / "oracle_pose", max_assignment_cost=cfg.max_assignment_cost, max_gap=cfg.max_gap,
                                use_temporal_rescue=False, rescue_max_assignment_cost=cfg.rescue_max_assignment_cost,
                            )
                            state_runtime = time.perf_counter() - tr0
                        else:
                            raise RuntimeError(f"Unknown benchmark experiment {exp!r}")

                        eval_frames = seq.evaluation_frames(w.window_start, w.window_end)
                        gt_h = _gt_records(seq, w, candidate_only=False)
                        pr_h = _pred_records(state, candidate_only=False)
                        gt_c = _gt_records(seq, w, candidate_only=True)
                        pr_c = _pred_records(state, candidate_only=True)
                        ev0 = time.perf_counter()
                        human_result = evaluate_tracking_window(gt_h, pr_h, eval_frames, w.target_frame)
                        candidate_result = evaluate_tracking_window(gt_c, pr_c, eval_frames, w.target_frame)
                        metric_runtime = time.perf_counter() - ev0
                        store.save_tracking(sid, w.target_frame, exp, {
                            "window": w.to_dict(),
                            "human": human_result,
                            "candidate": candidate_result,
                            "state_build_runtime_seconds": state_runtime,
                            "metric_runtime_seconds": metric_runtime,
                            "tracking_runtime_seconds": state_runtime + metric_runtime,
                        })
                        new_eval_units += 1
                        progress.eval_done(
                            f"{sid} t0={w.target_frame} tracking:{exp}",
                            metric_text=_tracking_text(candidate_result),
                        )
                        maybe_debug_interrupt()

                # Persist a lightweight aggregate after every fully completed window. Even if the
                # final CSV/report has not been written yet, progress and partial metrics survive.
                if store.load_detection(sid, w.target_frame) is not None and all(
                    store.load_tracking(sid, w.target_frame, exp) is not None for exp in cfg.experiments
                ):
                    _write_partial_summary(store, out)
                    progress.log(f"[WINDOW CHECKPOINT] {sid} t0={w.target_frame} fully committed")

        summary = _aggregate_completed(
            cfg=cfg, out=out, ids=ids, plans=plans, store=store,
            observed_model_fingerprints=observed_model_fingerprints,
            observed_perception_configurations=observed_perception_configurations,
            dataset_versions=dataset_versions, cache_audit=cache_audit,
            perception_timing=perception_timing, started=started, require_complete=True,
        )
        store.mark_status("COMPLETE")
        # Rewrite summary once so checkpoint status inside it also says COMPLETE.
        summary["checkpoint"] = store.status()
        write_json(out / "benchmark_summary.json", summary)
        progress.log(
            f"[DONE] CandidateRecall={summary['detection']['CandidateRecall']:.3f} "
            f"HOTA={summary['tracking_candidate']['HOTA']:.3f} "
            f"AssA={summary['tracking_candidate']['AssA']:.3f} "
            f"IDF1={summary['tracking_candidate']['IDF1']:.3f} "
            f"gate={'PASS' if summary['acceptance_gate']['passed'] else 'FAIL'}"
        )
        return summary
    except KeyboardInterrupt as exc:
        store.mark_status("INTERRUPTED", error=str(exc) or "KeyboardInterrupt")
        _write_partial_summary(store, out)
        progress.log(
            f"[INTERRUPTED] progress saved at {store.state_path}. Re-run the same command to resume."
        )
        raise
    except Exception as exc:
        store.mark_status("FAILED", error=f"{type(exc).__name__}: {exc}")
        _write_partial_summary(store, out)
        progress.log(
            f"[FAILED] {type(exc).__name__}: {exc} | completed work remains checkpointed; re-run after fixing the cause."
        )
        raise
