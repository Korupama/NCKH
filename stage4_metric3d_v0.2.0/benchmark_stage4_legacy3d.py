from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

from stage4_metric3d.camera import CameraStateLite
from stage4_metric3d.comparison import (
    compare_pipeline_outputs,
    convert_external_prediction,
    evaluate_system_manifest,
)
from stage4_metric3d.evaluation import (
    goalward_anchor_longitudinal_error,
    longitudinal_errors,
    mpjpe,
    pairwise_longitudinal_ordering_accuracy,
    procrustes_mpjpe,
    root_aligned_mpjpe,
    uncertainty_interval_coverage,
)
from stage4_metric3d.initializers.cache import write_jsonl
from stage4_metric3d.processor import run_stage4
from stage4_metric3d.schemas import Stage4Config
from stage4_metric3d.wholebody import LEGAL_GEOMETRY_CANDIDATE_23, POSE23_NAMES, WHOLEBODY_NAMES


def _pose_from_prediction_state(state: Mapping[str, object]) -> Dict[Tuple[int, str], np.ndarray]:
    out: Dict[Tuple[int, str], np.ndarray] = {}
    for tr in state.get("tracks", []):
        tid = str(tr.get("track_id"))
        for obs in tr.get("observations", []):
            frame = int(obs["frame_index"])
            arr = np.full((23, 3), np.nan, dtype=np.float64)
            for j in obs.get("metric_pose23", []):
                idx = int(j["index"])
                xyz = j.get("xyz_world_m")
                if xyz is not None:
                    arr[idx] = np.asarray(xyz, dtype=np.float64)
            out[(frame, tid)] = arr
    return out


def _pose_from_gt(gt: Mapping[str, object]) -> Dict[Tuple[int, str], np.ndarray]:
    out: Dict[Tuple[int, str], np.ndarray] = {}
    for rec in gt.get("poses", []):
        frame = int(rec["frame_index"])
        tid = str(rec["track_id"])
        if "xyz23_world_m" in rec:
            arr = np.asarray(rec["xyz23_world_m"], dtype=np.float64)
        else:
            joints = rec.get("joints", {})
            arr = np.full((23, 3), np.nan, dtype=np.float64)
            for j, name in enumerate(POSE23_NAMES):
                if name in joints:
                    arr[j] = np.asarray(joints[name], dtype=np.float64)
        if arr.shape != (23, 3):
            raise ValueError(f"GT {(frame, tid)} expected (23,3), got {arr.shape}")
        out[(frame, tid)] = arr
    return out


def _uncertainty_from_prediction_state(
    state: Mapping[str, object],
) -> Tuple[Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]], Dict[Tuple[int, str], Mapping[str, object]]]:
    joint_intervals: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
    extrema: Dict[Tuple[int, str], Mapping[str, object]] = {}
    selected_frame = int((state.get("replay_context") or {}).get("selected_frame", -1))
    for track in state.get("tracks", []):
        tid = str(track.get("track_id"))
        for observation in track.get("observations", []):
            frame = int(observation.get("frame_index", -1))
            lo = np.full((23, 3), np.nan, dtype=np.float64)
            hi = np.full((23, 3), np.nan, dtype=np.float64)
            for joint in observation.get("metric_pose23", []):
                index = int(joint.get("index", -1))
                q025 = joint.get("q025_xyz_m")
                q975 = joint.get("q975_xyz_m")
                if 0 <= index < 23 and q025 is not None and q975 is not None:
                    lo[index] = np.asarray(q025, dtype=np.float64)
                    hi[index] = np.asarray(q975, dtype=np.float64)
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                joint_intervals[(frame, tid)] = (lo, hi)
        uncertainty = track.get("selected_frame_uncertainty") or {}
        legal_extrema = ((uncertainty.get("longitudinal") or {}).get("legal_extrema_x") or {})
        if legal_extrema:
            extrema[(selected_frame, tid)] = legal_extrema
    return joint_intervals, extrema


def _evaluate_uncertainty(
    pred_state: Mapping[str, object],
    gt: Mapping[Tuple[int, str], np.ndarray],
) -> dict:
    intervals, extrema = _uncertainty_from_prediction_state(pred_state)
    matched = sorted(set(gt).intersection(intervals))
    if not matched:
        return {
            "status": "NO_INTERVALS",
            "calibrated_claimed": False,
            "matched_pose_records": 0,
            "interpretation": "No selected-frame uncertainty intervals were available for GT comparison.",
        }

    gt_rows, lo_rows, hi_rows = [], [], []
    for key in matched:
        lo, hi = intervals[key]
        gt_rows.append(gt[key])
        lo_rows.append(lo)
        hi_rows.append(hi)
    gt_all = np.stack(gt_rows)
    lo_all = np.stack(lo_rows)
    hi_all = np.stack(hi_rows)
    coverage = uncertainty_interval_coverage(gt_all, lo_all, hi_all)
    width = hi_all - lo_all
    widths = {}
    for axis, name in enumerate(("x", "y", "z")):
        values = width[..., axis]
        values = values[np.isfinite(values)]
        widths[f"mean_width_{name}_m"] = None if not values.size else float(np.mean(values))
        widths[f"median_width_{name}_m"] = None if not values.size else float(np.median(values))

    legal = np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool)
    legal_gt = gt_all[:, legal, 0]
    legal_lo = lo_all[:, legal, 0]
    legal_hi = hi_all[:, legal, 0]
    legal_mask = np.isfinite(legal_gt) & np.isfinite(legal_lo) & np.isfinite(legal_hi)
    legal_x_coverage = None
    if np.any(legal_mask):
        legal_x_coverage = float(np.mean((legal_gt[legal_mask] >= legal_lo[legal_mask]) & (legal_gt[legal_mask] <= legal_hi[legal_mask])))

    extrema_coverage = {}
    for direction in ("min_legal_x", "max_legal_x"):
        covered = []
        widths_values = []
        for key in matched:
            summary = (extrema.get(key) or {}).get(direction) or {}
            lo = summary.get("q025_m")
            hi = summary.get("q975_m")
            if lo is None or hi is None:
                continue
            legal_gt_x = gt[key][legal, 0]
            legal_gt_x = legal_gt_x[np.isfinite(legal_gt_x)]
            if not legal_gt_x.size:
                continue
            value = float(np.min(legal_gt_x) if direction == "min_legal_x" else np.max(legal_gt_x))
            covered.append(float(lo) <= value <= float(hi))
            widths_values.append(float(hi) - float(lo))
        extrema_coverage[direction] = {
            "coverage": None if not covered else float(np.mean(covered)),
            "count": len(covered),
            "mean_width_m": None if not widths_values else float(np.mean(widths_values)),
        }

    analysis = pred_state.get("uncertainty_analysis") or {}
    return {
        "status": "EVALUATED",
        "calibrated_claimed": bool(analysis.get("calibrated", False)),
        "matched_pose_records": len(matched),
        "joint_axis_coverage": coverage,
        "joint_axis_width": widths,
        "legal_anchor_x_coverage": legal_x_coverage,
        "legal_extrema_x_coverage": extrema_coverage,
        "interpretation": "Empirical coverage diagnostics; camera-fixed pixel sensitivity is not calibrated probability.",
    }


def evaluate_canonical(pred_path: str | Path, gt_path: str | Path, attack_sign: int = 1) -> dict:
    pred_state = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    gt_state = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    pred = _pose_from_prediction_state(pred_state)
    gt = _pose_from_gt(gt_state)
    keys = sorted(set(pred) & set(gt))
    if not keys:
        raise ValueError("No matching (frame_index, track_id) records between prediction and GT")

    p_all, g_all = [], []
    frame_records = []
    longitudinal = []
    gale = []
    by_frame_pairs: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
    for key in keys:
        p, g = pred[key], gt[key]
        valid = np.isfinite(p).all(axis=1) & np.isfinite(g).all(axis=1)
        if valid.any():
            p_all.append(p[valid])
            g_all.append(g[valid])
        le = longitudinal_errors(p, g)
        legal_le = le[np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool)]
        longitudinal.extend(legal_le[np.isfinite(legal_le)].tolist())
        try:
            gale.append(goalward_anchor_longitudinal_error(
                p, g, attack_sign=attack_sign, legal_mask=LEGAL_GEOMETRY_CANDIDATE_23
            ))
        except Exception:
            pass
        frame_records.append({
            "frame_index": key[0],
            "track_id": key[1],
            "mpjpe_m": mpjpe(p, g),
            "root_aligned_mpjpe_m": root_aligned_mpjpe(p, g),
            "pa_mpjpe_m": procrustes_mpjpe(p, g),
            "longitudinal_mae_m": float(np.nanmean(legal_le)) if np.isfinite(legal_le).any() else None,
        })
        by_frame_pairs.setdefault(key[0], []).append((key[1], p, g))

    pcat = np.concatenate(p_all, axis=0)
    gcat = np.concatenate(g_all, axis=0)

    ordering = {}
    for threshold in (2.0, 1.0, 0.5, 0.25):
        correct = total = 0
        for frame, rows in by_frame_pairs.items():
            pred_x = []
            gt_x = []
            for _, p, g in rows:
                valid = LEGAL_GEOMETRY_CANDIDATE_23 & np.isfinite(p).all(axis=1) & np.isfinite(g).all(axis=1)
                if not valid.any():
                    continue
                pred_x.append(float(attack_sign * np.nanmax(attack_sign * p[valid, 0])))
                gt_x.append(float(attack_sign * np.nanmax(attack_sign * g[valid, 0])))
            if len(pred_x) >= 2:
                pp = np.asarray(pred_x, dtype=float)
                gg = np.asarray(gt_x, dtype=float)
                for a in range(len(pp)):
                    for b in range(a + 1, len(pp)):
                        sep = abs(float(gg[a] - gg[b]))
                        if sep >= threshold or sep < 1e-9:
                            continue
                        total += 1
                        correct += int(np.sign(pp[a] - pp[b]) == np.sign(gg[a] - gg[b]))
        ordering[f"gt_separation_le_{threshold:g}m"] = {
            "accuracy": None if total == 0 else correct / total,
            "correct": correct,
            "total": total,
        }

    result = {
        "schema_version": "stage4-benchmark-summary-1.0",
        "matched_pose_records": len(keys),
        "valid_joint_observations": int(len(pcat)),
        "metrics": {
            "GlobalMPJPE_m": mpjpe(pcat, gcat),
            "GlobalMPJPE_mm": 1000.0 * mpjpe(pcat, gcat),
            "LongitudinalMAE_m": float(np.mean(longitudinal)) if longitudinal else None,
            "LongitudinalMedian_m": float(np.median(longitudinal)) if longitudinal else None,
            "LongitudinalP90_m": float(np.percentile(longitudinal, 90)) if longitudinal else None,
            "GALE_mean_m": float(np.mean(gale)) if gale else None,
            "GALE_median_m": float(np.median(gale)) if gale else None,
            "PairwiseLongitudinalOrdering": ordering,
        },
        "per_pose": frame_records,
        "notes": {
            "pa_mpjpe": "reported per-pose because concatenated Procrustes alignment is not meaningful",
            "gale": "Goalward Anchor Longitudinal Error over project-scope legal anchor candidates; Stage 5 builds full legal-body surfaces",
            "attack_sign": attack_sign,
        },
    }
    result["uncertainty"] = _evaluate_uncertainty(pred_state, gt)
    pa = [x["pa_mpjpe_m"] for x in frame_records if x["pa_mpjpe_m"] is not None and math.isfinite(x["pa_mpjpe_m"])]
    root = [x["root_aligned_mpjpe_m"] for x in frame_records if x["root_aligned_mpjpe_m"] is not None and math.isfinite(x["root_aligned_mpjpe_m"])]
    result["metrics"]["PA_MPJPE_mean_mm"] = None if not pa else 1000.0 * float(np.mean(pa))
    result["metrics"]["RootAlignedMPJPE_mean_mm"] = None if not root else 1000.0 * float(np.mean(root))
    return result


REQUIRED_LANES = (
    "GT2D_GTCAMERA",
    "STAGE3_GTCAMERA",
    "GT2D_STAGE1CAMERA",
    "STAGE3_STAGE1CAMERA",
)


def evaluate_lane_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    gt_value = manifest.get("gt")
    if not gt_value:
        raise ValueError("Lane manifest requires a canonical GT path in 'gt'")
    gt_path = Path(gt_value)
    if not gt_path.is_absolute():
        gt_path = (base_dir / gt_path).resolve()
    lanes = manifest.get("lanes") or {}
    missing = [name for name in REQUIRED_LANES if name not in lanes]
    if missing:
        raise ValueError("Lane manifest is missing required lanes: " + ", ".join(missing))
    attack_sign = int(manifest.get("attack_sign", 1))
    results = {}
    for name in REQUIRED_LANES:
        spec = lanes[name]
        pred_value = spec.get("pred") if isinstance(spec, Mapping) else spec
        pred_path = Path(pred_value)
        if not pred_path.is_absolute():
            pred_path = (base_dir / pred_path).resolve()
        results[name] = evaluate_canonical(pred_path, gt_path, attack_sign)

    primary = ("LongitudinalMAE_m", "LongitudinalP90_m", "GALE_mean_m")
    attribution = {}
    oracle = results["GT2D_GTCAMERA"]["metrics"]
    for lane_name in REQUIRED_LANES[1:]:
        lane_metrics = results[lane_name]["metrics"]
        attribution[lane_name] = {
            metric: (
                None if oracle.get(metric) is None or lane_metrics.get(metric) is None
                else float(lane_metrics[metric]) - float(oracle[metric])
            )
            for metric in primary
        }
    return {
        "schema_version": "stage4-benchmark-lane-summary-1.0",
        "manifest": str(manifest_path),
        "gt": str(gt_path),
        "attack_sign": attack_sign,
        "lanes": results,
        "delta_from_gt2d_gtcamera_oracle": attribution,
        "interpretation": "Lane deltas diagnose 2D-pose and camera contributions; they are not causal guarantees.",
    }


def _look_at_camera(frame: int, width: int = 1280, height: int = 720) -> dict:
    C = np.asarray([0.0, -42.0, 15.0], dtype=np.float64)
    target = np.asarray([4.0, 4.0, 0.9], dtype=np.float64)
    forward = target - C
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    R = np.stack([right, down, forward], axis=0)
    fx = fy = 1600.0
    K = np.asarray([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])
    return {
        "schema_version": "1.2",
        "frame_index": frame,
        "status": "VALID",
        "image": {"width": width, "height": height},
        "intrinsics": {"K": K.tolist()},
        "extrinsics": {"R_world_to_camera": R.tolist(), "camera_center_world_m": C.tolist()},
        "distortion": {"radial": [0, 0, 0, 0, 0, 0], "tangential": [0, 0], "thin_prism": [0, 0, 0, 0]},
        "pitch": {"length_m": 105.0, "width_m": 68.0, "origin": "center"},
    }


def _synthetic_skeleton(frame: int) -> np.ndarray:
    # A deliberately modest, non-degenerate standing/running pose in pitch world.
    dx = 0.035 * (frame - 86)
    x0, y0 = 4.0 + dx, 4.0
    p = np.zeros((23, 3), dtype=np.float64)
    p[0] = [x0, y0, 1.76]
    p[1] = [x0 - .025, y0 - .015, 1.78]
    p[2] = [x0 + .025, y0 - .015, 1.78]
    p[3] = [x0 - .075, y0, 1.75]
    p[4] = [x0 + .075, y0, 1.75]
    p[5] = [x0 - .20, y0, 1.48]
    p[6] = [x0 + .20, y0, 1.48]
    p[7] = [x0 - .33, y0 + .03, 1.17]
    p[8] = [x0 + .33, y0 - .03, 1.19]
    p[9] = [x0 - .39, y0 + .05, .92]
    p[10] = [x0 + .39, y0 - .05, .94]
    p[11] = [x0 - .13, y0, .94]
    p[12] = [x0 + .13, y0, .94]
    p[13] = [x0 - .14, y0 + .04, .50]
    p[14] = [x0 + .14, y0 - .04, .50]
    p[15] = [x0 - .14, y0 + .02, .075]
    p[16] = [x0 + .14, y0 - .02, .075]
    p[17] = [x0 - .14, y0 + .20, .015]
    p[18] = [x0 - .10, y0 + .19, .015]
    p[19] = [x0 - .14, y0 - .05, .015]
    p[20] = [x0 + .14, y0 + .16, .015]
    p[21] = [x0 + .18, y0 + .15, .015]
    p[22] = [x0 + .14, y0 - .07, .015]
    return p


def build_synthetic_case(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    camera_dir = root / "cameras"
    camera_dir.mkdir(exist_ok=True)
    frames = list(range(83, 90))
    cameras = {}
    for frame in frames:
        data = _look_at_camera(frame)
        path = camera_dir / f"camera_state_{frame:06d}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        cameras[frame] = CameraStateLite.from_dict(data)

    track_obs = []
    init_records = []
    gt_records = []
    for frame in frames:
        xyz = _synthetic_skeleton(frame)
        uv = cameras[frame].project_world(xyz, distort=True)
        x1, y1 = np.min(uv, axis=0) - np.asarray([12, 10])
        x2, y2 = np.max(uv, axis=0) + np.asarray([12, 10])
        kps = []
        for i, name in enumerate(WHOLEBODY_NAMES):
            if i < 23:
                x, y = uv[i]
                score, state = 3.0, "VALID"
            else:
                x, y = None, None
                score, state = None, "MISSING"
            kps.append({"index": i, "name": name, "x": x, "y": y, "raw_model_score": score, "state": state, "source": "SYNTHETIC", "temporal_estimate_xy": None})
        track_obs.append({
            "frame_index": frame,
            "source_bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "keypoints_133": kps,
            "pose_status": "VALID",
        })
        depth = np.asarray(cameras[frame].camera_depth(xyz), dtype=float)
        raw = np.zeros((133, 3), dtype=float)
        raw[:23, :2] = uv
        raw[:23, 2] = depth
        root_depth = float(np.mean(depth[[11, 12]]))
        rel = np.zeros(133, dtype=float)
        rel[:23] = depth - root_depth
        init_records.append({
            "record_type": "observation",
            "track_id": "track_001",
            "frame_index": frame,
            "backend": "SYNTHETIC_RELATIVE3D",
            "raw_keypoints_133": raw.tolist(),
            "keypoint_scores_133": [3.0] * 133,
            "relative_depth_133": rel.tolist(),
            "global_position_known": False,
        })
        gt_records.append({"frame_index": frame, "track_id": "track_001", "xyz23_world_m": xyz.tolist()})

    stage3 = {
        "schema_version": "tracked-pose-2d-state-1.0",
        "stage3_version": "synthetic-stage3",
        "coordinate_space": "RAW_DISTORTED_PIXEL",
        "keypoint_schema": {"name": "COCO_WHOLEBODY_133", "count": 133, "names": list(WHOLEBODY_NAMES)},
        "replay_context": {"selected_frame": 86, "window_start": 83, "window_end": 89, "image_width": 1280, "image_height": 720, "fps": 30.0},
        "tracks": [{"track_id": "track_001", "upstream_role": "player", "upstream_identity_confidence": 1.0, "candidate_for_stage3": True, "observations": track_obs}],
    }
    stage3_path = root / "tracked_pose_2d_state.json"
    stage3_path.write_text(json.dumps(stage3, indent=2), encoding="utf-8")
    init_path = root / "initializer.jsonl"
    write_jsonl(init_path, {"schema_version": "stage4-relative3d-initializer-cache-1.0", "backend": "SYNTHETIC_RELATIVE3D"}, init_records)
    gt = {"schema_version": "metric-pose23-gt-1.0", "poses": gt_records}
    gt_path = root / "gt_metric_pose23.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return {"stage3": stage3_path, "camera_dir": camera_dir, "initializer": init_path, "gt": gt_path}


def synthetic_smoke(output_dir: str | Path) -> dict:
    out = Path(output_dir).expanduser().resolve()
    case = build_synthetic_case(out / "synthetic_inputs")
    cfg = Stage4Config(
        window_radius_frames=3,
        min_temporal_frames=7,
        uncertainty_samples=0,
        require_initializer=True,
        optimizer_max_nfev=500,
    )
    state = run_stage4(
        stage3_state=case["stage3"],
        camera_dir=case["camera_dir"],
        initializer_cache=case["initializer"],
        output_dir=out / "stage4_output",
        config=cfg,
    )
    summary = evaluate_canonical(out / "stage4_output" / "metric_pose_3d_state.json", case["gt"])
    summary_path = out / "synthetic_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"output": str(summary_path), "summary": summary, "stage4_metrics": state["metrics"]}


def main() -> int:
    p = argparse.ArgumentParser(description="Stage-4 metric-3D benchmark utilities")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate-canonical", help="Evaluate MetricPose3DState against canonical XYZ23 GT JSON")
    e.add_argument("--pred", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--attack-sign", type=int, choices=[-1, 1], default=1)
    e.add_argument("--output", default=None)
    s = sub.add_parser("synthetic-smoke", help="Run a deterministic calibrated-camera end-to-end smoke benchmark")
    s.add_argument("--output-dir", default="benchmark_results/synthetic_smoke")
    lanes = sub.add_parser("evaluate-lanes", help="Evaluate the required GT2D/Stage3 x GTCamera/Stage1Camera matrix")
    lanes.add_argument("--manifest", required=True)
    lanes.add_argument("--output", default=None)
    systems = sub.add_parser(
        "evaluate-systems",
        help="Fair paired evaluation of current, legacy and third-party systems",
    )
    systems.add_argument("--manifest", required=True)
    systems.add_argument("--output", required=True)
    regression = sub.add_parser(
        "compare-pipelines",
        help="Compare Stage-4 point estimates/statuses with an older pipeline output without requiring GT",
    )
    regression.add_argument("--candidate", required=True)
    regression.add_argument("--baseline", required=True)
    regression.add_argument("--max-abs-delta-m", type=float, default=None)
    regression.add_argument("--require-no-finite-loss", action="store_true")
    regression.add_argument("--output", required=True)
    external = sub.add_parser(
        "convert-external",
        help="Convert an external model export into the coordinate-scope-aware canonical prediction schema",
    )
    external.add_argument("--input", required=True)
    external.add_argument("--adapter", required=True)
    external.add_argument("--output", required=True)
    args = p.parse_args()

    if args.command == "evaluate-canonical":
        result = evaluate_canonical(args.pred, args.gt, args.attack_sign)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "synthetic-smoke":
        result = synthetic_smoke(args.output_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "evaluate-lanes":
        result = evaluate_lane_manifest(args.manifest)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "evaluate-systems":
        result = evaluate_system_manifest(args.manifest)
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "compare-pipelines":
        result = compare_pipeline_outputs(
            args.candidate,
            args.baseline,
            max_abs_delta_m=args.max_abs_delta_m,
            require_no_finite_loss=args.require_no_finite_loss,
        )
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        result = convert_external_prediction(args.input, args.adapter, args.output)
        print(json.dumps({
            "output": str(Path(args.output).expanduser().resolve()),
            "pose_records": len(result["poses"]),
            "coordinate_scope": result["coordinate_scope"],
        }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
