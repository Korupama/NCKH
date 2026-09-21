from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import csv
import importlib
import json
import math
import sys
import time
import zipfile

import cv2
import numpy as np

from ..candidates import _compact
from ..contracts import CameraStatus
from ..evidence import build_keypoint_correspondences, summarize_keypoint_coverage
from ..pnlcalib_adapter import PnLCalibAdapter
from ..quality_gate import CameraQualityGate
from ..policies import Vertical3DPolicyConfig


@dataclass
class SoccerNetBenchmarkConfig:
    split: str = "valid"
    limit: Optional[int] = None
    selection: str = "uniform"
    kp_threshold: float = 0.0712
    line_threshold: float = 0.2571
    max_reproj_err_px: float = 38.0
    model_width: int = 960
    model_height: int = 540
    refine: bool = False
    refine_lines: bool = True
    candidate_modes: Tuple[str, ...] = ("full", "ground_plane", "main")
    ransac_values: Tuple[int, ...] = (0, 5, 10, 15, 25, 50)
    preferred_full_threshold_px: float = 5.0


@dataclass
class DatasetSplitIntegrity:
    split: str
    directory: str
    num_jpg: int
    num_json: int
    num_paired: int
    missing_json: List[str]
    orphan_json: List[str]
    ready: bool


def ensure_split_extracted(root: Union[str, Path], split: str) -> Path:
    """Return an extracted SoccerNet split directory, extracting split.zip if needed."""
    root = Path(root)
    split_dir = root / split
    if split_dir.exists() and any(split_dir.glob("*.jpg")):
        return split_dir
    archive = root / f"{split}.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"Neither extracted split nor archive exists: {split_dir} / {archive}"
        )
    split_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(split_dir)
    # Some archives may contain an extra split/ directory.
    nested = split_dir / split
    if nested.exists() and any(nested.glob("*.jpg")):
        return nested
    return split_dir


def inspect_dataset(root: Union[str, Path], splits: Sequence[str] = ("train", "valid", "test"), *, auto_extract: bool = False) -> Dict[str, Any]:
    root = Path(root)
    results: Dict[str, Any] = {"root": str(root), "splits": {}}
    for split in splits:
        split_dir = root / split
        if auto_extract and not (split_dir.exists() and any(split_dir.glob("*.jpg"))):
            try:
                split_dir = ensure_split_extracted(root, split)
            except FileNotFoundError:
                pass
        jpgs = sorted(split_dir.glob("*.jpg")) if split_dir.exists() else []
        jsons = sorted(split_dir.glob("*.json")) if split_dir.exists() else []
        jpg_stems = {p.stem for p in jpgs}
        ann_jsons = [p for p in jsons if p.stem not in {"match_info_cam_gt", "match_info"}]
        json_stems = {p.stem for p in ann_jsons}
        missing = sorted(jpg_stems - json_stems)
        orphan = sorted(json_stems - jpg_stems)
        item = DatasetSplitIntegrity(
            split=split,
            directory=str(split_dir),
            num_jpg=len(jpgs),
            num_json=len(ann_jsons),
            num_paired=len(jpg_stems & json_stems),
            missing_json=missing[:20],
            orphan_json=orphan[:20],
            ready=bool(jpgs) and not missing,
        )
        results["splits"][split] = asdict(item)
    results["ready"] = all(v["ready"] for v in results["splits"].values())
    return results


def select_image_paths(split_dir: Union[str, Path], limit: Optional[int] = None, strategy: str = "uniform") -> List[Path]:
    paths = sorted(Path(split_dir).glob("*.jpg"))
    if limit is None or limit <= 0 or limit >= len(paths):
        return paths
    if strategy == "first":
        return paths[:limit]
    if strategy != "uniform":
        raise ValueError(f"Unsupported selection strategy: {strategy}")
    idx = np.linspace(0, len(paths) - 1, num=int(limit), dtype=int)
    # linspace may duplicate indices for pathological inputs; preserve order while deduplicating.
    seen = set()
    out = []
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            out.append(paths[i])
    return out


def _finite(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return None if arr.size == 0 else float(np.quantile(arr, q))


def _mean(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return None if arr.size == 0 else float(arr.mean())


def _counter(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in records:
        value = str(row.get(key, "None"))
        out[value] = out.get(value, 0) + 1
    return out


def summarize_records(records: Sequence[Dict[str, Any]], config: SoccerNetBenchmarkConfig) -> Dict[str, Any]:
    n = len(records)
    solved = sum(bool(r.get("solved")) for r in records)
    official_pred = sum(bool(r.get("official_prediction_written")) for r in records)
    rep = [_finite(r.get("rep_err_px")) for r in records]
    runtime = [_finite(r.get("runtime_ms")) for r in records]
    official_acc = [_finite(r.get("official_frame_accuracy")) for r in records]
    official_prec = [_finite(r.get("official_frame_precision")) for r in records]
    official_rec = [_finite(r.get("official_frame_recall")) for r in records]
    mean_acc = _mean(official_acc)
    completeness = float(official_pred / n) if n else 0.0
    return {
        "report_schema_version": "1.1",
        "config": asdict(config),
        "num_frames": n,
        "num_solved": solved,
        "solver_coverage": float(solved / n) if n else 0.0,
        "num_official_predictions": official_pred,
        "official_completeness": completeness,
        "camera_status_counts": _counter(records, "camera_status"),
        "ground_status_counts": _counter(records, "ground_status"),
        "vertical_3d_status_counts": _counter(records, "vertical_3d_status"),
        "selected_mode_counts": _counter(records, "mode"),
        "selected_ransac_counts": _counter(records, "use_ransac"),
        "offside_3d_ready_count": sum(bool(r.get("offside_3d_ready")) for r in records),
        "offside_3d_ready_rate": float(sum(bool(r.get("offside_3d_ready")) for r in records) / n) if n else 0.0,
        "reprojection_error_px": {
            "mean": _mean(rep),
            "median": _quantile(rep, 0.5),
            "p95": _quantile(rep, 0.95),
        },
        "runtime_ms": {
            "mean": _mean(runtime),
            "median": _quantile(runtime, 0.5),
            "p95": _quantile(runtime, 0.95),
        },
        "official_like_metrics": {
            # These intentionally preserve the naming/threshold behavior of the bundled
            # SoccerNet evaluator. Do not relabel this quantity as JaC@5.
            "meanAccuracies": mean_acc,
            "meanPrecision": _mean(official_prec),
            "meanRecall": _mean(official_rec),
            "completeness": completeness,
            "finalScore": None if mean_acc is None else float(completeness * mean_acc),
            "internal_distance_threshold_px": 20,
        },
        "evidence_quantiles": {
            "keypoint_count_median": _quantile([_finite(r.get("num_keypoints")) for r in records], 0.5),
            "line_count_median": _quantile([_finite(r.get("num_lines")) for r in records], 0.5),
            "hull_ratio_p10": _quantile([_finite(r.get("keypoint_hull_ratio")) for r in records], 0.10),
            "x_span_ratio_p10": _quantile([_finite(r.get("keypoint_x_span_ratio")) for r in records], 0.10),
            "y_span_ratio_p10": _quantile([_finite(r.get("keypoint_y_span_ratio")) for r in records], 0.10),
        },
    }



def review_vertical_policy_csv(
    csv_path: Union[str, Path],
    policy: Optional[Vertical3DPolicyConfig] = None,
) -> Dict[str, Any]:
    """Reclassify an existing per-frame VALID report with the frozen v12 vertical policy.

    This is intentionally inference-free: it lets a v11 ``valid_full_frames.csv``
    audit the v12 policy without rerunning the neural networks or candidate sweep.
    Rows already marked INVALID remain ineligible because the CSV does not retain
    every underlying physical-plausibility reason needed to reconstruct that gate.
    """
    vp = policy or Vertical3DPolicyConfig()
    rows: List[Dict[str, str]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def num(row: Dict[str, str], key: str) -> Optional[float]:
        return _finite(row.get(key))

    def old_ready(row: Dict[str, str]) -> bool:
        return str(row.get("offside_3d_ready", "")).strip().lower() in {"true", "1", "yes"}

    def passes(row: Dict[str, str]) -> bool:
        if str(row.get("camera_status", "")).upper() == CameraStatus.INVALID.value:
            return False
        mode = row.get("mode")
        ransac = num(row, "use_ransac")
        rep = num(row, "rep_err_px")
        kp = num(row, "num_keypoints")
        q = num(row, "keypoint_quadrants")
        hull = num(row, "keypoint_hull_ratio")
        xs = num(row, "keypoint_x_span_ratio")
        ys = num(row, "keypoint_y_span_ratio")
        values = (rep, kp, q, hull, xs, ys)
        if any(v is None for v in values):
            return False
        if vp.require_full_no_ransac and not (mode == "full" and int(ransac) == 0):
            return False
        return bool(
            rep <= vp.max_rep_err_px
            and kp >= vp.min_keypoints
            and q >= vp.min_image_quadrants
            and hull >= vp.min_image_hull_ratio
            and xs >= vp.min_image_x_span_ratio
            and ys >= vp.min_image_y_span_ratio
        )

    selected = [r for r in rows if passes(r)]
    old_selected = [r for r in rows if old_ready(r)]
    acc = [num(r, "official_frame_accuracy") for r in selected]
    acc_arr = np.asarray([x for x in acc if x is not None], dtype=np.float64)
    n = len(rows)
    count = len(selected)
    result = {
        "policy": vp.to_dict(),
        "num_frames": n,
        "old_ready_count": len(old_selected),
        "old_ready_rate": float(len(old_selected) / n) if n else 0.0,
        "v12_ready_count": count,
        "v12_ready_rate": float(count / n) if n else 0.0,
        "demoted_from_old_ready": sum(old_ready(r) and not passes(r) for r in rows),
        "promoted_from_old_not_ready": sum((not old_ready(r)) and passes(r) for r in rows),
        "official_frame_accuracy_on_v12_ready": {
            "mean": None if acc_arr.size == 0 else float(acc_arr.mean()),
            "p10": None if acc_arr.size == 0 else float(np.quantile(acc_arr, 0.10)),
            "fraction_ge_0_9": None if acc_arr.size == 0 else float(np.mean(acc_arr >= 0.90)),
        },
    }
    result["matches_frozen_validation_reference"] = bool(
        n == vp.threshold_source_frames
        and count == vp.validation_ready_count
        and abs(result["v12_ready_rate"] - vp.validation_ready_rate) < 1e-12
    )
    return result

def write_reports(records: Sequence[Dict[str, Any]], summary: Dict[str, Any], output_dir: Union[str, Path], prefix: str) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_frames.csv"
    json_path = output_dir / f"{prefix}_summary.json"
    jsonl_path = output_dir / f"{prefix}_frames.jsonl"

    keys: List[str] = []
    for row in records:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: _csv_value(row.get(k)) for k in keys})
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"csv": str(csv_path), "summary_json": str(json_path), "frames_jsonl": str(jsonl_path)}


def _csv_value(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v


class SoccerNetCalibrationBenchmark:
    """In-process SoccerNet Calibration-2023 benchmark using the official PnLCalib backend.

    It mirrors ``scripts/inference_sn.py`` model preprocessing and candidate policy,
    while additionally recording Stage-1 quality/capability diagnostics per frame.
    """

    def __init__(
        self,
        *,
        pnl_dir: Union[str, Path],
        weights_kp: Union[str, Path],
        weights_line: Union[str, Path],
        device: str = "cpu",
        config: Optional[SoccerNetBenchmarkConfig] = None,
        quality_gate: Optional[CameraQualityGate] = None,
    ):
        self.pnl_dir = Path(pnl_dir).resolve()
        self.weights_kp = Path(weights_kp).resolve()
        self.weights_line = Path(weights_line).resolve()
        self.device_name = str(device)
        self.cfg = config or SoccerNetBenchmarkConfig()
        self.quality_gate = quality_gate or CameraQualityGate()
        self._loaded = False

    def _load_backend(self) -> None:
        if self._loaded:
            return
        for p in (self.pnl_dir, self.pnl_dir / "sn_calibration", self.pnl_dir / "sn_calibration" / "src"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        import torch
        import yaml
        import torchvision.transforms as T
        import torchvision.transforms.functional as tvf
        from PIL import Image
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import (
            FramebyFrameCalib,
            keypoint_aux_world_coords_2D,
            keypoint_world_coords_2D,
        )
        from utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        self.torch = torch
        self.T = T
        self.tvf = tvf
        self.Image = Image
        self.FramebyFrameCalib = FramebyFrameCalib
        self.keypoint_world_coords_2D = keypoint_world_coords_2D
        self.keypoint_aux_world_coords_2D = keypoint_aux_world_coords_2D
        self.complete_keypoints = complete_keypoints
        self.coords_to_dict = coords_to_dict
        self.get_kp = get_keypoints_from_heatmap_batch_maxpool
        self.get_line = get_keypoints_from_heatmap_batch_maxpool_l

        self.device = torch.device(self.device_name if torch.cuda.is_available() and "cuda" in self.device_name else "cpu")
        cfg = yaml.safe_load((self.pnl_dir / "config/hrnetv2_w48.yaml").read_text(encoding="utf-8"))
        cfg_l = yaml.safe_load((self.pnl_dir / "config/hrnetv2_w48_l.yaml").read_text(encoding="utf-8"))
        self.model = get_cls_net(cfg)
        self.model_l = get_cls_net_l(cfg_l)
        self.model.load_state_dict(torch.load(self.weights_kp, map_location=self.device))
        self.model_l.load_state_dict(torch.load(self.weights_line, map_location=self.device))
        self.model.to(self.device).eval()
        self.model_l.to(self.device).eval()
        self.resize = T.Resize((self.cfg.model_height, self.cfg.model_width))

        # Official SoccerNet evaluator helpers bundled by PnLCalib.
        try:
            from evaluate_camera import get_polylines, scale_points, evaluate_camera_prediction
            from evaluate_extremities import mirror_labels
            self.get_polylines = get_polylines
            self.scale_points = scale_points
            self.evaluate_camera_prediction = evaluate_camera_prediction
            self.mirror_labels = mirror_labels
        except Exception:
            self.get_polylines = None
            self.scale_points = None
            self.evaluate_camera_prediction = None
            self.mirror_labels = None
        self._loaded = True

    def evaluate_zip_pair(self, ground_truth_zip: Union[str, Path], prediction_zip: Union[str, Path]) -> Optional[Dict[str, float]]:
        """Run the bundled SoccerNet evaluator in-process on generated ZIPs.

        The evaluator currently reports ``completeness``, ``meanRecall``,
        ``meanPrecision``, ``meanAccuracies`` and ``finalScore``.  We preserve
        those names verbatim instead of relabeling them as another metric.
        """
        self._load_backend()
        try:
            evalai = importlib.import_module("evalai_camera")
            result = evalai.evaluate(
                str(ground_truth_zip), str(prediction_zip),
                self.cfg.model_width, self.cfg.model_height,
            )
            return {k: float(v) for k, v in result.items()}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _world_lookups(self, keypoints: Dict[Any, Any]) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
        xy_lookup: Dict[int, List[float]] = {}
        xyz_lookup: Dict[int, List[float]] = {}
        elevated_main_ids = {12, 15, 16, 19}
        n_main = len(self.keypoint_world_coords_2D)
        Tpc = PnLCalibAdapter.PNL_TO_CANONICAL
        for raw_id in keypoints.keys():
            try:
                kp_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if 1 <= kp_id <= n_main:
                xy = self.keypoint_world_coords_2D[kp_id - 1]
                z_pnl = -2.44 if kp_id in elevated_main_ids else 0.0
            else:
                aux_idx = kp_id - 1 - n_main
                if not (0 <= aux_idx < len(self.keypoint_aux_world_coords_2D)):
                    continue
                xy = self.keypoint_aux_world_coords_2D[aux_idx]
                z_pnl = 0.0
            xyz_pnl = np.array([float(xy[0]), float(xy[1]), float(z_pnl)], dtype=np.float64)
            xyz_can = Tpc @ xyz_pnl
            xy_lookup[kp_id] = xyz_can[:2].tolist()
            xyz_lookup[kp_id] = xyz_can.tolist()
        return xy_lookup, xyz_lookup

    def _vote_with_diagnostics(self, calib) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        full_results: List[Dict[str, Any]] = []
        for mode in self.cfg.candidate_modes:
            for ransac in self.cfg.ransac_values:
                row: Dict[str, Any] = {
                    "mode": str(mode), "use_ransac": int(ransac), "success": False,
                    "rep_err_px": None, "selected": False,
                }
                try:
                    cam_params, rep_err = calib.get_cam_params(
                        mode=mode,
                        use_ransac=ransac,
                        refine=self.cfg.refine,
                        refine_w_lines=self.cfg.refine_lines,
                    )
                    rep = _finite(rep_err)
                    if cam_params is not None and rep is not None:
                        result = {
                            "mode": str(mode), "use_ransac": int(ransac), "rep_err": rep,
                            "cam_params": cam_params,
                            "calib_plane": int(np.asarray(getattr(calib, "ord_pts", [0])).reshape(-1)[0]),
                        }
                        full_results.append(result)
                        row.update({
                            "success": True,
                            "rep_err_px": rep,
                            "x_focal_length": _finite(cam_params.get("x_focal_length")),
                            "y_focal_length": _finite(cam_params.get("y_focal_length")),
                            "position_meters_pnl": [float(x) for x in np.asarray(cam_params.get("position_meters")).reshape(3)],
                        })
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

        selected = None
        if full_results:
            ordered = sorted(full_results, key=lambda x: (x["rep_err"], x["mode"]))
            selected = next(
                (
                    r for r in ordered
                    if r["mode"] == "full" and r["use_ransac"] == 0
                    and r["rep_err"] <= self.cfg.preferred_full_threshold_px
                ),
                ordered[0],
            )
            for row in rows:
                if (
                    row.get("success")
                    and row["mode"] == selected["mode"]
                    and row["use_ransac"] == selected["use_ransac"]
                    and abs(float(row["rep_err_px"]) - float(selected["rep_err"])) <= 1e-6
                ):
                    row["selected"] = True
                    break

        successful = [r for r in rows if r["success"]]
        sorted_rows = sorted(successful, key=lambda r: (r["rep_err_px"], r["mode"], r["use_ransac"]))
        best_by_mode: Dict[str, Any] = {}
        for mode in self.cfg.candidate_modes:
            m = [r for r in sorted_rows if r["mode"] == mode]
            best_by_mode[mode] = _compact(m[0]) if m else None
        full0 = next((r for r in successful if r["mode"] == "full" and r["use_ransac"] == 0), None)
        diag = {
            "grid": {
                "modes": list(self.cfg.candidate_modes),
                "ransac_values": list(self.cfg.ransac_values),
                "refine": self.cfg.refine,
                "refine_lines": self.cfg.refine_lines,
            },
            "num_attempted": len(rows),
            "num_successful": len(successful),
            "selected": None if selected is None else {
                "mode": selected["mode"], "use_ransac": selected["use_ransac"], "rep_err_px": selected["rep_err"]
            },
            "best_overall": _compact(sorted_rows[0]) if sorted_rows else None,
            "best_by_mode": best_by_mode,
            "full_no_ransac": _compact(full0),
            "candidates": rows,
        }
        return selected, diag

    def _official_frame_metrics(self, annotation: Dict[str, Any], cam_params: Dict[str, Any]) -> Dict[str, Optional[float]]:
        if self.get_polylines is None:
            return {"accuracy": None, "precision": None, "recall": None}
        gt = self.scale_points(annotation, self.cfg.model_width, self.cfg.model_height)
        pred = self.get_polylines(cam_params, self.cfg.model_width, self.cfg.model_height, sampling_factor=0.9)
        c1, _, _ = self.evaluate_camera_prediction(pred, gt, 20)
        c2, _, _ = self.evaluate_camera_prediction(pred, self.mirror_labels(gt), 20)
        a1 = float(c1[0, 0] / c1.sum()) if c1.sum() > 0 else 0.0
        a2 = float(c2[0, 0] / c2.sum()) if c2.sum() > 0 else 0.0
        c = c1 if a1 > a2 else c2
        acc = max(a1, a2)
        precision = float(c[0, 0] / c[0, :].sum()) if c[0, :].sum() > 0 else None
        recall = float(c[0, 0] / (c[0, 0] + c[1, 0])) if (c[0, 0] + c[1, 0]) > 0 else None
        return {"accuracy": acc, "precision": precision, "recall": recall}

    def process_frame(self, image_path: Union[str, Path], frame_index: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
        self._load_backend()
        image_path = Path(image_path)
        annotation_path = image_path.with_suffix(".json")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8")) if annotation_path.exists() else None
        t0 = time.perf_counter()
        with self.Image.open(image_path) as pil_image:
            original_size = pil_image.size
            with self.torch.no_grad():
                tensor = self.tvf.to_tensor(pil_image).float().to(self.device).unsqueeze(0)
                if tensor.shape[-2:] != (self.cfg.model_height, self.cfg.model_width):
                    tensor = self.resize(tensor)
                heatmaps = self.model(tensor)
                heatmaps_l = self.model_l(tensor)
                kp_coords = self.get_kp(heatmaps[:, :-1, :, :])
                line_coords = self.get_line(heatmaps_l[:, :-1, :, :])
                kp_dict = self.coords_to_dict(kp_coords, threshold=self.cfg.kp_threshold)
                lines_dict = self.coords_to_dict(line_coords, threshold=self.cfg.line_threshold)
                kp_dict, lines_dict = self.complete_keypoints(
                    kp_dict[0], lines_dict[0], w=self.cfg.model_width, h=self.cfg.model_height
                )

        calib = self.FramebyFrameCalib(iwidth=self.cfg.model_width, iheight=self.cfg.model_height)
        calib.update(kp_dict, lines_dict)
        selected, cand = self._vote_with_diagnostics(calib)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        row: Dict[str, Any] = {
            "frame_index": int(frame_index),
            "frame_id": image_path.stem,
            "filename": image_path.name,
            "original_width": int(original_size[0]),
            "original_height": int(original_size[1]),
            "runtime_ms": float(elapsed_ms),
            "solved": selected is not None,
            "official_prediction_written": False,
            "camera_status": CameraStatus.INVALID.value,
            "ground_status": CameraStatus.INVALID.value,
            "vertical_3d_status": CameraStatus.INVALID.value,
            "offside_3d_ready": False,
        }
        if selected is None:
            row["error"] = "no_camera_solution"
            return row, None, annotation or {}

        keypoints_used = getattr(calib, "keypoints_dict", {}) or {}
        lines_used = getattr(calib, "lines_dict", {}) or {}
        world_xy, world_xyz = self._world_lookups(keypoints_used)
        coverage = summarize_keypoint_coverage(
            keypoints_used,
            self.cfg.model_width,
            self.cfg.model_height,
            world_xy_lookup=world_xy,
        )
        correspondences = build_keypoint_correspondences(keypoints_used, world_xyz)
        selected["pnl_refine"] = bool(self.cfg.refine_lines)
        selected["candidate_diagnostics"] = cand
        selected["evidence"] = {
            "num_keypoints_used": len(keypoints_used),
            "num_lines_used": len(lines_used),
            "kp_threshold": self.cfg.kp_threshold,
            "line_threshold": self.cfg.line_threshold,
            "keypoint_correspondences": correspondences,
            **{f"keypoint_{k}": v for k, v in coverage.items()},
        }

        cam = PnLCalibAdapter.camera_state_from_result(
            selected,
            frame_index=frame_index,
            image_width=self.cfg.model_width,
            image_height=self.cfg.model_height,
            pnl_refine=self.cfg.refine_lines,
            source_extra={
                "weights_kp": self.weights_kp.name,
                "weights_lines": self.weights_line.name,
                "kp_threshold": self.cfg.kp_threshold,
                "line_threshold": self.cfg.line_threshold,
                "benchmark_split": self.cfg.split,
            },
        )
        self.quality_gate.evaluate(cam)
        caps = cam.diagnostics.get("capabilities", {}) or {}
        ground = (caps.get("ground_geometry") or {}).get("status", CameraStatus.INVALID.value)
        vertical = (caps.get("vertical_3d") or {}).get("status", CameraStatus.INVALID.value)
        rep = _finite(selected.get("rep_err"))
        official_written = rep is not None and rep <= self.cfg.max_reproj_err_px
        official_metrics = {"accuracy": None, "precision": None, "recall": None}
        if official_written and annotation is not None:
            official_metrics = self._official_frame_metrics(annotation, selected["cam_params"])

        C = cam.camera_center_world_m.tolist()
        row.update({
            "rep_err_px": rep,
            "mode": selected.get("mode"),
            "use_ransac": selected.get("use_ransac"),
            "num_keypoints": len(keypoints_used),
            "num_lines": len(lines_used),
            "keypoint_hull_ratio": coverage.get("image_convex_hull_area_ratio"),
            "keypoint_x_span_ratio": coverage.get("image_x_span_ratio"),
            "keypoint_y_span_ratio": coverage.get("image_y_span_ratio"),
            "keypoint_quadrants": coverage.get("image_quadrants_occupied"),
            "world_x_span_m": coverage.get("world_x_span_m"),
            "world_y_span_m": coverage.get("world_y_span_m"),
            "camera_status": cam.status.value,
            "ground_status": ground,
            "vertical_3d_status": vertical,
            "offside_3d_ready": bool(caps.get("offside_3d_ready", False)),
            "ground_policy_version": ((caps.get("ground_geometry") or {}).get("policy_version")),
            "vertical_3d_policy_version": ((caps.get("vertical_3d") or {}).get("policy_version")),
            "vertical_3d_reasons": ((caps.get("vertical_3d") or {}).get("reasons", [])),
            "line_refinement_effective": bool((cam.diagnostics.get("line_refinement") or {}).get("effective", False)),
            "fx": float(cam.K[0, 0]),
            "camera_x_m": float(C[0]), "camera_y_m": float(C[1]), "camera_z_m": float(C[2]),
            "best_full_rep_err_px": _nested_number(cand, "best_by_mode", "full", "rep_err_px"),
            "full_no_ransac_rep_err_px": _nested_number(cand, "full_no_ransac", "rep_err_px"),
            "official_prediction_written": official_written,
            "official_frame_accuracy": official_metrics["accuracy"],
            "official_frame_precision": official_metrics["precision"],
            "official_frame_recall": official_metrics["recall"],
        })
        return row, selected["cam_params"] if official_written else None, annotation or {}

    def run(self, dataset_root: Union[str, Path], output_dir: Union[str, Path], *, prefix: Optional[str] = None) -> Dict[str, Any]:
        self._load_backend()
        split_dir = ensure_split_extracted(dataset_root, self.cfg.split)
        paths = select_image_paths(split_dir, self.cfg.limit, self.cfg.selection)
        if not paths:
            raise FileNotFoundError(f"No .jpg frames found in {split_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix or f"soccernet_{self.cfg.split}_{'full' if self.cfg.limit is None else f'n{len(paths)}'}"
        pred_zip = output_dir / f"{prefix}_pred.zip"
        gt_zip = output_dir / f"{prefix}_gt.zip"
        records: List[Dict[str, Any]] = []
        with zipfile.ZipFile(pred_zip, "w", zipfile.ZIP_DEFLATED) as pzf, zipfile.ZipFile(gt_zip, "w", zipfile.ZIP_DEFLATED) as gzf:
            for idx, path in enumerate(paths):
                row, cam_params, annotation = self.process_frame(path, idx)
                records.append(row)
                if annotation:
                    gzf.writestr(f"{path.stem}.json", json.dumps(annotation))
                if cam_params is not None:
                    pzf.writestr(f"camera_{path.stem}.json", json.dumps(cam_params))
                if len(paths) <= 20 or (idx + 1) % max(1, len(paths) // 10) == 0:
                    print(f"[{idx+1}/{len(paths)}] {path.name}: {row.get('camera_status')} {row.get('mode')} rep={row.get('rep_err_px')}")

        summary = summarize_records(records, self.cfg)
        summary["capability_policies"] = {
            "ground_geometry": self.quality_gate.ground_policy.to_dict(),
            "vertical_3d": self.quality_gate.vertical_policy.to_dict(),
        }
        summary["thresholds_frozen"] = bool(
            self.quality_gate.ground_policy.thresholds_frozen
            and self.quality_gate.vertical_policy.thresholds_frozen
        )
        summary["prediction_zip"] = str(pred_zip)
        summary["ground_truth_zip"] = str(gt_zip)
        official = self.evaluate_zip_pair(gt_zip, pred_zip)
        summary["official_evaluator_metrics"] = official
        if isinstance(official, dict) and "error" not in official:
            local = summary.get("official_like_metrics", {})
            summary["official_evaluator_parity_delta"] = {
                key: (None if local.get(key) is None else float(local.get(key)) - float(official.get(key)))
                for key in ("completeness", "meanRecall", "meanPrecision", "meanAccuracies", "finalScore")
                if official.get(key) is not None
            }
        report_paths = write_reports(records, summary, output_dir, prefix)
        summary["report_paths"] = report_paths
        # Rewrite once with paths included.
        Path(report_paths["summary_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"records": records, "summary": summary, "paths": report_paths}


def _nested_number(obj: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return _finite(cur)
