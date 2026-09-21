from __future__ import annotations

"""Fair Stage-4 model, pipeline and third-party benchmark comparisons.

This module deliberately separates three questions which are often mixed in a
single headline number:

* Does a new Stage-4 release change an already frozen point estimate?
* Does it improve accuracy against metric ground truth on the same examples?
* Does an external/root-relative pose model improve articulation, even though
  it cannot claim pitch-world or offside accuracy?

All comparisons use paired records.  External models must declare their
coordinate scope so a root-relative prediction can never accidentally receive
GlobalMPJPE, longitudinal or GALE scores.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .evaluation import mpjpe, procrustes_mpjpe, root_aligned_mpjpe
from .wholebody import LEGAL_GEOMETRY_CANDIDATE_23, POSE23_NAMES


PoseKey = Tuple[int, str]
PITCH_WORLD_METRIC = "PITCH_WORLD_METRIC"
ROOT_RELATIVE_METRIC = "ROOT_RELATIVE_METRIC"
SCALE_AMBIGUOUS = "SCALE_AMBIGUOUS"
COORDINATE_SCOPES = (PITCH_WORLD_METRIC, ROOT_RELATIVE_METRIC, SCALE_AMBIGUOUS)
LOWER_IS_BETTER_METRICS = (
    "GlobalMPJPE_m",
    "RootAlignedMPJPE_m",
    "PA_MPJPE_m",
    "LongitudinalMAE_m",
    "LongitudinalP90_m",
    "GALE_m",
)


@dataclass(frozen=True)
class PredictionBundle:
    name: str
    path: Path
    format: str
    coordinate_scope: str
    poses: Mapping[PoseKey, np.ndarray]
    metadata: Mapping[str, object]


def _read_json(path: str | Path) -> Mapping[str, object]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _pose23_from_record(record: Mapping[str, object]) -> np.ndarray:
    for field in ("xyz23_m", "xyz23_world_m"):
        if field in record:
            value = record[field]
            arr = np.asarray(
                [[np.nan if item is None else item for item in row] for row in value],
                dtype=np.float64,
            )
            if arr.shape != (23, 3):
                raise ValueError(f"Expected {field} shape (23,3), got {arr.shape}")
            return arr

    joints = record.get("joints") or {}
    if not isinstance(joints, Mapping):
        raise ValueError("'joints' must be an object keyed by METRIC_POSE_23 joint name")
    arr = np.full((23, 3), np.nan, dtype=np.float64)
    for index, name in enumerate(POSE23_NAMES):
        if name in joints and joints[name] is not None:
            arr[index] = np.asarray(joints[name], dtype=np.float64)
    return arr


def load_ground_truth(path: str | Path) -> Dict[PoseKey, np.ndarray]:
    document = _read_json(path)
    poses: Dict[PoseKey, np.ndarray] = {}
    for record in document.get("poses", []):
        key = (int(record["frame_index"]), str(record["track_id"]))
        if key in poses:
            raise ValueError(f"Duplicate GT pose record {key}")
        poses[key] = _pose23_from_record(record)
    if not poses:
        raise ValueError("Ground-truth document contains no pose records")
    return poses


def _load_native_stage4(document: Mapping[str, object]) -> Dict[PoseKey, np.ndarray]:
    poses: Dict[PoseKey, np.ndarray] = {}
    for track in document.get("tracks", []):
        track_id = str(track.get("track_id"))
        for observation in track.get("observations", []):
            key = (int(observation["frame_index"]), track_id)
            if key in poses:
                raise ValueError(f"Duplicate Stage-4 pose record {key}")
            arr = np.full((23, 3), np.nan, dtype=np.float64)
            for joint in observation.get("metric_pose23", []):
                index = int(joint.get("index", -1))
                xyz = joint.get("xyz_world_m")
                if 0 <= index < 23 and xyz is not None:
                    arr[index] = np.asarray(xyz, dtype=np.float64)
            poses[key] = arr
    return poses


def load_prediction(
    path: str | Path,
    *,
    name: Optional[str] = None,
    format: str = "auto",
    coordinate_scope: Optional[str] = None,
) -> PredictionBundle:
    resolved = Path(path).expanduser().resolve()
    document = _read_json(resolved)
    schema = str(document.get("schema_version", ""))
    detected = format
    if format == "auto":
        if schema.startswith("metric-pose-3d-state-") or "tracks" in document:
            detected = "stage4"
        elif schema.startswith("metric-pose23-prediction-") or "poses" in document:
            detected = "canonical"
        else:
            raise ValueError(f"Cannot detect prediction format for schema '{schema}'")

    if detected == "stage4":
        poses = _load_native_stage4(document)
        default_scope = PITCH_WORLD_METRIC
        metadata = {
            "schema_version": schema,
            "stage4_version": document.get("stage4_version"),
            "source_stage1": document.get("source_stage1"),
            "source_stage3": document.get("source_stage3"),
            "source_initializer": document.get("source_initializer"),
        }
    elif detected == "canonical":
        poses = {}
        for record in document.get("poses", []):
            key = (int(record["frame_index"]), str(record["track_id"]))
            if key in poses:
                raise ValueError(f"Duplicate canonical prediction record {key}")
            poses[key] = _pose23_from_record(record)
        default_scope = str(document.get("coordinate_scope", ""))
        metadata = {
            "schema_version": schema,
            "model": document.get("model"),
            "provenance": document.get("provenance"),
        }
    else:
        raise ValueError("format must be one of: auto, stage4, canonical")

    scope = coordinate_scope or default_scope
    if scope not in COORDINATE_SCOPES:
        raise ValueError(
            f"Prediction coordinate_scope must be one of {COORDINATE_SCOPES}; got {scope!r}"
        )
    if not poses:
        raise ValueError(f"Prediction {resolved} contains no pose records")
    return PredictionBundle(
        name=name or resolved.stem,
        path=resolved,
        format=detected,
        coordinate_scope=scope,
        poses=poses,
        metadata=metadata,
    )


def _finite_joint_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)


def _masked_pose(pose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[~mask] = np.nan
    return result


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not finite.size else float(np.mean(finite))


def _percentile_or_none(values: Iterable[float], q: float) -> Optional[float]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not finite.size else float(np.percentile(finite, q))


def _per_pose_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    *,
    coordinate_scope: str,
    attack_sign: int,
) -> Mapping[str, Optional[float]]:
    p = _masked_pose(pred, mask)
    g = _masked_pose(gt, mask)
    result: Dict[str, Optional[float]] = {
        "GlobalMPJPE_m": None,
        "RootAlignedMPJPE_m": None,
        "PA_MPJPE_m": None,
        "LongitudinalMAE_m": None,
        "LongitudinalP90_m": None,
        "GALE_m": None,
    }
    if int(np.sum(mask)) >= 3:
        result["PA_MPJPE_m"] = _finite_or_none(procrustes_mpjpe(p, g))
    if coordinate_scope in (PITCH_WORLD_METRIC, ROOT_RELATIVE_METRIC):
        result["RootAlignedMPJPE_m"] = _finite_or_none(root_aligned_mpjpe(p, g))
    if coordinate_scope != PITCH_WORLD_METRIC or not np.any(mask):
        return result

    result["GlobalMPJPE_m"] = _finite_or_none(mpjpe(p, g))
    legal = mask & np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool)
    if np.any(legal):
        errors = np.abs(pred[legal, 0] - gt[legal, 0])
        result["LongitudinalMAE_m"] = float(np.mean(errors))
        result["LongitudinalP90_m"] = float(np.percentile(errors, 90))
        pred_goalward = float(np.max(attack_sign * pred[legal, 0]))
        gt_goalward = float(np.max(attack_sign * gt[legal, 0]))
        result["GALE_m"] = abs(pred_goalward - gt_goalward)
    return result


def evaluate_prediction_bundle(
    bundle: PredictionBundle,
    gt: Mapping[PoseKey, np.ndarray],
    *,
    attack_sign: int,
    masks: Optional[Mapping[PoseKey, np.ndarray]] = None,
    cohort_name: str = "available_records",
) -> Mapping[str, object]:
    if attack_sign not in (-1, 1):
        raise ValueError("attack_sign must be -1 or +1")
    matched_keys = sorted(set(bundle.poses).intersection(gt))
    per_pose: List[Mapping[str, object]] = []
    global_joint_errors: List[float] = []
    longitudinal_errors: List[float] = []
    goalward_by_frame: Dict[int, List[Tuple[str, float, float]]] = {}

    for key in matched_keys:
        pred = bundle.poses[key]
        truth = gt[key]
        mask = _finite_joint_mask(pred, truth)
        if masks is not None:
            requested = masks.get(key)
            if requested is None:
                continue
            mask &= np.asarray(requested, dtype=bool)
        metrics = _per_pose_metrics(
            pred,
            truth,
            mask,
            coordinate_scope=bundle.coordinate_scope,
            attack_sign=attack_sign,
        )
        record: Dict[str, object] = {
            "frame_index": key[0],
            "track_id": key[1],
            "joint_count": int(np.sum(mask)),
        }
        record.update(metrics)
        per_pose.append(record)

        if bundle.coordinate_scope == PITCH_WORLD_METRIC and np.any(mask):
            global_joint_errors.extend(np.linalg.norm(pred[mask] - truth[mask], axis=1).tolist())
            legal = mask & np.asarray(LEGAL_GEOMETRY_CANDIDATE_23, dtype=bool)
            if np.any(legal):
                longitudinal_errors.extend(np.abs(pred[legal, 0] - truth[legal, 0]).tolist())
                pred_anchor = float(np.max(attack_sign * pred[legal, 0]))
                gt_anchor = float(np.max(attack_sign * truth[legal, 0]))
                goalward_by_frame.setdefault(key[0], []).append((key[1], pred_anchor, gt_anchor))

    metric_names = list(LOWER_IS_BETTER_METRICS)
    metrics: Dict[str, object] = {
        metric: _mean_or_none(record.get(metric) for record in per_pose)
        for metric in metric_names
    }
    metrics["GlobalMPJPE_joint_weighted_m"] = _mean_or_none(global_joint_errors)
    metrics["LongitudinalMedian_m"] = _percentile_or_none(longitudinal_errors, 50)
    metrics["LongitudinalP90_m"] = _percentile_or_none(longitudinal_errors, 90)

    ordering: Dict[str, Mapping[str, object]] = {}
    ordering_pairs: Dict[str, List[Mapping[str, object]]] = {}
    for threshold in (2.0, 1.0, 0.5, 0.25):
        correct = 0
        total = 0
        pair_records: List[Mapping[str, object]] = []
        for frame, rows in goalward_by_frame.items():
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    gt_separation = abs(rows[left][2] - rows[right][2])
                    if gt_separation >= threshold or gt_separation < 1e-9:
                        continue
                    is_correct = bool(
                        np.sign(rows[left][1] - rows[right][1])
                        == np.sign(rows[left][2] - rows[right][2])
                    )
                    total += 1
                    correct += int(is_correct)
                    pair_records.append({
                        "frame_index": frame,
                        "track_a": rows[left][0],
                        "track_b": rows[right][0],
                        "gt_separation_m": gt_separation,
                        "correct": is_correct,
                    })
        stratum = f"gt_separation_lt_{threshold:g}m"
        ordering[stratum] = {
            "accuracy": None if not total else float(correct / total),
            "correct": correct,
            "total": total,
        }
        ordering_pairs[stratum] = pair_records
    metrics["PairwiseLongitudinalOrdering"] = ordering

    finite_joint_count = int(sum(int(record["joint_count"]) for record in per_pose))
    unavailable = []
    if bundle.coordinate_scope == ROOT_RELATIVE_METRIC:
        unavailable = [
            "GlobalMPJPE",
            "LongitudinalMAE",
            "LongitudinalP90",
            "GALE",
            "PairwiseLongitudinalOrdering",
        ]
    elif bundle.coordinate_scope == SCALE_AMBIGUOUS:
        unavailable = [
            "GlobalMPJPE",
            "RootAlignedMPJPE",
            "LongitudinalMAE",
            "LongitudinalP90",
            "GALE",
            "PairwiseLongitudinalOrdering",
        ]
    return {
        "system": bundle.name,
        "path": str(bundle.path),
        "format": bundle.format,
        "coordinate_scope": bundle.coordinate_scope,
        "cohort": cohort_name,
        "prediction_pose_records": len(bundle.poses),
        "gt_pose_records": len(gt),
        "matched_pose_records": len(matched_keys),
        "evaluated_pose_records": len(per_pose),
        "valid_joint_observations": finite_joint_count,
        "coverage": {
            "prediction_gt_pose_match_fraction": float(len(matched_keys) / len(gt)),
            "evaluated_gt_pose_fraction": float(len(per_pose) / len(gt)),
        },
        "metrics": metrics,
        "per_pose": per_pose,
        "ordering_pairs": ordering_pairs,
        "unavailable_metrics": unavailable,
        "metadata": dict(bundle.metadata),
    }


def _common_masks(
    bundles: Sequence[PredictionBundle], gt: Mapping[PoseKey, np.ndarray]
) -> Dict[PoseKey, np.ndarray]:
    common_keys = set(gt)
    for bundle in bundles:
        common_keys &= set(bundle.poses)
    masks: Dict[PoseKey, np.ndarray] = {}
    for key in sorted(common_keys):
        mask = np.isfinite(gt[key]).all(axis=1)
        for bundle in bundles:
            mask &= np.isfinite(bundle.poses[key]).all(axis=1)
        if np.any(mask):
            masks[key] = mask
    return masks


def _per_pose_index(evaluation: Mapping[str, object]) -> Dict[PoseKey, Mapping[str, object]]:
    return {
        (int(record["frame_index"]), str(record["track_id"])): record
        for record in evaluation.get("per_pose", [])
    }


def _bootstrap_mean_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Mapping[str, object]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if not delta.size:
        return {"count": 0, "mean_delta": None, "ci95": [None, None]}
    mean_delta = float(np.mean(delta))
    if samples <= 0 or delta.size == 1:
        return {"count": int(delta.size), "mean_delta": mean_delta, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    means = np.mean(delta[indices], axis=1)
    return {
        "count": int(delta.size),
        "mean_delta": mean_delta,
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**32))


def paired_comparison(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Mapping[str, object]:
    candidate_index = _per_pose_index(candidate)
    baseline_index = _per_pose_index(baseline)
    keys = sorted(set(candidate_index).intersection(baseline_index))
    metrics: Dict[str, object] = {}
    for metric in LOWER_IS_BETTER_METRICS:
        candidate_values: List[float] = []
        baseline_values: List[float] = []
        for key in keys:
            c = candidate_index[key].get(metric)
            b = baseline_index[key].get(metric)
            if c is None or b is None or not np.isfinite(c) or not np.isfinite(b):
                continue
            candidate_values.append(float(c))
            baseline_values.append(float(b))
        comparison = dict(
            _bootstrap_mean_delta(
                np.asarray(candidate_values),
                np.asarray(baseline_values),
                samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, str(candidate["system"]), str(baseline["system"]), metric),
            )
        )
        comparison["direction"] = "candidate_minus_baseline; negative_is_better"
        comparison["candidate_mean"] = _mean_or_none(candidate_values)
        comparison["baseline_mean"] = _mean_or_none(baseline_values)
        metrics[metric] = comparison
    candidate_ordering = candidate.get("ordering_pairs") or {}
    baseline_ordering = baseline.get("ordering_pairs") or {}
    for stratum in sorted(set(candidate_ordering).intersection(baseline_ordering)):
        def index_pairs(rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[int, str, str], Mapping[str, object]]:
            return {
                (int(row["frame_index"]), str(row["track_a"]), str(row["track_b"])): row
                for row in rows
            }

        candidate_pairs = index_pairs(candidate_ordering[stratum])
        baseline_pairs = index_pairs(baseline_ordering[stratum])
        pair_keys = sorted(set(candidate_pairs).intersection(baseline_pairs))
        candidate_errors = np.asarray(
            [1.0 - float(candidate_pairs[key]["correct"]) for key in pair_keys], dtype=np.float64
        )
        baseline_errors = np.asarray(
            [1.0 - float(baseline_pairs[key]["correct"]) for key in pair_keys], dtype=np.float64
        )
        metric = f"PairwiseOrderingError_{stratum}"
        comparison = dict(
            _bootstrap_mean_delta(
                candidate_errors,
                baseline_errors,
                samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, str(candidate["system"]), str(baseline["system"]), metric),
            )
        )
        comparison["direction"] = "candidate_error_minus_baseline_error; negative_is_better"
        comparison["candidate_accuracy"] = None if not pair_keys else float(1.0 - np.mean(candidate_errors))
        comparison["baseline_accuracy"] = None if not pair_keys else float(1.0 - np.mean(baseline_errors))
        metrics[metric] = comparison
    return {
        "candidate": candidate["system"],
        "baseline": baseline["system"],
        "cohort": candidate["cohort"],
        "paired_pose_records": len(keys),
        "metrics": metrics,
    }


def _evaluate_gates(
    gate_specs: Sequence[Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    checks = []
    for index, spec in enumerate(gate_specs):
        comparison_name = str(spec.get("comparison", ""))
        metric = str(spec.get("metric", ""))
        comparison = comparisons.get(comparison_name)
        result = None if comparison is None else comparison.get("metrics", {}).get(metric)
        reasons = []
        if not result or result.get("mean_delta") is None:
            reasons.append("comparison_or_metric_unavailable")
        else:
            delta = float(result["mean_delta"])
            if spec.get("max_delta") is not None and delta > float(spec["max_delta"]):
                reasons.append("mean_delta_above_max")
            if spec.get("min_delta") is not None and delta < float(spec["min_delta"]):
                reasons.append("mean_delta_below_min")
            ci = result.get("ci95") or [None, None]
            if spec.get("ci_upper_lte") is not None:
                if ci[1] is None or float(ci[1]) > float(spec["ci_upper_lte"]):
                    reasons.append("ci_upper_above_limit_or_unavailable")
            if spec.get("ci_lower_gte") is not None:
                if ci[0] is None or float(ci[0]) < float(spec["ci_lower_gte"]):
                    reasons.append("ci_lower_below_limit_or_unavailable")
        checks.append({
            "name": str(spec.get("name", f"gate_{index + 1}")),
            "comparison": comparison_name,
            "metric": metric,
            "passed": not reasons,
            "reasons": reasons,
            "thresholds": {key: spec[key] for key in ("max_delta", "min_delta", "ci_upper_lte", "ci_lower_gte") if key in spec},
        })
    return {
        "status": "NOT_CONFIGURED" if not checks else ("PASS" if all(x["passed"] for x in checks) else "FAIL"),
        "checks": checks,
    }


def evaluate_system_manifest(manifest_path: str | Path) -> Mapping[str, object]:
    resolved = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(resolved)
    base = resolved.parent
    gt_path = _resolve(base, manifest["gt"])
    gt = load_ground_truth(gt_path)
    attack_sign = int(manifest.get("attack_sign", 1))
    bootstrap = manifest.get("bootstrap") or {}
    bootstrap_samples = int(bootstrap.get("samples", 2000))
    bootstrap_seed = int(bootstrap.get("seed", 20260917))

    systems_spec = manifest.get("systems") or {}
    if not isinstance(systems_spec, Mapping) or len(systems_spec) < 2:
        raise ValueError("System manifest requires at least two named systems")
    bundles: Dict[str, PredictionBundle] = {}
    system_roles: Dict[str, str] = {}
    for name, raw_spec in systems_spec.items():
        spec = raw_spec if isinstance(raw_spec, Mapping) else {"pred": raw_spec}
        pred_value = spec.get("pred")
        if not pred_value:
            raise ValueError(f"System {name!r} is missing 'pred'")
        bundles[str(name)] = load_prediction(
            _resolve(base, pred_value),
            name=str(name),
            format=str(spec.get("format", "auto")),
            coordinate_scope=spec.get("coordinate_scope"),
        )
        system_roles[str(name)] = str(spec.get("role", "unspecified"))

    shared_masks = _common_masks(list(bundles.values()), gt)
    evaluations: Dict[str, Mapping[str, object]] = {}
    common_evaluations: Dict[str, Mapping[str, object]] = {}
    for name, bundle in bundles.items():
        evaluations[name] = evaluate_prediction_bundle(
            bundle, gt, attack_sign=attack_sign, cohort_name="system_available_records"
        )
        common_evaluations[name] = evaluate_prediction_bundle(
            bundle,
            gt,
            attack_sign=attack_sign,
            masks=shared_masks,
            cohort_name="all_systems_common_pose_and_joint_cohort",
        )

    requested_pairs = manifest.get("comparisons") or []
    if not requested_pairs:
        primary = str(manifest.get("primary_system", next(iter(bundles))))
        requested_pairs = [
            {"candidate": primary, "baseline": name}
            for name in bundles
            if name != primary
        ]
    comparisons: Dict[str, Mapping[str, object]] = {}
    for pair in requested_pairs:
        candidate_name = str(pair["candidate"])
        baseline_name = str(pair["baseline"])
        if candidate_name not in bundles or baseline_name not in bundles:
            raise ValueError(f"Unknown comparison pair {candidate_name!r} vs {baseline_name!r}")
        comparison_name = str(pair.get("name", f"{candidate_name}__vs__{baseline_name}"))
        pair_masks = _common_masks([bundles[candidate_name], bundles[baseline_name]], gt)
        pair_cohort_name = f"pairwise_common_pose_and_joint_cohort:{comparison_name}"
        candidate_pair = evaluate_prediction_bundle(
            bundles[candidate_name],
            gt,
            attack_sign=attack_sign,
            masks=pair_masks,
            cohort_name=pair_cohort_name,
        )
        baseline_pair = evaluate_prediction_bundle(
            bundles[baseline_name],
            gt,
            attack_sign=attack_sign,
            masks=pair_masks,
            cohort_name=pair_cohort_name,
        )
        comparison_result = dict(paired_comparison(
            candidate_pair,
            baseline_pair,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ))
        comparison_result["fairness"] = {
            "common_pose_records": len(pair_masks),
            "common_joint_observations": int(sum(np.sum(mask) for mask in pair_masks.values())),
        }
        comparisons[comparison_name] = comparison_result

    gate = _evaluate_gates(manifest.get("gates") or [], comparisons)
    return {
        "schema_version": "stage4-system-comparison-1.0",
        "manifest": str(resolved),
        "gt": str(gt_path),
        "attack_sign": attack_sign,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed, "unit": "pose_record"},
        "system_roles": system_roles,
        "fairness": {
            "common_pose_records": len(shared_masks),
            "common_joint_observations": int(sum(np.sum(mask) for mask in shared_masks.values())),
            "rule": "The shared table uses the all-system intersection; each headline pairwise delta uses that pair's own common pose/joint intersection.",
            "coordinate_scope_rule": "Only PITCH_WORLD_METRIC systems receive global/longitudinal/offside-oriented metrics.",
        },
        "systems_available_cohort": evaluations,
        "systems_common_cohort": common_evaluations,
        "comparisons": comparisons,
        "release_gate": gate,
    }


def _native_selected_status(document: Mapping[str, object]) -> Dict[str, str]:
    return {
        str(track.get("track_id")): str(track.get("selected_frame_pose_status", "UNKNOWN"))
        for track in document.get("tracks", [])
    }


def compare_pipeline_outputs(
    candidate_path: str | Path,
    baseline_path: str | Path,
    *,
    max_abs_delta_m: Optional[float] = None,
    require_no_finite_loss: bool = False,
) -> Mapping[str, object]:
    candidate = load_prediction(candidate_path, name="candidate", format="stage4")
    baseline = load_prediction(baseline_path, name="baseline", format="stage4")
    common_keys = sorted(set(candidate.poses).intersection(baseline.poses))
    euclidean: List[float] = []
    axis_abs: List[List[float]] = []
    lost = gained = 0
    for key in common_keys:
        c = candidate.poses[key]
        b = baseline.poses[key]
        c_finite = np.isfinite(c).all(axis=1)
        b_finite = np.isfinite(b).all(axis=1)
        both = c_finite & b_finite
        if np.any(both):
            delta = c[both] - b[both]
            euclidean.extend(np.linalg.norm(delta, axis=1).tolist())
            axis_abs.extend(np.abs(delta).tolist())
        lost += int(np.sum(b_finite & ~c_finite))
        gained += int(np.sum(c_finite & ~b_finite))

    distances = np.asarray(euclidean, dtype=np.float64)
    axes = np.asarray(axis_abs, dtype=np.float64)
    max_delta = None if not distances.size else float(np.max(distances))
    candidate_doc = _read_json(candidate.path)
    baseline_doc = _read_json(baseline.path)
    candidate_status = _native_selected_status(candidate_doc)
    baseline_status = _native_selected_status(baseline_doc)
    transitions: Dict[str, int] = {}
    per_track_transitions = []
    for track_id in sorted(set(candidate_status).union(baseline_status)):
        transition = f"{baseline_status.get(track_id, 'ABSENT')}->{candidate_status.get(track_id, 'ABSENT')}"
        transitions[transition] = transitions.get(transition, 0) + 1
        per_track_transitions.append({"track_id": track_id, "transition": transition})

    gate_checks = []
    if max_abs_delta_m is not None:
        gate_checks.append({
            "name": "max_abs_coordinate_delta_m",
            "passed": max_delta is not None and max_delta <= float(max_abs_delta_m),
            "observed": max_delta,
            "limit": float(max_abs_delta_m),
        })
    if require_no_finite_loss:
        gate_checks.append({
            "name": "no_finite_joint_loss",
            "passed": lost == 0,
            "observed_lost_joint_observations": lost,
            "limit": 0,
        })
    gate_status = "NOT_CONFIGURED" if not gate_checks else (
        "PASS" if all(check["passed"] for check in gate_checks) else "FAIL"
    )
    return {
        "schema_version": "stage4-pipeline-regression-1.0",
        "candidate": {
            "path": str(candidate.path),
            "version": candidate.metadata.get("stage4_version"),
            "pose_records": len(candidate.poses),
        },
        "baseline": {
            "path": str(baseline.path),
            "version": baseline.metadata.get("stage4_version"),
            "pose_records": len(baseline.poses),
        },
        "coverage": {
            "common_pose_records": len(common_keys),
            "candidate_only_pose_records": len(set(candidate.poses) - set(baseline.poses)),
            "baseline_only_pose_records": len(set(baseline.poses) - set(candidate.poses)),
            "paired_finite_joint_observations": int(distances.size),
            "lost_finite_joint_observations": lost,
            "gained_finite_joint_observations": gained,
        },
        "coordinate_drift": {
            "mean_euclidean_m": None if not distances.size else float(np.mean(distances)),
            "median_euclidean_m": None if not distances.size else float(np.median(distances)),
            "p95_euclidean_m": None if not distances.size else float(np.percentile(distances, 95)),
            "max_euclidean_m": max_delta,
            "axis_mae_m": {
                "x": None if not axes.size else float(np.mean(axes[:, 0])),
                "y": None if not axes.size else float(np.mean(axes[:, 1])),
                "z": None if not axes.size else float(np.mean(axes[:, 2])),
            },
        },
        "selected_frame_status_transitions": {
            "counts": transitions,
            "per_track": per_track_transitions,
            "note": "Status transitions are reported separately from XYZ drift because output-semantics repairs may intentionally change status only.",
        },
        "regression_gate": {"status": gate_status, "checks": gate_checks},
    }


def _unit_scale(units: str) -> float:
    normalized = units.lower()
    scales = {
        "m": 1.0,
        "metre": 1.0,
        "meter": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        # A scale-ambiguous model still needs a lossless canonical container.
        # These values must only be used with coordinate_scope=SCALE_AMBIGUOUS.
        "model": 1.0,
        "unitless": 1.0,
    }
    if normalized not in scales:
        raise ValueError("units must be one of: m, metre, meter, cm, mm, model, unitless")
    return scales[normalized]


def _json_safe_pose(pose: np.ndarray) -> List[List[Optional[float]]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in np.asarray(pose, dtype=np.float64)
    ]


def convert_external_prediction(
    input_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
) -> Mapping[str, object]:
    """Convert a simple external pose export into the canonical prediction schema.

    The input must contain a ``poses`` array.  Every record contains
    ``frame_index``, ``track_id`` and the configured XYZ field (default ``xyz``).
    Joint names may be global in the adapter or supplied per record.  The
    adapter maps destination METRIC_POSE_23 names to source names/indices.
    """

    source_path = Path(input_path).expanduser().resolve()
    config_path = Path(adapter_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    source = _read_json(source_path)
    config = _read_json(config_path)
    coordinate_scope = str(config.get("coordinate_scope", ""))
    if coordinate_scope not in COORDINATE_SCOPES:
        raise ValueError(f"adapter coordinate_scope must be one of {COORDINATE_SCOPES}")

    transform_supplied = "axis_transform_3x3" in config or "translation_m" in config
    if coordinate_scope == PITCH_WORLD_METRIC and not bool(config.get("already_pitch_world", False)) and not transform_supplied:
        raise ValueError(
            "PITCH_WORLD_METRIC external predictions require already_pitch_world=true "
            "or an explicit axis/translation transform"
        )
    matrix = np.asarray(config.get("axis_transform_3x3", np.eye(3)), dtype=np.float64)
    translation = np.asarray(config.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("axis_transform_3x3 must be (3,3) and translation_m must be (3,)")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("axis_transform_3x3 must be invertible")

    units = str(config.get("units", "m"))
    if units.lower() in {"model", "unitless"} and coordinate_scope != SCALE_AMBIGUOUS:
        raise ValueError("model/unitless units require coordinate_scope=SCALE_AMBIGUOUS")
    scale = _unit_scale(units)
    xyz_field = str(config.get("xyz_field", "xyz"))
    default_names = list(config.get("source_joint_names") or source.get("joint_names") or [])
    joint_map = config.get("joint_map") or {}
    if not isinstance(joint_map, Mapping) or not joint_map:
        raise ValueError("adapter requires a non-empty joint_map of destination name -> source name/index")
    unknown_destinations = sorted(set(joint_map) - set(POSE23_NAMES))
    if unknown_destinations:
        raise ValueError("Unknown METRIC_POSE_23 destinations: " + ", ".join(unknown_destinations))

    converted = []
    for record in source.get("poses", []):
        xyz = np.asarray(
            [[np.nan if value is None else value for value in row] for row in record[xyz_field]],
            dtype=np.float64,
        )
        names = list(record.get("joint_names") or default_names)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"External record must contain Jx3 '{xyz_field}', got {xyz.shape}")
        if names and len(names) != xyz.shape[0]:
            raise ValueError("source_joint_names length does not match external XYZ rows")
        name_to_index = {str(name): index for index, name in enumerate(names)}
        output = np.full((23, 3), np.nan, dtype=np.float64)
        for destination_name, source_ref in joint_map.items():
            if isinstance(source_ref, str):
                if source_ref not in name_to_index:
                    continue
                source_index = name_to_index[source_ref]
            else:
                source_index = int(source_ref)
            if not 0 <= source_index < xyz.shape[0]:
                raise ValueError(f"Source joint index {source_index} is outside record shape {xyz.shape}")
            point = xyz[source_index]
            if np.isfinite(point).all():
                output[POSE23_NAMES.index(destination_name)] = matrix @ (point * scale) + translation
        converted.append({
            "frame_index": int(record["frame_index"]),
            "track_id": str(record["track_id"]),
            "xyz23_m": _json_safe_pose(output),
        })

    result = {
        "schema_version": "metric-pose23-prediction-1.0",
        "coordinate_scope": coordinate_scope,
        "pose_schema": {"name": "METRIC_POSE_23", "count": 23, "names": list(POSE23_NAMES)},
        "model": config.get("model") or {"name": "external_model"},
        "provenance": {
            "source_file": str(source_path),
            "adapter_file": str(config_path),
            "source_units": units,
            "axis_transform_3x3": matrix.tolist(),
            "translation_m": translation.tolist(),
            "mapped_joint_count": len(joint_map),
            "checkpoint_sha256": (config.get("provenance") or {}).get("checkpoint_sha256"),
            "repository_revision": (config.get("provenance") or {}).get("repository_revision"),
            "dataset_release": (config.get("provenance") or {}).get("dataset_release"),
        },
        "poses": converted,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
