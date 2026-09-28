from __future__ import annotations

"""Stage-6 contact + production Ball-X evaluator (protocol v1.4).

This evaluator deliberately separates three questions that were previously mixed:

1. Did Stage 6 assign the correct contacting player/actor at the user-selected t0?
2. Did Stage 6 choose the correct FOOT vs NON_FOOT geometry branch?
3. Given the production output (including fallback when used), how accurate is Ball-X?

The evaluator is dataset-agnostic.  For FOOTPASS, actor GT comes from the official
(frame, team, jersey, class) annotations; a deterministic bridge/track-map is still
required to relate project track IDs to FOOTPASS identities.  FOOTPASS itself does
not provide metric ball-X GT, so Ball-X stays null unless an independent frozen GT
source supplies ``gt_x_m``/``gt_xyz_m``.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import json
import math

import numpy as np

from ..version import runtime_provenance


EXACT_REGIONS = {
    "HEAD", "TORSO", "THIGH", "KNEE", "LOWER_LEG", "FOOT", "ARM_HAND"
}


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _none_if_blank(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _normalize_region(value: Any) -> str | None:
    value = _none_if_blank(value)
    if value is None:
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ARM": "ARM_HAND", "HAND": "ARM_HAND", "ARMS": "ARM_HAND",
        "LEG": "LOWER_LEG", "LOWERLEG": "LOWER_LEG",
        "FEET": "FOOT", "FOOT_GROUND": "FOOT",
        "NONFOOT": "NON_FOOT", "NON_FOOT": "NON_FOOT",
    }
    text = aliases.get(text, text)
    if text in EXACT_REGIONS or text in {"NON_FOOT", "FOOT"}:
        return text
    return text


def _binary_region(value: Any) -> str | None:
    region = _normalize_region(value)
    if region is None:
        return None
    if region == "FOOT":
        return "FOOT"
    if region == "NON_FOOT" or region in EXACT_REGIONS:
        return "NON_FOOT"
    return None


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    manifest = Path(path).expanduser().resolve()
    suffix = manifest.suffix.lower()
    if suffix == ".csv":
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        return manifest, {"name": manifest.stem, "frozen": False}, rows
    if suffix == ".jsonl":
        rows = [dict(json.loads(line)) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        return manifest, {"name": manifest.stem, "frozen": False}, rows
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return manifest, {"name": manifest.stem, "frozen": False}, [dict(x) for x in payload]
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Manifest must be JSON {metadata,cases}, JSON list, JSONL, or CSV")
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("name", manifest.stem)
    metadata.setdefault("frozen", False)
    return manifest, metadata, [dict(x) for x in payload["cases"]]


def _resolve_relative(base: Path, value: Any) -> Path | None:
    value = _none_if_blank(value)
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = base.parent / p
    return p.resolve()


def _prediction_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    selected = state.get("selected_frame_ball") or {}
    contact = selected.get("contact") or {}
    localization = selected.get("localization") or {}
    fallback = selected.get("fallback") or {}
    assigned_track = _none_if_blank(contact.get("track_id"))
    predicted_region = _normalize_region(contact.get("region")) if assigned_track is not None else None

    x = _finite(localization.get("X_world_m"))
    if x is None:
        x = _finite(selected.get("X_world_m"))
    if x is None:
        xyz = selected.get("center_xyz_world_m")
        if isinstance(xyz, (list, tuple)) and xyz:
            x = _finite(xyz[0])

    usable = localization.get("usable_for_offside")
    if usable is None:
        usable = selected.get("usable_for_offside_longitudinal_coordinate")
    if usable is None:
        usable = x is not None

    return {
        "selected_frame": selected.get("frame_index"),
        "predicted_track": assigned_track,
        "predicted_region": predicted_region,
        "predicted_binary_region": _binary_region(predicted_region),
        "predicted_x_m": x,
        "contact_status": contact.get("status"),
        "localization_status": selected.get("status") or localization.get("status"),
        "localization_method": selected.get("method") or localization.get("selected_method"),
        "fallback_used": bool(fallback.get("used", False)),
        "fallback_reason": fallback.get("reason"),
        "usable_for_offside": bool(usable),
    }


def _load_track_map(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, tuple[str, str]] = {}
    if isinstance(payload, dict) and isinstance(payload.get("tracks"), list):
        rows = payload["tracks"]
        for row in rows:
            tid = _none_if_blank(row.get("track_id"))
            team = _none_if_blank(row.get("team"))
            jersey = _none_if_blank(row.get("jersey"))
            if tid is not None and team is not None and jersey is not None:
                mapping[str(tid)] = (str(team), str(jersey))
        return mapping
    if isinstance(payload, dict):
        for tid, row in payload.items():
            if isinstance(row, dict):
                team = _none_if_blank(row.get("team"))
                jersey = _none_if_blank(row.get("jersey"))
                if team is not None and jersey is not None:
                    mapping[str(tid)] = (str(team), str(jersey))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                mapping[str(tid)] = (str(row[0]), str(row[1]))
        return mapping
    raise ValueError(f"Unsupported track identity map schema: {path}")


def _identity_metrics(rows: Iterable[Mapping[str, Any]], gt_key: str, pred_key: str) -> dict[str, Any]:
    data = [r for r in rows if r.get(gt_key) is not None]
    if not data:
        return {
            "gt_count": 0, "assigned_count": 0, "correct_count": 0,
            "assignment_coverage": None, "assigned_precision": None, "overall_accuracy": None,
        }
    assigned = [r for r in data if r.get(pred_key) is not None]
    correct = sum(r.get(gt_key) == r.get(pred_key) for r in assigned)
    return {
        "gt_count": len(data),
        "assigned_count": len(assigned),
        "correct_count": int(correct),
        "assignment_coverage": len(assigned) / len(data),
        "assigned_precision": correct / len(assigned) if assigned else None,
        "overall_accuracy": correct / len(data),
    }


def _exact_region_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    data = [r for r in rows if r.get("gt_region") in EXACT_REGIONS]
    if not data:
        return {"gt_count": 0, "prediction_coverage": None, "accuracy": None, "accuracy_when_predicted": None}
    predicted = [r for r in data if r.get("predicted_region") in EXACT_REGIONS]
    correct_all = sum(r.get("predicted_region") == r.get("gt_region") for r in data)
    correct_pred = sum(r.get("predicted_region") == r.get("gt_region") for r in predicted)
    return {
        "gt_count": len(data),
        "predicted_count": len(predicted),
        "prediction_coverage": len(predicted) / len(data),
        "accuracy": correct_all / len(data),
        "accuracy_when_predicted": correct_pred / len(predicted) if predicted else None,
    }


def _binary_region_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    data = [r for r in rows if r.get("gt_binary_region") in {"FOOT", "NON_FOOT"}]
    if not data:
        return {
            "gt_count": 0, "prediction_coverage": None, "accuracy": None,
            "accuracy_when_predicted": None, "foot_precision": None,
            "foot_recall": None, "foot_f1": None, "confusion": None,
        }
    predicted = [r for r in data if r.get("predicted_binary_region") in {"FOOT", "NON_FOOT"}]
    tp = sum(r["gt_binary_region"] == "FOOT" and r.get("predicted_binary_region") == "FOOT" for r in data)
    fp = sum(r["gt_binary_region"] == "NON_FOOT" and r.get("predicted_binary_region") == "FOOT" for r in data)
    fn = sum(r["gt_binary_region"] == "FOOT" and r.get("predicted_binary_region") != "FOOT" for r in data)
    tn = sum(r["gt_binary_region"] == "NON_FOOT" and r.get("predicted_binary_region") == "NON_FOOT" for r in data)
    correct_all = tp + tn
    correct_pred = sum(r["gt_binary_region"] == r.get("predicted_binary_region") for r in predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "gt_count": len(data),
        "predicted_count": len(predicted),
        "prediction_coverage": len(predicted) / len(data),
        "accuracy": correct_all / len(data),
        "accuracy_when_predicted": correct_pred / len(predicted) if predicted else None,
        "foot_precision": precision,
        "foot_recall": recall,
        "foot_f1": f1,
        "confusion": {"TP_FOOT": int(tp), "FP_FOOT": int(fp), "FN_FOOT": int(fn), "TN_NON_FOOT": int(tn)},
    }


def _x_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    data = [r for r in rows if _finite(r.get("gt_x_m")) is not None]
    errors: list[float] = []
    signed: list[float] = []
    for row in data:
        gt = _finite(row.get("gt_x_m"))
        pred = _finite(row.get("predicted_x_m"))
        if gt is None or pred is None:
            continue
        errors.append(abs(pred - gt))
        signed.append(pred - gt)
    if not data:
        return {
            "gt_count": 0, "predicted_count": 0, "coverage": None,
            "mae_m": None, "rmse_m": None, "median_m": None,
            "p90_m": None, "p95_m": None, "bias_m": None,
        }
    if not errors:
        return {
            "gt_count": len(data), "predicted_count": 0, "coverage": 0.0,
            "mae_m": None, "rmse_m": None, "median_m": None,
            "p90_m": None, "p95_m": None, "bias_m": None,
        }
    arr = np.asarray(errors, dtype=float)
    signed_arr = np.asarray(signed, dtype=float)
    return {
        "gt_count": len(data),
        "predicted_count": len(errors),
        "coverage": len(errors) / len(data),
        "mae_m": float(np.mean(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr ** 2))),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.percentile(arr, 90)),
        "p95_m": float(np.percentile(arr, 95)),
        "bias_m": float(np.mean(signed_arr)),
    }


def _target_assessment(track: Mapping[str, Any], binary: Mapping[str, Any], x: Mapping[str, Any]) -> dict[str, Any]:
    def yes(value: Any, op, threshold: float) -> bool | None:
        if value is None:
            return None
        return bool(op(float(value), float(threshold)))

    import operator

    research_contact = None
    offside_contact = None
    if track.get("assigned_precision") is not None and track.get("assignment_coverage") is not None:
        research_contact = bool(track["assigned_precision"] >= 0.90 and track["assignment_coverage"] >= 0.85)
        offside_contact = bool(track["assigned_precision"] >= 0.95 and track["assignment_coverage"] >= 0.90)

    research_region = None if binary.get("accuracy") is None else bool(binary["accuracy"] >= 0.90)
    offside_region = None if binary.get("accuracy") is None else bool(
        binary["accuracy"] >= 0.95
        and (binary.get("foot_precision") is None or binary["foot_precision"] >= 0.95)
        and (binary.get("foot_recall") is None or binary["foot_recall"] >= 0.95)
    )

    strong_x = None
    offside_x = None
    stretch_x = None
    if x.get("mae_m") is not None:
        strong_x = bool(x["mae_m"] <= 0.50 and (x.get("p95_m") is None or x["p95_m"] <= 1.0))
        offside_x = bool(
            x["mae_m"] <= 0.20
            and x.get("p95_m") is not None and x["p95_m"] <= 0.50
            and x.get("coverage") is not None and x["coverage"] >= 0.95
        )
        stretch_x = bool(
            x["mae_m"] <= 0.10
            and x.get("p95_m") is not None and x["p95_m"] <= 0.20
            and x.get("coverage") is not None and x["coverage"] >= 0.98
        )

    return {
        "targets_are_project_targets_not_external_standards": True,
        "contact_research_target": research_contact,
        "contact_offside_target": offside_contact,
        "foot_nonfoot_research_target": research_region,
        "foot_nonfoot_offside_target": offside_region,
        "ball_x_strong_research_target": strong_x,
        "ball_x_offside_target": offside_x,
        "ball_x_stretch_target": stretch_x,
    }


def benchmark_contact_production_manifest(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    allow_missing_states: bool = False,
) -> dict[str, Any]:
    manifest_path, metadata, cases = _load_manifest(manifest)
    evaluated: list[dict[str, Any]] = []
    missing_states: list[str] = []
    seen: set[str] = set()

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id") or f"case_{index:05d}")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)

        state_path = _resolve_relative(manifest_path, case.get("state_json"))
        prediction = {
            "selected_frame": None, "predicted_track": None, "predicted_region": None,
            "predicted_binary_region": None, "predicted_x_m": None,
            "contact_status": "MISSING_STATE", "localization_status": None,
            "localization_method": None, "fallback_used": False,
            "fallback_reason": None, "usable_for_offside": False,
        }
        if state_path is None:
            raise ValueError(f"{case_id}: state_json is required")
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            prediction = _prediction_from_state(state)
        else:
            missing_states.append(str(state_path))
            if not allow_missing_states:
                raise FileNotFoundError(state_path)

        gt_track = _none_if_blank(case.get("gt_track_id"))
        gt_team = _none_if_blank(case.get("gt_team"))
        gt_jersey = _none_if_blank(case.get("gt_jersey"))
        gt_actor = None if gt_team is None or gt_jersey is None else (str(gt_team), str(gt_jersey))
        gt_region = _normalize_region(case.get("gt_region"))
        gt_binary = _binary_region(case.get("gt_contact_binary")) or _binary_region(gt_region)
        gt_x = _finite(case.get("gt_x_m"))
        if gt_x is None:
            xyz = case.get("gt_xyz_m")
            if isinstance(xyz, (list, tuple)) and xyz:
                gt_x = _finite(xyz[0])

        track_map_path = _resolve_relative(manifest_path, case.get("track_identity_map_json"))
        track_map = _load_track_map(track_map_path) if track_map_path is not None else {}
        pred_actor = None
        if prediction["predicted_track"] is not None and track_map:
            pred_actor = track_map.get(str(prediction["predicted_track"]))

        direct_track_correct = None if gt_track is None else str(prediction["predicted_track"]) == str(gt_track)
        actor_correct = None if gt_actor is None or pred_actor is None else pred_actor == gt_actor
        contact_correct = direct_track_correct if direct_track_correct is not None else actor_correct

        evaluated.append({
            "case_id": case_id,
            "source_dataset": case.get("source_dataset") or metadata.get("dataset"),
            "split": case.get("split") or metadata.get("split"),
            "game_key": case.get("game_key"),
            "action_class": case.get("action_class"),
            "event_frame": case.get("event_frame") or case.get("frame"),
            "state_json": str(state_path),
            "gt_track": None if gt_track is None else str(gt_track),
            "predicted_track": None if prediction["predicted_track"] is None else str(prediction["predicted_track"]),
            "gt_actor": gt_actor,
            "predicted_actor": pred_actor,
            "gt_region": gt_region,
            "predicted_region": prediction["predicted_region"],
            "gt_binary_region": gt_binary,
            "predicted_binary_region": prediction["predicted_binary_region"],
            "gt_x_m": gt_x,
            "predicted_x_m": prediction["predicted_x_m"],
            "contact_correct": contact_correct,
            "track_identity_map_json": None if track_map_path is None else str(track_map_path),
            **{k: v for k, v in prediction.items() if k not in {"predicted_track", "predicted_region", "predicted_binary_region", "predicted_x_m"}},
        })

    direct_track = _identity_metrics(evaluated, "gt_track", "predicted_track")
    actor = _identity_metrics(evaluated, "gt_actor", "predicted_actor")
    contact_primary = direct_track if direct_track["gt_count"] > 0 else actor
    contact_primary_semantics = "project_track_id" if direct_track["gt_count"] > 0 else "FOOTPASS_actor_identity"
    exact_region = _exact_region_metrics(evaluated)
    binary_region = _binary_region_metrics(evaluated)
    x_all = _x_summary(evaluated)

    by_binary = {}
    for label in ("FOOT", "NON_FOOT"):
        subset = [r for r in evaluated if r.get("gt_binary_region") == label]
        by_binary[label] = _x_summary(subset)

    by_method = {}
    for method in sorted({str(r.get("localization_method")) for r in evaluated if r.get("localization_method")}):
        by_method[method] = _x_summary([r for r in evaluated if str(r.get("localization_method")) == method])

    by_fallback = {
        "fallback_false": _x_summary([r for r in evaluated if not bool(r.get("fallback_used"))]),
        "fallback_true": _x_summary([r for r in evaluated if bool(r.get("fallback_used"))]),
    }
    contact_conditioned = {
        "correct_contact": _x_summary([r for r in evaluated if r.get("contact_correct") is True]),
        "wrong_or_unassigned_contact": _x_summary([r for r in evaluated if r.get("contact_correct") is False]),
    }

    status = "COMPLETE" if not missing_states else "PARTIAL"
    frozen = bool(metadata.get("frozen", False))
    readiness = {
        "contact_track_or_actor": contact_primary.get("gt_count", 0) > 0,
        "foot_nonfoot": binary_region.get("gt_count", 0) > 0,
        "production_ball_x": x_all.get("gt_count", 0) > 0,
    }
    scientific_claim_ready = bool(frozen and status == "COMPLETE" and any(readiness.values()))

    metrics = {
        "contact_primary_semantics": contact_primary_semantics,
        "contact_primary": contact_primary,
        "direct_track_id": direct_track,
        "actor_identity": actor,
        "exact_region": exact_region,
        "foot_vs_nonfoot": binary_region,
        "production_ball_x": x_all,
        "ball_x_by_gt_branch": by_binary,
        "ball_x_by_localization_method": by_method,
        "ball_x_by_fallback": by_fallback,
        "ball_x_by_contact_correctness": contact_conditioned,
    }
    assessment = _target_assessment(contact_primary, binary_region, x_all)

    report = {
        "schema_version": "stage6-contact-production-benchmark-1.0",
        "runtime_provenance": runtime_provenance(),
        "status": status,
        "scientific_claim_ready": scientific_claim_ready,
        "metric_readiness": readiness,
        "manifest": {
            "path": str(manifest_path),
            "metadata": metadata,
            "cases": len(evaluated),
            "missing_state_files": len(missing_states),
        },
        "metrics": metrics,
        "project_target_assessment": assessment,
        "limitations": {
            "footpass_ball_x": "FOOTPASS does not supply independent metric ball-X GT; production Ball-X requires another frozen GT source.",
            "footpass_track_bridge": "FOOTPASS actor identity is team+jersey; project track-ID scoring needs gt_track_id or a deterministic track identity map.",
            "region_gt": "Only score FOOT/non-FOOT where independent region GT is present; do not infer Pass/Cross/Shot as FOOT automatically.",
        },
    }

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage6_contact_production_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if evaluated:
        serial_rows = []
        for row in evaluated:
            x = dict(row)
            if isinstance(x.get("gt_actor"), tuple):
                x["gt_actor"] = ":".join(x["gt_actor"])
            if isinstance(x.get("predicted_actor"), tuple):
                x["predicted_actor"] = ":".join(x["predicted_actor"])
            serial_rows.append(x)
        with (out / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(serial_rows[0].keys()))
            writer.writeheader(); writer.writerows(serial_rows)

    lines = [
        "# Stage 6 contact + production Ball-X benchmark", "",
        f"Status: **{status}**", "",
        f"Scientific-claim ready: **{scientific_claim_ready}**", "",
        "| Metric | Value |", "|---|---:|",
        f"| Contact assignment coverage | {contact_primary.get('assignment_coverage')} |",
        f"| Contact assigned precision | {contact_primary.get('assigned_precision')} |",
        f"| Contact overall accuracy | {contact_primary.get('overall_accuracy')} |",
        f"| FOOT/non-FOOT accuracy | {binary_region.get('accuracy')} |",
        f"| FOOT precision | {binary_region.get('foot_precision')} |",
        f"| FOOT recall | {binary_region.get('foot_recall')} |",
        f"| Production Ball-X MAE (m) | {x_all.get('mae_m')} |",
        f"| Production Ball-X P95 (m) | {x_all.get('p95_m')} |",
        f"| Production Ball-X coverage | {x_all.get('coverage')} |",
        "", "## Project target assessment", "", "```json",
        json.dumps(assessment, indent=2), "```", "",
        "> Missing GT stays unevaluated; it is never converted into PASS.",
    ]
    (out / "stage6_contact_production_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
