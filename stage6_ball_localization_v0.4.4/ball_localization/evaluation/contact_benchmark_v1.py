from __future__ import annotations

"""Contact benchmark for Stage 6 v0.5.x selected-frame output.

The evaluator is dataset-agnostic on purpose.  A canonical frozen manifest allows
FOOTPASS-derived annotations or a project-held contact set to be scored without
hard-coding a private directory layout.  Missing predictions count against coverage
and contact accuracy; unavailable GT metrics remain null.
"""

from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import json
import math

import numpy as np

from ..version import runtime_provenance


def _finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(path).expanduser().resolve()
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        return {"name": manifest_path.stem, "frozen": False}, rows
    if suffix == ".jsonl":
        rows = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return {"name": manifest_path.stem, "frozen": False}, rows
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"name": manifest_path.stem, "frozen": False}, [dict(x) for x in payload]
    if not isinstance(payload, dict):
        raise ValueError("Contact manifest must be a JSON object/list, JSONL or CSV")
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("JSON contact manifest object must contain a 'cases' list")
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("name", manifest_path.stem)
    metadata.setdefault("frozen", False)
    return metadata, [dict(x) for x in rows]


def _resolve_state_path(manifest_path: Path, raw: Any) -> Path:
    if raw is None or not str(raw).strip():
        raise ValueError("Each contact case requires state_json")
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = manifest_path.parent / p
    return p.resolve()


def _prediction_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    selected = state.get("selected_frame_ball") or {}
    contact = selected.get("contact") or {}
    localization = selected.get("localization") or {}
    assigned_track = contact.get("track_id")
    # A nearest body region without an assigned toucher is not scored as a region
    # prediction.  This prevents AMBIGUOUS/NO_CONTACT_EVIDENCE cases from receiving
    # accidental region credit.
    predicted_region = contact.get("region") if assigned_track is not None else None
    predicted_x = _finite_number(localization.get("X_world_m"))
    if predicted_x is None:
        predicted_x = _finite_number(selected.get("X_world_m"))
    if predicted_x is None:
        xyz = selected.get("center_xyz_world_m")
        if isinstance(xyz, (list, tuple)) and xyz:
            predicted_x = _finite_number(xyz[0])
    return {
        "selected_frame": selected.get("frame_index"),
        "predicted_track": assigned_track,
        "predicted_region": predicted_region,
        "predicted_x": predicted_x,
        "contact_status": contact.get("status"),
        "localization_status": selected.get("status"),
        "localization_method": selected.get("method") or localization.get("selected_method"),
    }


def _accuracy(rows: Iterable[Mapping[str, Any]], gt_key: str, pred_key: str) -> tuple[int, int, float | None, float | None]:
    evaluable = [r for r in rows if r.get(gt_key) not in (None, "")]
    if not evaluable:
        return 0, 0, None, None
    predicted = [r for r in evaluable if r.get(pred_key) not in (None, "")]
    correct = sum(str(r.get(pred_key)) == str(r.get(gt_key)) for r in evaluable)
    return len(evaluable), correct, correct / len(evaluable), len(predicted) / len(evaluable)


def _x_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    gt_rows = [r for r in rows if _finite_number(r.get("gt_x_m")) is not None]
    errors = []
    for row in gt_rows:
        gt = _finite_number(row.get("gt_x_m"))
        pred = _finite_number(row.get("predicted_x"))
        if gt is not None and pred is not None:
            errors.append(abs(pred - gt))
    if not gt_rows:
        return {
            "gt_count": 0,
            "coverage": None,
            "Ball_X_MAE_m": None,
            "Ball_X_median_m": None,
            "Ball_X_P90_m": None,
            "Ball_X_P95_m": None,
        }
    arr = np.asarray(errors, float)
    return {
        "gt_count": len(gt_rows),
        "coverage": len(errors) / len(gt_rows),
        "Ball_X_MAE_m": float(np.mean(arr)) if len(arr) else None,
        "Ball_X_median_m": float(np.median(arr)) if len(arr) else None,
        "Ball_X_P90_m": float(np.percentile(arr, 90)) if len(arr) else None,
        "Ball_X_P95_m": float(np.percentile(arr, 95)) if len(arr) else None,
    }


def benchmark_contact_manifest(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    allow_missing_states: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest).expanduser().resolve()
    metadata, cases = _load_manifest(manifest_path)
    evaluated: list[dict[str, Any]] = []
    missing_states: list[str] = []
    seen_case_ids: set[str] = set()

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id") or f"case_{index:05d}")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate contact case_id: {case_id}")
        seen_case_ids.add(case_id)
        state_path = _resolve_state_path(manifest_path, case.get("state_json"))
        prediction = {
            "selected_frame": None,
            "predicted_track": None,
            "predicted_region": None,
            "predicted_x": None,
            "contact_status": "MISSING_STATE",
            "localization_status": None,
            "localization_method": None,
        }
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            prediction = _prediction_from_state(state)
        else:
            missing_states.append(str(state_path))
            if not allow_missing_states:
                raise FileNotFoundError(state_path)

        gt_x = _finite_number(case.get("gt_x_m"))
        evaluated.append({
            "case_id": case_id,
            "source_dataset": case.get("source_dataset") or metadata.get("dataset"),
            "split": case.get("split") or metadata.get("split"),
            "action_class": case.get("action_class"),
            "state_json": str(state_path),
            "gt_track": case.get("gt_track_id"),
            "gt_region": case.get("gt_region"),
            "gt_x_m": gt_x,
            **prediction,
        })

    track_n, track_correct, track_acc, track_cov = _accuracy(evaluated, "gt_track", "predicted_track")
    region_n, region_correct, region_acc, region_cov = _accuracy(evaluated, "gt_region", "predicted_region")
    x_metrics = _x_metrics(evaluated)

    per_action = {}
    for action in sorted({str(r["action_class"]) for r in evaluated if r.get("action_class") not in (None, "")}):
        subset = [r for r in evaluated if str(r.get("action_class")) == action]
        n, correct, acc, cov = _accuracy(subset, "gt_track", "predicted_track")
        per_action[action] = {
            "cases": len(subset),
            "contact_track_gt_count": n,
            "contact_track_accuracy": acc,
            "contact_assignment_coverage": cov,
        }

    frozen = bool(metadata.get("frozen", False))
    status = "COMPLETE" if not missing_states else "PARTIAL"
    scientific_claim_ready = bool(frozen and status == "COMPLETE" and len(evaluated) > 0)
    report = {
        "schema_version": "stage6-contact-benchmark-1.0",
        "runtime_provenance": runtime_provenance(),
        "status": status,
        "scientific_claim_ready": scientific_claim_ready,
        "manifest": {
            "path": str(manifest_path),
            "metadata": metadata,
            "cases": len(evaluated),
            "missing_state_files": len(missing_states),
        },
        "metrics": {
            "contact_track_gt_count": track_n,
            "contact_track_correct": track_correct,
            "contact_track_accuracy": track_acc,
            "contact_assignment_coverage": track_cov,
            "contact_region_gt_count": region_n,
            "contact_region_correct": region_correct,
            "contact_region_accuracy": region_acc,
            "contact_region_prediction_coverage": region_cov,
            **x_metrics,
        },
        "per_action_class": per_action,
        "interpretation": {
            "recommended_external_source": "FOOTPASS validation events after deterministic (frame, team, jersey) -> Stage track mapping",
            "hidden_test_note": "FOOTPASS challenge/test GT is hidden; local contact-track scoring should use validation or a separately frozen project set.",
            "qa_policy": "Random project replay frames without independent GT are regression/QA only.",
        },
    }

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "contact_benchmark_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if evaluated:
        fieldnames = list(evaluated[0].keys())
        with (out / "contact_cases.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluated)

    m = report["metrics"]
    lines = [
        "# Stage 6 contact benchmark",
        "",
        f"Status: **{status}**",
        "",
        f"Scientific-claim ready: **{scientific_claim_ready}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Contact track accuracy | {m['contact_track_accuracy']} |",
        f"| Contact assignment coverage | {m['contact_assignment_coverage']} |",
        f"| Contact region accuracy | {m['contact_region_accuracy']} |",
        f"| Ball-X MAE (m) | {m['Ball_X_MAE_m']} |",
        f"| Ball-X P90 (m) | {m['Ball_X_P90_m']} |",
        f"| Ball-X P95 (m) | {m['Ball_X_P95_m']} |",
        f"| Ball-X coverage | {m['coverage']} |",
    ]
    (out / "contact_benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_manifest_template(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "name": "stage6-contact-eval",
            "dataset": "FOOTPASS-derived-or-project-frozen",
            "split": "validation",
            "frozen": False,
            "note": "Set frozen=true only after annotations and case selection are locked before final evaluation.",
        },
        "cases": [
            {
                "case_id": "example_0001",
                "source_dataset": "FOOTPASS",
                "split": "validation",
                "action_class": "Pass",
                "state_json": "relative/path/to/ball_trajectory_state.json",
                "gt_track_id": "track_006",
                "gt_region": "FOOT",
                "gt_x_m": None,
            }
        ],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
