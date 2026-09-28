from __future__ import annotations

"""Aggregate remaining Stage-6 TBD benchmarks into one readiness report."""

from pathlib import Path
from typing import Any, Mapping
import json


GROUND_PRED = "GROUND_PLANE_PRED_CENTER"
SIZE_E2E = "SIZE_PRIOR_TOP1"


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _metric(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _bool_status(value: bool | None) -> str:
    if value is None:
        return "NOT_EVALUATED"
    return "PASS" if value else "BELOW_TARGET"


def _ground_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"status": "NOT_EVALUATED"}
    primary = report.get("primary_near_ground_proxy") or {}
    common = primary.get("common_valid_frames") or {}
    metrics = common.get("metrics") or {}
    gp = metrics.get(GROUND_PRED) or {}
    sp = metrics.get(SIZE_E2E) or {}
    mae = gp.get("mae_m")
    p95 = gp.get("p95_m")
    coverage = gp.get("coverage")
    base_mae = sp.get("mae_m")
    improvement = None
    if mae is not None and base_mae not in (None, 0):
        improvement = (float(base_mae) - float(mae)) / float(base_mae)
    feasibility = None if improvement is None else bool(improvement >= 0.50)
    strong = None if mae is None else bool(float(mae) <= 0.50 and (p95 is None or float(p95) <= 1.0))
    offside = None if mae is None or p95 is None or coverage is None else bool(
        float(mae) <= 0.20 and float(p95) <= 0.50 and float(coverage) >= 0.95
    )
    stretch = None if mae is None or p95 is None or coverage is None else bool(
        float(mae) <= 0.10 and float(p95) <= 0.20 and float(coverage) >= 0.98
    )
    return {
        "status": "EVALUATED",
        "scientific_diagnostic_ready": report.get("scientific_diagnostic_ready"),
        "records": common.get("records"),
        "ball_x_mae_m": mae,
        "ball_x_p95_m": p95,
        "coverage": coverage,
        "size_prior_common_mae_m": base_mae,
        "relative_improvement_vs_size_prior": improvement,
        "feasibility_50pct_improvement": _bool_status(feasibility),
        "strong_research_target": _bool_status(strong),
        "offside_oriented_target": _bool_status(offside),
        "stretch_target": _bool_status(stretch),
    }


def _first_x_metric(block: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
    for key in ("BLE_X_MAE_m", "Ball_X_MAE_m", "MAE_X_m", "X_MAE_m"):
        if block.get(key) is not None:
            return key, block
    return None, None


def _extract_method_x(method_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    block = method_payload.get("all") if isinstance(method_payload.get("all"), Mapping) else method_payload
    if not isinstance(block, Mapping):
        return None
    mae = None
    for key in ("BLE_X_MAE_m", "Ball_X_MAE_m", "MAE_X_m", "X_MAE_m"):
        if block.get(key) is not None:
            mae = float(block[key]); break
    if mae is None:
        return None
    p95 = None
    for key in ("BLE_X_P95_m", "Ball_X_P95_m", "P95_X_m", "X_P95_m"):
        if block.get(key) is not None:
            p95 = float(block[key]); break
    coverage = block.get("coverage")
    return {"mae_m": mae, "p95_m": p95, "coverage": coverage}


def _issia_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"status": "NOT_EVALUATED"}
    methods = report.get("methods") or {}
    baseline = _extract_method_x(methods.get("V03_SIZE_PRIOR") or {})
    temporal = _extract_method_x(methods.get("V04_TEMPORAL") or {})
    hybrid = _extract_method_x(methods.get("V044_HYBRID") or {})
    if baseline is None:
        return {
            "status": "REPORT_LACKS_X_METRICS",
            "note": "ISSIA report exists but no Ball-X metric key was found; keep 3D metrics as secondary only.",
        }
    candidates = [(name, m) for name, m in (("V04_TEMPORAL", temporal), ("V044_HYBRID", hybrid)) if m is not None]
    if not candidates:
        return {"status": "NO_TEMPORAL_X_RESULT", "baseline": baseline}
    best_name, best = min(candidates, key=lambda item: float(item[1]["mae_m"]))
    improvement = (baseline["mae_m"] - best["mae_m"]) / baseline["mae_m"] if baseline["mae_m"] else None
    p95_not_worse = True
    if baseline.get("p95_m") is not None and best.get("p95_m") is not None:
        p95_not_worse = bool(best["p95_m"] <= baseline["p95_m"])
    coverage_ok = best.get("coverage") is not None and float(best["coverage"]) >= 0.95
    research = None if improvement is None else bool(improvement >= 0.10 and p95_not_worse and coverage_ok)
    return {
        "status": "EVALUATED",
        "baseline": baseline,
        "best_method": best_name,
        "best": best,
        "relative_mae_improvement": improvement,
        "p95_not_worse": p95_not_worse,
        "coverage_ge_0_95": coverage_ok,
        "research_value_target_10pct_improvement": _bool_status(research),
    }


def _contact_status(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"status": "NOT_EVALUATED"}
    metrics = report.get("metrics") or {}
    contact = metrics.get("contact_primary") or {}
    binary = metrics.get("foot_vs_nonfoot") or {}
    x = metrics.get("production_ball_x") or {}
    assessment = report.get("project_target_assessment") or {}
    return {
        "status": "EVALUATED",
        "scientific_claim_ready": report.get("scientific_claim_ready"),
        "metric_readiness": report.get("metric_readiness"),
        "contact": {
            "assignment_coverage": contact.get("assignment_coverage"),
            "assigned_precision": contact.get("assigned_precision"),
            "overall_accuracy": contact.get("overall_accuracy"),
            "research_target": _bool_status(assessment.get("contact_research_target")),
            "offside_target": _bool_status(assessment.get("contact_offside_target")),
        },
        "foot_nonfoot": {
            "accuracy": binary.get("accuracy"),
            "foot_precision": binary.get("foot_precision"),
            "foot_recall": binary.get("foot_recall"),
            "research_target": _bool_status(assessment.get("foot_nonfoot_research_target")),
            "offside_target": _bool_status(assessment.get("foot_nonfoot_offside_target")),
        },
        "production_ball_x": {
            "mae_m": x.get("mae_m"),
            "p95_m": x.get("p95_m"),
            "coverage": x.get("coverage"),
            "strong_research_target": _bool_status(assessment.get("ball_x_strong_research_target")),
            "offside_target": _bool_status(assessment.get("ball_x_offside_target")),
            "stretch_target": _bool_status(assessment.get("ball_x_stretch_target")),
        },
    }


def build_tbd_status_report(
    *,
    output_dir: str | Path,
    ground_report: str | Path | None = None,
    issia_report: str | Path | None = None,
    contact_report: str | Path | None = None,
) -> dict[str, Any]:
    ground = _ground_status(_load(ground_report))
    issia = _issia_status(_load(issia_report))
    contact = _contact_status(_load(contact_report))
    report = {
        "schema_version": "stage6-tbd-readiness-1.0",
        "targets_semantics": "Project research/offside targets, not external benchmark standards.",
        "ground_plane_ball_x": ground,
        "temporal_hybrid_ball_x": issia,
        "contact_and_production": contact,
        "remaining_tbd": [
            name for name, block in (
                ("ground_plane_ball_x", ground),
                ("temporal_hybrid_ball_x", issia),
                ("contact_and_production", contact),
            ) if block.get("status") == "NOT_EVALUATED"
        ],
    }
    out = Path(output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "stage6_tbd_readiness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stage 6 remaining-TBD readiness", "",
        "| Area | Status | Key result |", "|---|---|---|",
        f"| Ground-plane Ball-X | {ground.get('status')} | MAE={ground.get('ball_x_mae_m')}, P95={ground.get('ball_x_p95_m')} |",
        f"| Temporal/hybrid | {issia.get('status')} | best={issia.get('best_method')}, improvement={issia.get('relative_mae_improvement')} |",
        f"| Contact/production | {contact.get('status')} | ready={contact.get('metric_readiness')} |",
        "", "```json", json.dumps(report, indent=2), "```",
    ]
    (out / "stage6_tbd_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
