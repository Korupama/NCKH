from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import build_offside_reference


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def _resolve(base: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def evaluate_manifest(manifest_path: str, output_dir: str) -> Dict[str, Any]:
    mpath = Path(manifest_path).resolve()
    payload = json.loads(mpath.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    rows: List[Dict[str, Any]] = []
    den = {"status": 0, "second_last": 0, "reference_source": 0, "reference_x": 0, "ranking": 0}
    num = {k: 0 for k in den}
    x_abs_errors: List[float] = []
    pred_valid = 0

    for case in cases:
        if not isinstance(case, dict):
            continue
        state = build_offside_reference(
            _resolve(mpath.parent, case["stage4"]),
            _resolve(mpath.parent, case["stage6"]),
            _resolve(mpath.parent, case["stage7"]),
        ).to_dict()
        gt = case.get("gt") or {}
        pred_valid += int(state["status"] == "VALID")
        row = {"case_id": case.get("case_id"), "pred_status": state["status"], "reasons": state.get("reasons", [])}

        if "status" in gt:
            den["status"] += 1
            ok = state["status"] == gt["status"]
            num["status"] += int(ok); row["status_correct"] = ok
        if "second_last_track_id" in gt:
            den["second_last"] += 1
            pred = state.get("second_last_opponent") or {}
            candidates = set(pred.get("candidate_track_ids") or ([] if pred.get("track_id") is None else [pred.get("track_id")]))
            ok = str(gt["second_last_track_id"]) in candidates
            num["second_last"] += int(ok); row["second_last_correct"] = ok
        if "reference_source" in gt:
            den["reference_source"] += 1
            ok = (state.get("reference") or {}).get("source") == gt["reference_source"]
            num["reference_source"] += int(ok); row["reference_source_correct"] = ok
        if "reference_x_m" in gt:
            den["reference_x"] += 1
            px = (state.get("reference") or {}).get("X_world_m")
            if px is not None:
                err = abs(float(px) - float(gt["reference_x_m"]))
                x_abs_errors.append(err)
                num["reference_x"] += 1
                row["reference_x_abs_error_m"] = err
        if "opponent_order" in gt:
            den["ranking"] += 1
            pred_order = [str(r["track_id"]) for r in state.get("opponent_ranking") or []]
            gt_order = [str(x) for x in gt["opponent_order"]]
            ok = pred_order == gt_order
            num["ranking"] += int(ok); row["ranking_exact"] = ok
        rows.append(row)

    n = len(cases)
    report = {
        "schema_version": "stage8-evaluation-report-1.0",
        "stage_version": "stage8-offside-reference-0.1.0",
        "status": "COMPLETE",
        "cases": n,
        "pred_valid_rate": _safe_div(pred_valid, n),
        "abstention_rate": None if n == 0 else 1.0 - pred_valid / n,
        "denominators": den,
        "metrics": {
            "status_accuracy": _safe_div(num["status"], den["status"]),
            "second_last_opponent_agreement": _safe_div(num["second_last"], den["second_last"]),
            "reference_source_agreement": _safe_div(num["reference_source"], den["reference_source"]),
            "opponent_ranking_exact_match": _safe_div(num["ranking"], den["ranking"]),
            "reference_x_mae_m": None if not x_abs_errors else sum(x_abs_errors) / len(x_abs_errors),
        },
        "rows": rows,
        "notes": "Metrics with zero labeled denominator are NOT_EVALUATED. This evaluator does not invent GT from Stage-8 predictions.",
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "stage8_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
