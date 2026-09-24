from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .core import build_game_state_context


def _set_f1(pred: Iterable[str], gt: Iterable[str]) -> Tuple[int, int, int]:
    p, g = set(pred), set(gt)
    return len(p & g), len(p - g), len(g - p)


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def _resolve(base: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def evaluate_manifest(manifest_path: str, output_dir: str) -> Dict[str, Any]:
    mpath = Path(manifest_path).resolve()
    payload = json.loads(mpath.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    if not isinstance(cases, list):
        raise ValueError("Manifest must contain a 'cases' list or be a list.")

    rows: List[Dict[str, Any]] = []
    counts = {
        "cases": 0,
        "attacking_team_correct": 0,
        "direction_correct": 0,
        "attacker_exact": 0,
        "opponent_exact": 0,
        "referee_exact": 0,
        "full_exact": 0,
        "invariant_pass": 0,
        "pred_valid": 0,
        "gt_valid": 0,
    }
    tp = fp = fn = 0

    for case in cases:
        if not isinstance(case, dict):
            continue
        counts["cases"] += 1
        base = mpath.parent
        ctx = build_game_state_context(
            _resolve(base, case["stage1"]),
            _resolve(base, case["stage5"]),
            _resolve(base, case["stage6"]),
        ).to_dict()
        gt = case.get("gt", {})
        pred_sets = ctx.get("sets", {})
        gt_sets = gt.get("sets", {})

        pred_valid = ctx.get("status") == "VALID"
        gt_valid = gt.get("status", "VALID") == "VALID"
        counts["pred_valid"] += int(pred_valid)
        counts["gt_valid"] += int(gt_valid)

        team_ok = ctx.get("attacking_team_id") == gt.get("attacking_team_id")
        dir_ok = (ctx.get("attack_direction") or {}).get("s") == gt.get("attack_direction_s")
        att_ok = set(pred_sets.get("attackers", [])) == set(gt_sets.get("attackers", []))
        opp_ok = set(pred_sets.get("opponents", [])) == set(gt_sets.get("opponents", []))
        ref_ok = set(pred_sets.get("referees_excluded", [])) == set(gt_sets.get("referees_excluded", []))
        inv_ok = bool((ctx.get("diagnostics") or {}).get("invariant_pass", False))
        full_ok = (
            ctx.get("status") == gt.get("status", "VALID")
            and team_ok and dir_ok and att_ok and opp_ok and ref_ok
        )

        counts["attacking_team_correct"] += int(team_ok)
        counts["direction_correct"] += int(dir_ok)
        counts["attacker_exact"] += int(att_ok)
        counts["opponent_exact"] += int(opp_ok)
        counts["referee_exact"] += int(ref_ok)
        counts["full_exact"] += int(full_ok)
        counts["invariant_pass"] += int(inv_ok)

        for key in ("attackers", "opponents"):
            a, b, c = _set_f1(pred_sets.get(key, []), gt_sets.get(key, []))
            tp += a; fp += b; fn += c

        rows.append({
            "case_id": case.get("case_id"),
            "status_pred": ctx.get("status"),
            "status_gt": gt.get("status", "VALID"),
            "attacking_team_correct": team_ok,
            "direction_correct": dir_ok,
            "attackers_exact": att_ok,
            "opponents_exact": opp_ok,
            "referees_exact": ref_ok,
            "invariant_pass": inv_ok,
            "full_exact": full_ok,
            "reasons": ctx.get("reasons", []),
        })

    n = counts["cases"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    micro_f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)

    report = {
        "schema_version": "1.0",
        "stage_version": "stage7-game-state-0.1.0",
        "status": "COMPLETE",
        "counts": counts,
        "metrics": {
            "toucher_handoff_validity": _safe_div(counts["pred_valid"], n),
            "attacking_team_accuracy": _safe_div(counts["attacking_team_correct"], n),
            "attack_direction_accuracy": _safe_div(counts["direction_correct"], n),
            "attacker_set_exact_match": _safe_div(counts["attacker_exact"], n),
            "opponent_set_exact_match": _safe_div(counts["opponent_exact"], n),
            "referee_exclusion_exact_match": _safe_div(counts["referee_exact"], n),
            "set_micro_precision": precision,
            "set_micro_recall": recall,
            "set_micro_f1": micro_f1,
            "full_game_state_exact_match": _safe_div(counts["full_exact"], n),
            "invariant_pass_rate": _safe_div(counts["invariant_pass"], n),
        },
        "rows": rows,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def oracle_fixtures() -> List[Dict[str, Any]]:
    return [
        {
            "name": "right_half_team0",
            "stage1": {"view": {"centre_ray_pitch_hit_m": [30.0, 0.0, 0.0]}},
            "stage5": {"players": [
                {"track_id": "03", "team_id": 0, "role": "PLAYER"},
                {"track_id": "05", "team_id": 0, "role": "GOALKEEPER"},
                {"track_id": "07", "team_id": 1, "role": "PLAYER"},
                {"track_id": "11", "team_id": 1, "role": "GOALKEEPER"},
                {"track_id": "18", "role": "REFEREE"},
            ]},
            "stage6": {"selected_frame_ball": {"frame_index": 100, "contact": {"track_id": "03"}}},
            "gt": {"status": "VALID", "team": 0, "s": 1, "attackers": {"03", "05"}, "opponents": {"07", "11"}, "refs": {"18"}},
        },
        {
            "name": "left_half_team1",
            "stage1": {"view": {"centre_ray_pitch_hit_m": [-28.0, 4.0, 0.0]}},
            "stage5": {"by_track": {
                "1": {"team_id": 0, "role": "PLAYER"},
                "2": {"team_id": 1, "role": "PLAYER"},
                "3": {"team_id": 1, "role": "GOALKEEPER"},
                "9": {"role": "REFEREE"},
            }},
            "stage6": {"contact_track_id": 2, "frame_index": 200},
            "gt": {"status": "VALID", "team": 1, "s": -1, "attackers": {"2", "3"}, "opponents": {"1"}, "refs": {"9"}},
        },
        {
            "name": "missing_toucher",
            "stage1": {"view": {"centre_ray_pitch_hit_m": [20.0, 0.0, 0.0]}},
            "stage5": {"players": [{"track_id": "1", "team_id": 0}, {"track_id": "2", "team_id": 1}]},
            "stage6": {},
            "gt": {"status": "UNRESOLVED", "reason": "MISSING_CONTACT_TRACK_ID"},
        },
        {
            "name": "ambiguous_centre_ray",
            "stage1": {"view": {"centre_ray_pitch_hit_m": [0.0, 0.0, 0.0]}},
            "stage5": {"players": [{"track_id": "1", "team_id": 0}, {"track_id": "2", "team_id": 1}]},
            "stage6": {"contact_track_id": "1"},
            "gt": {"status": "UNRESOLVED", "reason": "CENTRE_RAY_X_AMBIGUOUS"},
        },
        {
            "name": "referee_cannot_be_toucher",
            "stage1": {"view": {"centre_ray_pitch_hit_m": [20.0, 0.0, 0.0]}},
            "stage5": {"players": [
                {"track_id": "1", "team_id": 0},
                {"track_id": "2", "team_id": 1},
                {"track_id": "9", "role": "REFEREE"},
            ]},
            "stage6": {"contact_track_id": "9"},
            "gt": {"status": "UNRESOLVED", "reason": "TOUCHER_IS_REFEREE"},
        },
    ]


def run_oracle_benchmark(output_dir: str) -> Dict[str, Any]:
    rows = []
    all_ok = True
    for fixture in oracle_fixtures():
        ctx = build_game_state_context(fixture["stage1"], fixture["stage5"], fixture["stage6"]).to_dict()
        gt = fixture["gt"]
        if gt["status"] == "VALID":
            ok = (
                ctx["status"] == gt["status"]
                and ctx["attacking_team_id"] == gt["team"]
                and ctx["attack_direction"]["s"] == gt["s"]
                and set(ctx["sets"]["attackers"]) == gt["attackers"]
                and set(ctx["sets"]["opponents"]) == gt["opponents"]
                and set(ctx["sets"]["referees_excluded"]) == gt["refs"]
                and ctx["diagnostics"]["invariant_pass"]
            )
        else:
            ok = ctx["status"] == gt["status"] and gt["reason"] in ctx.get("reasons", [])
        all_ok &= ok
        rows.append({"fixture": fixture["name"], "pass": ok, "prediction": ctx})
    report = {
        "status": "PASS" if all_ok else "FAIL",
        "oracle_logic_accuracy": sum(int(r["pass"]) for r in rows) / len(rows),
        "target": 1.0,
        "rows": rows,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_oracle_benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
