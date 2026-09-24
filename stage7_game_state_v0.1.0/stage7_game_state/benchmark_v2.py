from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .benchmark import run_oracle_benchmark
from .core import build_game_state_context

SCHEMA_VERSION = "stage7-benchmark-v2.0"
STAGE_VERSION = "stage7-game-state-0.1.0"

# These are reference measurements from related third-party tasks. They are not
# treated as directly comparable Stage-7 scores. The purpose is to keep internal
# engineering targets in a realistic range and to preserve provenance.
THIRD_PARTY_REFERENCES: List[Dict[str, Any]] = [
    {
        "id": "soccernet_gsr_2024_baseline",
        "task": "SoccerNet Game State Reconstruction",
        "year": 2024,
        "dataset": "SoccerNet-GSR",
        "metric": "GS-HOTA",
        "value": 22.26,
        "scope": "full baseline",
        "relevance": "role/team/track context; holistic metric, not Stage-7 accuracy",
        "source": "Somers et al., CVPRW 2024",
        "url": "https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.html",
    },
    {
        "id": "soccernet_gsr_2024_team_side_oracle_ablation",
        "task": "SoccerNet Game State Reconstruction",
        "year": 2024,
        "dataset": "SoccerNet-GSR",
        "metric": "GS-HOTA",
        "value": 92.00,
        "scope": "team-side module with other modules replaced by ground-truth oracles",
        "relevance": "closest published reference for team-side semantics; still not direct team accuracy",
        "source": "Somers et al., CVPRW 2024, Table 2",
        "url": "https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf",
    },
    {
        "id": "soccernet_gsr_2025_challenge_top",
        "task": "SoccerNet Game State Reconstruction Challenge",
        "year": 2025,
        "dataset": "SoccerNet-GSR",
        "metric": "GS-HOTA",
        "value": 63.90,
        "scope": "top challenge submission",
        "relevance": "modern full-pipeline reference; holistic and stricter than Stage 7",
        "source": "SoccerNet 2025 Challenges Results",
        "url": "https://repository.kaust.edu.sa/bitstreams/25d37341-ad3e-4b74-8168-2286c4c4ac4b/download",
    },
    {
        "id": "broadcast2pitch_2026",
        "task": "Game State Reconstruction",
        "year": 2026,
        "dataset": "SoccerNet-GSR test",
        "metric": "GS-HOTA",
        "value": 61.48,
        "secondary": {"GS-AssA": 78.00, "IDF1": 64.20},
        "scope": "full Broadcast2Pitch pipeline",
        "relevance": "recent role/team/identity-aware GSR reference",
        "source": "Oo et al., WACV 2026",
        "url": "https://openaccess.thecvf.com/content/WACV2026/html/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.html",
    },
    {
        "id": "broadcast2pitch_2026_attribute_ablation",
        "task": "GSR attribute ablation",
        "year": 2026,
        "dataset": "SoccerNet-GSR test",
        "metric": "GS-HOTA",
        "values": {
            "pitch_only": 79.52,
            "pitch_plus_role": 79.20,
            "pitch_plus_team": 73.58,
            "pitch_plus_role_plus_team": 64.70,
        },
        "scope": "attribute ablation",
        "relevance": "shows team errors are materially harder than role errors in this system",
        "source": "Oo et al., WACV 2026, Table 7",
        "url": "https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf",
    },
    {
        "id": "footpass_2026_official_validation_baselines",
        "task": "Player-Centric Ball-Action Spotting",
        "year": 2026,
        "dataset": "FOOTPASS validation",
        "metric": "Micro F1",
        "values": {"TAAD": 0.359, "TAAD+GNN": 0.521, "TAAD+DST": 0.675},
        "scope": "who/what/when joint event attribution",
        "relevance": "related lower-bound reference for actor/team attribution; harder than manual-t0 Stage 7",
        "source": "official FOOTPASS baselines as reported by Wang et al. 2026",
        "url": "https://arxiv.org/abs/2608.01696",
    },
    {
        "id": "footpass_2026_me_dst",
        "task": "Player-Centric Ball-Action Spotting",
        "year": 2026,
        "dataset": "FOOTPASS validation",
        "metric": "Micro F1",
        "value": 0.778,
        "secondary": {"precision": 0.792, "recall": 0.765},
        "scope": "ME-DST",
        "relevance": "strong recent actor-aware context reference; not directly comparable to Stage 7",
        "source": "Wang et al., arXiv 2026",
        "url": "https://arxiv.org/abs/2608.01696",
    },
]

# Internal targets informed by the references above. These are deliberately
# labeled engineering gates, not public benchmark standards.
ENGINEERING_PROFILES: Dict[str, Dict[str, Any]] = {
    "pilot-minimum": {
        "description": "First realistic end-to-end Stage-7 gate for a frozen pilot set.",
        "gates": {
            "toucher_track_accuracy": {
                "minimum": 0.70,
                "min_cases": 30,
                "basis": "EXTERNAL_INSPIRED",
                "note": "FOOTPASS official TAAD+DST validation Micro F1 is 0.675 on the harder who/what/when task.",
            },
            "attacking_team_accuracy": {
                "minimum": 0.85,
                "min_cases": 30,
                "basis": "EXTERNAL_INSPIRED",
                "note": "Below the 92.00 GS-HOTA team-side oracle ablation because Stage 7 also depends on toucher and team handoffs.",
            },
            "attack_direction_accuracy": {
                "minimum": 0.90,
                "min_cases": 30,
                "basis": "INTERNAL_ONLY",
                "note": "No directly comparable public benchmark was found. Requires independent direction GT; do not derive GT from centre-ray sign.",
            },
            "participant_set_micro_f1": {
                "minimum": 0.90,
                "min_cases": 30,
                "basis": "INTERNAL_WITH_EXTERNAL_CONTEXT",
                "note": "Role/team attributes are tractable in modern GSR, but exact set construction is stricter than attribute-only scoring.",
            },
            "referee_exclusion_f1": {
                "minimum": 0.90,
                "min_cases": 10,
                "basis": "INTERNAL_WITH_EXTERNAL_CONTEXT",
                "note": "Applies only to cases with independently labeled referee sets.",
            },
            "full_game_state_exact_match": {
                "minimum": 0.50,
                "min_cases": 30,
                "basis": "INTERNAL_COMPOSITE",
                "note": "Exact-match is conjunctive across multiple upstream-dependent fields, so the pilot floor is intentionally lower than component targets.",
            },
            "invariant_pass_rate": {
                "minimum": 1.00,
                "min_cases": 1,
                "basis": "DETERMINISTIC_SOFTWARE_CONTRACT",
                "note": "Any invariant failure is an implementation defect, not a model-accuracy trade-off.",
            },
        },
    },
    "research-target": {
        "description": "Stronger target for a larger frozen independent test set after pilot tuning is frozen.",
        "gates": {
            "toucher_track_accuracy": {
                "minimum": 0.80,
                "min_cases": 100,
                "basis": "EXTERNAL_INSPIRED",
                "note": "Manual t0 removes temporal spotting uncertainty, so a target above the 0.675 official FOOTPASS baseline is reasonable.",
            },
            "attacking_team_accuracy": {
                "minimum": 0.90,
                "min_cases": 100,
                "basis": "EXTERNAL_INSPIRED",
                "note": "Kept below the related 92.00 team-side oracle GS-HOTA reference.",
            },
            "attack_direction_accuracy": {
                "minimum": 0.95,
                "min_cases": 100,
                "basis": "INTERNAL_ONLY",
                "note": "Must be tested on independent semantics; current centre-ray rule is a project scope assumption, not a universal football law.",
            },
            "participant_set_micro_f1": {
                "minimum": 0.95,
                "min_cases": 100,
                "basis": "INTERNAL_WITH_EXTERNAL_CONTEXT",
                "note": "Requires high-quality team and role inputs because one player error changes downstream defender/attacker sets.",
            },
            "referee_exclusion_f1": {
                "minimum": 0.95,
                "min_cases": 30,
                "basis": "INTERNAL_WITH_EXTERNAL_CONTEXT",
                "note": "Referee leakage is especially harmful because it can corrupt the opponent ranking in Stage 8.",
            },
            "full_game_state_exact_match": {
                "minimum": 0.65,
                "min_cases": 100,
                "basis": "INTERNAL_COMPOSITE",
                "note": "Conjunctive metric; not numerically comparable to GS-HOTA.",
            },
            "invariant_pass_rate": {
                "minimum": 1.00,
                "min_cases": 1,
                "basis": "DETERMINISTIC_SOFTWARE_CONTRACT",
                "note": "Hard software gate.",
            },
        },
    },
}


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else float(a) / float(b)


def _resolve(base: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def _norm_set(value: Any) -> Optional[set[str]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"Expected list-like set label, got {type(value).__name__}")
    return {str(x) for x in value}


def _update_acc(acc: Dict[str, Dict[str, int]], name: str, eligible: bool, correct: bool) -> None:
    row = acc.setdefault(name, {"correct": 0, "eligible": 0})
    if eligible:
        row["eligible"] += 1
        row["correct"] += int(bool(correct))


def _metric_from_acc(acc: Dict[str, Dict[str, int]], name: str) -> Dict[str, Any]:
    row = acc.get(name, {"correct": 0, "eligible": 0})
    return {
        "value": _safe_div(row["correct"], row["eligible"]),
        "correct": int(row["correct"]),
        "eligible": int(row["eligible"]),
    }


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    # All missed or all incorrect predictions have F1=0, not missing GT.
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _full_gt_available(gt: Mapping[str, Any]) -> bool:
    sets = gt.get("sets") if isinstance(gt.get("sets"), Mapping) else {}
    required = [
        "status" in gt,
        "toucher_track_id" in gt,
        "attacking_team_id" in gt,
        "attack_direction_s" in gt,
        "attackers" in sets,
        "opponents" in sets,
        "referees_excluded" in sets,
    ]
    return all(required)


def _case_independent(case: Mapping[str, Any]) -> bool:
    provenance = case.get("provenance") if isinstance(case.get("provenance"), Mapping) else {}
    return provenance.get("independent_gt") is True


def _evaluate_rows(cases: List[Mapping[str, Any]], manifest_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    acc_all: Dict[str, Dict[str, int]] = {}
    acc_ind: Dict[str, Dict[str, int]] = {}
    reason_counter: Counter[str] = Counter()

    participant_counts = {
        "all": {"tp": 0, "fp": 0, "fn": 0, "eligible_cases": 0},
        "independent": {"tp": 0, "fp": 0, "fn": 0, "eligible_cases": 0},
    }
    referee_counts = {
        "all": {"tp": 0, "fp": 0, "fn": 0, "eligible_cases": 0},
        "independent": {"tp": 0, "fp": 0, "fn": 0, "eligible_cases": 0},
    }

    pred_valid = 0
    pred_toucher_present = 0
    independent_cases = 0

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", f"case_{index:04d}"))
        base = manifest_path.parent
        ctx = build_game_state_context(
            _resolve(base, str(case["stage1"])),
            _resolve(base, str(case["stage5"])),
            _resolve(base, str(case["stage6"])),
        ).to_dict()
        gt = case.get("gt") if isinstance(case.get("gt"), Mapping) else {}
        independent = _case_independent(case)
        independent_cases += int(independent)
        if ctx.get("status") == "VALID":
            pred_valid += 1
        pred_toucher_id = (ctx.get("toucher") or {}).get("track_id")
        pred_toucher_present += int(pred_toucher_id is not None)
        for reason in ctx.get("reasons", []) or []:
            reason_counter[str(reason)] += 1

        pred_sets = ctx.get("sets") if isinstance(ctx.get("sets"), Mapping) else {}
        gt_sets = gt.get("sets") if isinstance(gt.get("sets"), Mapping) else {}
        checks: Dict[str, Optional[bool]] = {}

        def record(name: str, eligible: bool, correct: bool) -> None:
            _update_acc(acc_all, name, eligible, correct)
            if independent:
                _update_acc(acc_ind, name, eligible, correct)
            checks[name] = bool(correct) if eligible else None

        record("status_accuracy", "status" in gt, ctx.get("status") == gt.get("status"))
        record("frame_alignment_accuracy", "frame_index" in gt, ctx.get("frame_index") == gt.get("frame_index"))
        record("toucher_track_accuracy", "toucher_track_id" in gt, pred_toucher_id == (None if gt.get("toucher_track_id") is None else str(gt.get("toucher_track_id"))))
        record("attacking_team_accuracy", "attacking_team_id" in gt, ctx.get("attacking_team_id") == gt.get("attacking_team_id"))
        pred_s = (ctx.get("attack_direction") or {}).get("s")
        record("attack_direction_accuracy", "attack_direction_s" in gt, pred_s == gt.get("attack_direction_s"))

        for metric_name, set_key in (
            ("attacker_set_exact_match", "attackers"),
            ("opponent_set_exact_match", "opponents"),
            ("referee_exclusion_exact_match", "referees_excluded"),
        ):
            eligible = set_key in gt_sets
            correct = False
            if eligible:
                correct = {str(x) for x in pred_sets.get(set_key, [])} == (_norm_set(gt_sets.get(set_key)) or set())
            record(metric_name, eligible, correct)

        inv_ok = bool((ctx.get("diagnostics") or {}).get("invariant_pass", False))
        record("invariant_pass_rate", True, inv_ok)

        full_eligible = _full_gt_available(gt)
        full_ok = False
        if full_eligible:
            full_ok = all(
                checks.get(name) is True
                for name in (
                    "status_accuracy",
                    "toucher_track_accuracy",
                    "attacking_team_accuracy",
                    "attack_direction_accuracy",
                    "attacker_set_exact_match",
                    "opponent_set_exact_match",
                    "referee_exclusion_exact_match",
                )
            )
        record("full_game_state_exact_match", full_eligible, full_ok)

        # Tagged participant labels avoid accidentally crediting a player placed
        # in the wrong set. A track predicted attacker but GT opponent contributes
        # one false positive and one false negative.
        participant_eligible = "attackers" in gt_sets and "opponents" in gt_sets
        if participant_eligible:
            pred_tagged = {f"A:{x}" for x in pred_sets.get("attackers", [])} | {f"O:{x}" for x in pred_sets.get("opponents", [])}
            gt_tagged = {f"A:{x}" for x in (_norm_set(gt_sets.get("attackers")) or set())} | {f"O:{x}" for x in (_norm_set(gt_sets.get("opponents")) or set())}
            for bucket in (["all", "independent"] if independent else ["all"]):
                participant_counts[bucket]["tp"] += len(pred_tagged & gt_tagged)
                participant_counts[bucket]["fp"] += len(pred_tagged - gt_tagged)
                participant_counts[bucket]["fn"] += len(gt_tagged - pred_tagged)
                participant_counts[bucket]["eligible_cases"] += 1

        referee_eligible = "referees_excluded" in gt_sets
        if referee_eligible:
            pred_ref = {str(x) for x in pred_sets.get("referees_excluded", [])}
            gt_ref = _norm_set(gt_sets.get("referees_excluded")) or set()
            for bucket in (["all", "independent"] if independent else ["all"]):
                referee_counts[bucket]["tp"] += len(pred_ref & gt_ref)
                referee_counts[bucket]["fp"] += len(pred_ref - gt_ref)
                referee_counts[bucket]["fn"] += len(gt_ref - pred_ref)
                referee_counts[bucket]["eligible_cases"] += 1

        rows.append({
            "case_id": case_id,
            "dataset": (case.get("provenance") or {}).get("dataset") if isinstance(case.get("provenance"), Mapping) else None,
            "independent_gt": independent,
            "status_pred": ctx.get("status"),
            "status_gt": gt.get("status"),
            "checks": checks,
            "reasons": ctx.get("reasons", []),
            "prediction_summary": {
                "frame_index": ctx.get("frame_index"),
                "toucher_track_id": pred_toucher_id,
                "attacking_team_id": ctx.get("attacking_team_id"),
                "attack_direction_s": pred_s,
                "sets": pred_sets,
            },
        })

    def metrics_for(acc: Dict[str, Dict[str, int]], bucket: str) -> Dict[str, Any]:
        names = [
            "status_accuracy",
            "frame_alignment_accuracy",
            "toucher_track_accuracy",
            "attacking_team_accuracy",
            "attack_direction_accuracy",
            "attacker_set_exact_match",
            "opponent_set_exact_match",
            "referee_exclusion_exact_match",
            "full_game_state_exact_match",
            "invariant_pass_rate",
        ]
        out = {name: _metric_from_acc(acc, name) for name in names}
        p = participant_counts[bucket]
        r = referee_counts[bucket]
        p_prf = _prf(p["tp"], p["fp"], p["fn"])
        r_prf = _prf(r["tp"], r["fp"], r["fn"])
        out["participant_set_micro_precision"] = {"value": p_prf["precision"], "eligible": p["eligible_cases"]}
        out["participant_set_micro_recall"] = {"value": p_prf["recall"], "eligible": p["eligible_cases"]}
        out["participant_set_micro_f1"] = {"value": p_prf["f1"], "eligible": p["eligible_cases"], **{k: p_prf[k] for k in ("tp", "fp", "fn")}}
        out["referee_exclusion_precision"] = {"value": r_prf["precision"], "eligible": r["eligible_cases"]}
        out["referee_exclusion_recall"] = {"value": r_prf["recall"], "eligible": r["eligible_cases"]}
        out["referee_exclusion_f1"] = {"value": r_prf["f1"], "eligible": r["eligible_cases"], **{k: r_prf[k] for k in ("tp", "fp", "fn")}}
        return out

    n = len(cases)
    summary = {
        "cases": n,
        "independent_gt_cases": independent_cases,
        "coverage": {
            "pred_valid_rate": _safe_div(pred_valid, n),
            "toucher_handoff_presence_rate": _safe_div(pred_toucher_present, n),
            "abstention_rate": _safe_div(n - pred_valid, n),
        },
        "metrics_all_labeled": metrics_for(acc_all, "all"),
        "metrics_independent_gt": metrics_for(acc_ind, "independent"),
        "unresolved_reason_distribution": dict(sorted(reason_counter.items())),
    }
    return rows, summary


def evaluate_manifest_v2(
    manifest_path: str,
    output_dir: str,
    *,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    mpath = Path(manifest_path).expanduser().resolve()
    payload = json.loads(mpath.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Manifest must contain a 'cases' list or be a list")
    normalized_cases = [c for c in cases if isinstance(c, Mapping)]
    rows, summary = _evaluate_rows(normalized_cases, mpath)

    gate_report: Dict[str, Any] = {
        "profile": profile,
        "status": "NOT_REQUESTED" if profile is None else "UNKNOWN_PROFILE",
        "gates": {},
        "semantics": "Engineering gates apply to independent GT only. They are not official third-party benchmark thresholds.",
    }
    if profile is not None:
        if profile not in ENGINEERING_PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; choose one of {sorted(ENGINEERING_PROFILES)}")
        config = ENGINEERING_PROFILES[profile]
        any_fail = False
        any_incomplete = False
        metrics = summary["metrics_independent_gt"]
        for name, gate in config["gates"].items():
            metric = metrics.get(name, {})
            value = metric.get("value")
            eligible = int(metric.get("eligible", 0) or 0)
            min_cases = int(gate["min_cases"])
            if value is None or eligible < min_cases:
                status = "NOT_EVALUATED"
                any_incomplete = True
            elif float(value) >= float(gate["minimum"]):
                status = "PASS"
            else:
                status = "BELOW_TARGET"
                any_fail = True
            gate_report["gates"][name] = {
                "status": status,
                "value": value,
                "eligible": eligible,
                **gate,
            }
        gate_report["description"] = config["description"]
        gate_report["status"] = "BELOW_TARGET" if any_fail else ("INCOMPLETE" if any_incomplete else "PASS")

    report = {
        "schema_version": SCHEMA_VERSION,
        "stage_version": STAGE_VERSION,
        "status": "COMPLETE",
        "manifest": str(mpath),
        "protocol": payload.get("protocol", {}) if isinstance(payload, dict) else {},
        "summary": summary,
        "acceptance": gate_report,
        "third_party_reference_ids": [x["id"] for x in THIRD_PARTY_REFERENCES],
        "rows": rows,
    }
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_benchmark_v2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def reference_report() -> Dict[str, Any]:
    return {
        "schema_version": "stage7-third-party-reference-1.0",
        "comparison_warning": (
            "GS-HOTA and FOOTPASS F1 are related-task references, not numerically equivalent to Stage-7 exact-match metrics. "
            "Only the oracle logic target of 1.0 is a hard algorithmic standard."
        ),
        "third_party_references": THIRD_PARTY_REFERENCES,
        "engineering_profiles": ENGINEERING_PROFILES,
        "public_gt_mapping": {
            "SoccerNet-GSR": {
                "supports": ["role", "team affiliation/team side", "track identity", "pitch position"],
                "stage7_use": ["participant team/role quality", "referee exclusion context"],
                "does_not_directly_support": ["toucher at manual t0", "attacking team from possession", "offside attack direction"],
            },
            "FOOTPASS": {
                "supports": ["event frame", "event team", "event jersey/player identity", "tracking/tactical context"],
                "stage7_use": ["actor/toucher-related evaluation", "attacking-team-at-event evaluation with identity mapping"],
                "does_not_directly_support": ["referee set GT", "verified Stage-7 attack-direction semantics"],
            },
        },
    }


def write_reference_report(output_dir: str) -> Dict[str, Any]:
    report = reference_report()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_third_party_reference.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def manifest_template() -> Dict[str, Any]:
    return {
        "schema_version": "stage7-eval-manifest-2.0",
        "protocol": {
            "name": "stage7-frozen-context-eval",
            "split": "locked_test",
            "notes": "GT fields may be partial. Metrics are evaluated only where the corresponding independent label exists.",
        },
        "cases": [
            {
                "case_id": "case001",
                "stage1": "relative/path/to/stage1.json",
                "stage5": "relative/path/to/stage5.json",
                "stage6": "relative/path/to/stage6.json",
                "provenance": {
                    "dataset": "PROJECT_FROZEN",
                    "independent_gt": True,
                    "gt_source": "manual_double_review_or_public_annotation",
                },
                "gt": {
                    "frame_index": 104,
                    "status": "VALID",
                    "toucher_track_id": "track_006",
                    "attacking_team_id": 1,
                    "attack_direction_s": 1,
                    "sets": {
                        "attackers": ["track_006", "track_008"],
                        "opponents": ["track_001", "track_002"],
                        "referees_excluded": ["track_003"],
                    },
                },
            },
            {
                "case_id": "partial_public_gt_example",
                "stage1": "relative/path/to/stage1.json",
                "stage5": "relative/path/to/stage5.json",
                "stage6": "relative/path/to/stage6.json",
                "provenance": {
                    "dataset": "FOOTPASS",
                    "independent_gt": True,
                    "gt_source": "public_event_annotation_plus_track_identity_map",
                },
                "gt": {
                    "frame_index": 12345,
                    "attacking_team_id": 0,
                },
            },
        ],
    }


def run_oracle(output_dir: str) -> Dict[str, Any]:
    return run_oracle_benchmark(output_dir)
